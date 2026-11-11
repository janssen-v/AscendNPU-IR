#!/usr/bin/env python3
"""Validate 12 Triton kernels and prepare shared-checkpoint recipe experiments."""
import argparse
import difflib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import KERNELS, PROFILES, specification, make_case
from experiment import create, digest, replay, run_process, write_json
from triton_fused import entry_arguments
from evaluate import summarize_samples


def worker(args):
    import torch
    import torch_npu
    import triton
    import triton.backends.ascend.compiler as backend
    from triton.runtime import driver
    import kernels

    spec = specification(args.kernel, args.profile)
    out = args.output
    compiler = args.build_dir / 'bin/bishengir-compile'
    report = {'status': 'error', 'specification': spec, 'frontend': [],
              'performance_eligible': False, 'candidate_status': 'not_requested',
              'versions': {'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'triton': triton.__version__},
              'sources': {p.name: digest(p.read_bytes()) for p in (Path(__file__), Path(__file__).with_name('kernels.py'), Path(__file__).with_name('catalog.py'))}}
    try:
        torch.npu.set_device(args.device)
        report['device'] = torch.npu.get_device_name(args.device)
        report['frontend_adapter'] = {'path': backend.__file__, 'sha256': digest(Path(backend.__file__).read_bytes())}
        cases = []
        kernel = None
        for seed in spec['seeds']:
            inputs, reference, constants, grid = make_case(torch, spec, seed)
            device_inputs = [x.to('npu') for x in inputs]
            output = torch.full(reference.shape, float('nan'), device='npu', dtype=torch.float32)
            if kernel is None:
                old = backend._get_npucompiler_path
                backend._get_npucompiler_path = lambda: (str(compiler), dict(os.environ))
                try:
                    kernel = getattr(kernels, args.kernel).warmup(*device_inputs, output, *constants,
                        grid=grid, enable_fp_fusion=False, disable_fma=True,
                        enable_auto_bind_sub_block=False,
                        enable_preload=False, multibuffer=False)
                finally:
                    backend._get_npucompiler_path = old
                for kind in ('ttir', 'ttadapter'):
                    (out / ('kernel.' + kind)).write_text(kernel.asm[kind])
                write_json(out / 'metadata.json', json.loads(json.dumps(kernel.metadata._asdict(), default=str)))
                (out / 'kernel.py').write_bytes(Path(__file__).with_name('kernels.py').read_bytes())
                report['grid'] = grid
                report['constants'] = constants
                report['frontend_binary_sha256'] = digest(kernel.asm['npubin'])
            kernel[grid](*device_inputs, output, *constants)
            actual = output.cpu()
            verdict = compare(torch, reference, actual, spec)
            report['frontend'].append(dict(verdict, seed=seed))
            torch.save({'inputs': inputs, 'reference': reference, 'actual': actual}, out / f'frontend-{seed}.pt')
            cases.append((device_inputs, reference, constants, grid))
        if not all(v['passed'] for v in report['frontend']):
            report['status'] = 'frontend_incorrect'
            return report
        report['status'] = 'validated'
        if args.mode == 'validate':
            return report

        capture = out / 'capture'
        capture.mkdir()
        original = ROOT / 'examples/original-vectorizer.mlir'
        command = [str(compiler), str(out / 'kernel.ttadapter'), '--target=Ascend950PR_9579',
            '--enable-triton-kernel-compile=true', '--disable-ffts', '--enable-auto-blockify-loop',
            '--enable-hfusion-compile=true', '--disable-hfusion-vectorize=true', '--use-pipeline-version=2',
            '--hivm-pre-vectorize-checkpoint=checkpoint.bc', '--hivm-replace-vectorization',
            '--hivm-pre-vectorize-recipe=' + str(original), '--enable-preload=false',
            '--enable-auto-multi-buffer=false', '--enable-lir-compile=false', '-o', 'baseline']
        report['capture'] = dict(run_process(command, capture, dict(os.environ), args.timeout), command=command)
        checkpoint = capture / 'checkpoint.bc'
        if not checkpoint.exists():
            report['status'] = 'capture_failed'
            return report
        verify_dir = out / 'verify-checkpoint'
        verify_dir.mkdir()
        report['checkpoint_verification'] = run_process([str(args.build_dir / 'bin/bishengir-opt'),
            str(checkpoint), '-o', 'checkpoint.mlir'], verify_dir, dict(os.environ), args.timeout)
        if report['checkpoint_verification']['status'] != 'success':
            report['status'] = 'invalid_checkpoint'
            return report
        report['checkpoint_sha256'] = digest(checkpoint.read_bytes())
        candidate = args.recipe or (ROOT / 'examples' / spec['candidate_recipe'] if spec['candidate_recipe'] else None)
        recipes = {'baseline': original}
        if candidate:
            recipes['candidate'] = candidate
            report['comparison_kind'] = ('identical_recipe_control'
                if candidate.read_bytes() == original.read_bytes() else 'experimental_recipe')
        report['candidate_status'] = 'configured' if candidate else 'recipe_required'
        dependencies = list((args.build_dir / 'lib').glob('*.bc'))
        if not dependencies:
            raise ValueError('missing template bitcode')
        for label, recipe in recipes.items():
            manifest = create(out / (label + '-bundle'), checkpoint, recipe, compiler, ROOT.parents[1],
                'Ascend950PR_9579', '__transform_main', adapter='driver-binary',
                toolchain_bin=args.toolchain_bin, dependency=dependencies,
                replace_vectorization=True, triton_kernel=True,
                backend_options='-mllvm -cce-simt-fpmath-combine=false')
            assert manifest['inputs']['checkpoint.bc'] == report['checkpoint_sha256']
        report['status'] = 'captured' if report['capture']['status'] == 'success' else 'checkpoint_only'
        if args.mode == 'compare' and not candidate:
            report['status'] = 'recipe_required'
            return report
        if args.mode != 'compare':
            return report

        loaded = {}
        report['compilations'] = {}
        for label in recipes:
            run_dir = out / (label + '-compile')
            result = replay(out / (label + '-bundle'), run_dir, timeout=args.timeout)
            report['compilations'][label] = result
            if result['status'] != 'success':
                report['status'] = 'recipe_compile_failed'
                return report
            expected = ['ptr addrspace(1)'] * (len(cases[0][0]) + 3) + ['i32'] * 3
            actual = entry_arguments((run_dir / 'intermediates/candidate.ll').read_text(), kernel.metadata.kernel_name)
            if actual != expected or kernel.metadata.mix_mode != 'aiv':
                raise ValueError('replay ABI differs from supported AIV pointer-only launcher')
            loaded[label] = driver.active.utils.load_binary(kernel.metadata.kernel_name,
                (run_dir / 'candidate.o').read_bytes(), kernel.metadata.shared, args.device, kernel.metadata.mix_mode)
        for name in ('module.hivm.opt.mlir', 'candidate.ll'):
            texts = [(out / (label + '-compile') / 'intermediates' / name).read_text().splitlines(True)
                     for label in ('baseline', 'candidate')]
            (out / (name + '.diff')).write_text(''.join(difflib.unified_diff(
                *texts, fromfile='baseline/' + name, tofile='candidate/' + name)))
        report['correctness'] = {}
        for label in loaded:
            kernel.module, kernel.function = loaded[label][:2]
            results = []
            for seed, (inputs, reference, constants, grid) in zip(spec['seeds'], cases):
                output = torch.full(reference.shape, float('nan'), device='npu', dtype=torch.float32)
                kernel[grid](*inputs, output, *constants)
                actual = output.cpu()
                results.append(dict(compare(torch, reference, actual, spec), seed=seed))
                torch.save(actual, out / f'{label}-{seed}.pt')
            report['correctness'][label] = results
        report['performance_eligible'] = all(v['passed'] for values in report['correctness'].values() for v in values)
        if not report['performance_eligible']:
            report['status'] = 'recipe_incorrect'
            return report
        inputs, reference, constants, grid = cases[-1]
        output = torch.empty_like(reference, device='npu')
        def launch(label):
            kernel.module, kernel.function = loaded[label][:2]
            kernel[grid](*inputs, output, *constants)
        for _ in range(5):
            for label in loaded:
                launch(label)
        torch.npu.synchronize()
        samples = []
        for i in range(20):
            for label in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
                start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(10):
                    launch(label)
                end.record()
                end.synchronize()
                samples.append({'round': i, 'variant': label, 'milliseconds': start.elapsed_time(end) / 10})
                write_json(out / 'samples.json', samples)
        report['benchmark'] = dict(summarize_samples(samples, 20), warmups=5, rounds=20, iterations=10,
                                   clock='torch.npu.Event', unit='milliseconds per launch including scheduling gaps')
        report['status'] = 'compared'
        return report
    except Exception as error:
        report['status'] = 'error'
        report['diagnostic'] = str(error)
        traceback.print_exc()
        return report
    finally:
        report['artifacts'] = {str(p.relative_to(out)): digest(p.read_bytes()) for p in out.rglob('*')
            if p.is_file() and 'cache' not in p.relative_to(out).parts and p.name not in ('report.json', 'stdout.log', 'stderr.log')}
        write_json(out / 'report.json', report)


def compare(torch, expected, actual, spec):
    close = torch.isclose(actual, expected, atol=spec['atol'], rtol=spec['rtol']) & torch.isfinite(actual)
    return {'passed': bool(close.all()), 'mismatches': int((~close).sum()),
            'max_abs_error': float((actual - expected).abs().max()) if torch.isfinite(actual).all() else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--build-dir', type=Path, default=ROOT.parents[1] / 'build_a5')
    p.add_argument('--toolchain-bin', required=True, type=Path)
    p.add_argument('--kernels', nargs='+', choices=KERNELS, default=list(KERNELS))
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--mode', choices=('validate', 'capture', 'compare'), default='capture')
    p.add_argument('--recipe', type=Path, help='Candidate recipe; restrict --kernels when using a specialized recipe')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--kernel', choices=KERNELS, help=argparse.SUPPRESS)
    p.add_argument('--profile', choices=PROFILES, help=argparse.SUPPRESS)
    args = p.parse_args()
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.device < 0:
        p.error('invalid timeout/device')
    if len(set(args.kernels)) != len(args.kernels) or len(set(args.profiles)) != len(args.profiles):
        p.error('kernel/profile selections must be distinct')
    if args.worker and (args.kernel is None or args.profile is None):
        p.error('worker requires a kernel and profile')
    for key in ('output', 'build_dir', 'toolchain_bin', 'recipe'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        return 0 if worker(args)['status'] in ('validated', 'captured', 'checkpoint_only', 'compared') else 1
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'mode': args.mode, 'results': {}}
    for name in args.kernels:
        for profile in args.profiles:
            label = name + '-' + profile
            out = args.output / label
            out.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--kernel', name,
                '--profile', profile, '--output', str(out), '--build-dir', str(args.build_dir),
                '--toolchain-bin', str(args.toolchain_bin), '--mode', args.mode,
                '--device', str(args.device), '--timeout', str(args.timeout)]
            if args.recipe:
                command += ['--recipe', str(args.recipe)]
            env = dict(os.environ, BISHENG_INSTALL_PATH=str(args.toolchain_bin), TRITON_CACHE_DIR=str(out / 'cache'))
            env['PATH'] = str(args.toolchain_bin) + os.pathsep + env.get('PATH', '')
            execution = run_process(command, out, env, args.timeout)
            write_json(out / 'execution.json', dict(execution, command=command))
            report = json.loads((out / 'report.json').read_text()) if (out / 'report.json').exists() else {}
            summary['results'][label] = {'execution': execution, 'status': report.get('status', 'worker_failed'),
                'candidate_status': report.get('candidate_status'), 'checkpoint_sha256': report.get('checkpoint_sha256')}
            write_json(args.output / 'suite.json', summary)
            print(label, summary['results'][label]['status'], flush=True)
    return int(any(r['execution']['status'] != 'success' for r in summary['results'].values()))


if __name__ == '__main__':
    sys.exit(main())
