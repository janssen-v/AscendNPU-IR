#!/usr/bin/env python3
"""Isolated CANN ACL runner for the explicit three-pointer f32 addition ABI."""

import argparse
import ctypes as C
import json
import math
from pathlib import Path
import sys


class BinaryLoadOptionValue(C.Union):
    _fields_ = [("magic", C.c_uint32), ("reserved", C.c_uint32 * 4)]


class BinaryLoadOption(C.Structure):
    _fields_ = [("type", C.c_int), ("value", BinaryLoadOptionValue)]


class BinaryLoadOptions(C.Structure):
    _fields_ = [("options", C.POINTER(BinaryLoadOption)), ("num_options", C.c_size_t)]


class ACL:
    def __init__(self, library, device):
        self.lib = C.CDLL(library)
        self.stream = C.c_void_p()
        self.buffers = []
        self.binaries = []
        self.initialized = False
        self.device = device
        self.device_set = False
        p = C.c_void_p
        pp = C.POINTER(p)
        self.signatures = {
            "aclInit": [C.c_char_p], "aclFinalize": [],
            "aclrtSetDevice": [C.c_int32], "aclrtResetDevice": [C.c_int32],
            "aclrtCreateStream": [pp], "aclrtDestroyStream": [p],
            "aclrtSynchronizeStream": [p],
            "aclrtMalloc": [pp, C.c_size_t, C.c_int], "aclrtFree": [p],
            "aclrtMemcpy": [p, C.c_size_t, p, C.c_size_t, C.c_int],
            "aclrtBinaryLoadFromData": [p, C.c_size_t, p, pp],
            "aclrtBinaryUnLoad": [p],
            "aclrtBinaryGetFunction": [p, C.c_char_p, pp],
            "aclrtLaunchKernelWithHostArgs": [p, C.c_uint32, p, p, p,
                                               C.c_size_t, p, C.c_size_t],
            "aclrtCreateEventWithFlag": [pp, C.c_uint32],
            "aclrtDestroyEvent": [p], "aclrtRecordEvent": [p, p],
            "aclrtEventElapsedTime": [C.POINTER(C.c_float), p, p],
        }
        for name, args in self.signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = C.c_int
        try:
            self.call("aclInit", None)
            self.initialized = True
            self.call("aclrtSetDevice", device)
            self.device_set = True
            self.call("aclrtCreateStream", C.byref(self.stream))
        except Exception:
            self.close()
            raise

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with ACL error {code}")

    def allocate(self, data):
        pointer = C.c_void_p()
        self.call("aclrtMalloc", C.byref(pointer), len(data), 2)
        self.buffers.append(pointer)
        host = C.create_string_buffer(data)
        self.call("aclrtMemcpy", pointer, len(data), host, len(data), 1)
        return pointer

    def load(self, binary, entry):
        handle = C.c_void_p()
        function = C.c_void_p()
        # This runner executes standalone AIV kernels. CANN's file loader does
        # not accept a core-type override and defaults to AIC. Match the
        # Triton-Ascend loader: eager loading plus ELF_VECTOR_CORE magic.
        options = (BinaryLoadOption * 2)(
            BinaryLoadOption(1, BinaryLoadOptionValue(magic=0)),
            BinaryLoadOption(2, BinaryLoadOptionValue(magic=0x41415246)))
        configuration = BinaryLoadOptions(options, len(options))
        data = binary.read_bytes()
        buffer = C.create_string_buffer(data)
        self.call("aclrtBinaryLoadFromData", buffer, len(data),
                  C.byref(configuration), C.byref(handle))
        self.binaries.append(handle)
        self.call("aclrtBinaryGetFunction", handle, entry.encode(), C.byref(function))
        return function

    def launch(self, function, pointers, blocks, synchronize=True):
        arguments = (C.c_void_p * len(pointers))(*(p.value for p in pointers))
        # arguments lives in host memory; the device-argument API would pass
        # this host address to the kernel, causing an invalid GM access.
        self.call("aclrtLaunchKernelWithHostArgs", function, blocks, self.stream,
                  None, arguments, C.sizeof(arguments), None, 0)
        if synchronize:
            self.call("aclrtSynchronizeStream", self.stream)

    def measure(self, function, pointers, blocks, iterations):
        events = []
        try:
            for _ in range(2):
                event = C.c_void_p()
                self.call("aclrtCreateEventWithFlag", C.byref(event), 8)
                events.append(event)
            self.call("aclrtRecordEvent", events[0], self.stream)
            for _ in range(iterations):
                self.launch(function, pointers, blocks, synchronize=False)
            self.call("aclrtRecordEvent", events[1], self.stream)
            self.call("aclrtSynchronizeStream", self.stream)
            elapsed = C.c_float()
            self.call("aclrtEventElapsedTime", C.byref(elapsed), *events)
            if not math.isfinite(elapsed.value) or elapsed.value <= 0:
                raise RuntimeError("invalid or unresolved ACL event timing")
            return elapsed.value / iterations
        finally:
            for event in events:
                self.lib.aclrtDestroyEvent(event)

    def read(self, pointer, size):
        host = C.create_string_buffer(size)
        self.call("aclrtMemcpy", host, size, pointer, size, 2)
        return host.raw

    def release_buffers(self):
        for pointer in reversed(self.buffers):
            self.call("aclrtFree", pointer)
        self.buffers.clear()

    def close(self):
        # Preserve the original execution exception; cleanup is best effort in
        # this disposable process, which the parent can terminate on timeout.
        for pointer in reversed(self.buffers):
            self.lib.aclrtFree(pointer)
        for handle in reversed(self.binaries):
            self.lib.aclrtBinaryUnLoad(handle)
        if self.stream.value:
            self.lib.aclrtDestroyStream(self.stream)
        if self.device_set:
            self.lib.aclrtResetDevice(self.device)
        if self.initialized:
            self.lib.aclFinalize()


def benchmark(runtime, request):
    functions, pointers = {}, {}
    lhs, rhs = (Path(p).read_bytes() for p in request["inputs"])
    if len(lhs) != len(rhs) or len(lhs) != request["bytes"]:
        raise ValueError("benchmark input sizes do not match ABI")
    for label in ("baseline", "candidate"):
        functions[label] = runtime.load(Path(request["binaries"][label]), request["entry"])
        pointers[label] = [runtime.allocate(lhs), runtime.allocate(rhs),
                           runtime.allocate(b"\xff" * len(lhs))]
    for _ in range(request["warmups"]):
        for label in ("baseline", "candidate"):
            runtime.launch(functions[label], pointers[label], request["blocks"])
    samples = []
    for round_index in range(request["rounds"]):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for label in order:
            milliseconds = runtime.measure(functions[label], pointers[label],
                                           request["blocks"], request["iterations"])
            samples.append({"round": round_index, "variant": label, "milliseconds": milliseconds})
            # Incremental raw samples survive a later device error or timeout.
            Path(request["samples"]).write_text(json.dumps(samples) + "\n")
    return samples


def run(request):
    runtime = ACL(request["acl_library"], request["device"])
    try:
        if request.get("mode") == "benchmark":
            benchmark(runtime, request)
            return
        function = runtime.load(Path(request["binary"]), request["entry"])
        for case in request["cases"]:
            lhs, rhs = (Path(p).read_bytes() for p in case["inputs"])
            if len(lhs) != len(rhs) or len(lhs) != request["bytes"]:
                raise ValueError("input sizes do not match the ABI")
            pointers = [runtime.allocate(lhs), runtime.allocate(rhs),
                        runtime.allocate(b"\xff" * len(lhs))]
            runtime.launch(function, pointers, request["blocks"])
            Path(case["output"]).write_bytes(runtime.read(pointers[2], len(lhs)))
            runtime.release_buffers()
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(json.loads(args.request.read_text()))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
