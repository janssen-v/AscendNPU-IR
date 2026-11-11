#!/usr/bin/env python3
"""Isolated CANN ACL runner for the explicit three-pointer f32 addition ABI."""

import argparse
import ctypes as C
import json
from pathlib import Path
import sys


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
            "aclrtBinaryLoadFromFile": [C.c_char_p, p, pp],
            "aclrtBinaryUnLoad": [p],
            "aclrtBinaryGetFunction": [p, C.c_char_p, pp],
            "aclrtLaunchKernel": [p, C.c_uint32, p, C.c_size_t, p],
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
        self.call("aclrtBinaryLoadFromFile", str(binary).encode(), None, C.byref(handle))
        self.binaries.append(handle)
        self.call("aclrtBinaryGetFunction", handle, entry.encode(), C.byref(function))
        return function

    def launch(self, function, pointers, blocks):
        arguments = (C.c_void_p * len(pointers))(*(p.value for p in pointers))
        self.call("aclrtLaunchKernel", function, blocks, arguments,
                  C.sizeof(arguments), self.stream)
        self.call("aclrtSynchronizeStream", self.stream)

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


def run(request):
    runtime = ACL(request["acl_library"], request["device"])
    try:
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
