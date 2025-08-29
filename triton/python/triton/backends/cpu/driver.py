import functools
import operator
import os
import subprocess
import triton
import re
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

# dirname = os.path.dirname(os.path.realpath(__file__))
# include_dirs = [os.path.join(dirname, "include")]
# libdevice_dir = os.path.join(dirname, "lib")
# libraries = ['cuda']

class CpuUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CpuUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass


class CpuDriver(DriverBase):

    def __init__(self):
        self.utils = CpuUtils()  # TODO: make static
        self.launcher_cls = None
        super().__init__()

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        import torch
        return torch.device("cpu")

    def get_current_target(self):
        return GPUTarget("cpu", 0, 32)

    def get_active_torch_device(self):
        return self.get_current_device()

    def get_device_interface(self):
        import torch
        return torch.device

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()

