#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# cxx_args = ['-std=c++11']

# nvcc_args = [
#     '-gencode', 'arch=compute_50,code=sm_50',
#     '-gencode', 'arch=compute_52,code=sm_52',
#     '-gencode', 'arch=compute_60,code=sm_60',
# #    '-gencode', 'arch=compute_61,code=sm_61',
# #    '-gencode', 'arch=compute_70,code=sm_70',
# #    '-gencode', 'arch=compute_70,code=compute_70'
# '-gencode', 'arch=compute_61,code=sm_61'
# ]

setup(
    name='mtlu_cuda',
    ext_modules=[
        CUDAExtension('mtlu_cuda', [
            'mtlu_cuda.cc',
            'mtlu_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
