from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

assert torch.cuda.is_available(), "CUDA is not available."

setup(
    name="cuda_rasterizer",
    ext_modules=[
        CUDAExtension(
            name="cuda_rasterizer",
            sources=[
                "src/cuda/rasterizer.cpp",
                "src/cuda/kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
