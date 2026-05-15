"""
Build script for the optional kvboost_flash_attn CUDA extension.

Install with CUDA support:
    pip install -e ".[cuda]"

Install CPU-only (skips CUDA extension, Python fallback used):
    pip install -e .
"""

import os
import sys

from setuptools import setup

# Only attempt to build CUDA extension when torch is available and CUDA is present.
ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        nvcc_arches = [
            # Ampere + Hopper
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            # Turing (T4 etc.)
            "-gencode=arch=compute_75,code=sm_75",
            # Volta
            "-gencode=arch=compute_70,code=sm_70",
        ]
        ext_modules = [
            CUDAExtension(
                name="kvboost._flash_attn_cuda",
                sources=[
                    "src/kvboost/csrc/flash_attn.cpp",
                    "src/kvboost/csrc/flash_attn_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "--use_fast_math",
                        *nvcc_arches,
                    ],
                },
                include_dirs=[
                    os.path.join(os.path.dirname(__file__), "src", "kvboost", "csrc"),
                ],
            ),
            CUDAExtension(
                name="kvboost._streaming_glue",
                sources=[
                    "src/kvboost/streaming/csrc/streaming_glue.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        # Streaming glue currently only targets sm_80+.
                        "-gencode=arch=compute_80,code=sm_80",
                        "-gencode=arch=compute_86,code=sm_86",
                        "-gencode=arch=compute_89,code=sm_89",
                        "-gencode=arch=compute_90,code=sm_90",
                    ],
                },
                include_dirs=[
                    os.path.join(os.path.dirname(__file__), "src", "kvboost", "streaming", "csrc"),
                ],
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
        print("[kvboost] CUDA extensions will be built (flash_attn + streaming_glue).", file=sys.stderr)
    else:
        print("[kvboost] No CUDA device detected — skipping flash_attn extension.", file=sys.stderr)
except ImportError:
    print("[kvboost] torch not found — skipping flash_attn extension.", file=sys.stderr)

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
