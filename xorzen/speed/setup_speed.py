"""
setup_speed.py — Build Script for xorzen Lightning C++ Kernels
==============================================================
Compiles the Cython + C++ extension using MSVC or GCC/MinGW.

Usage
-----
    python setup_speed.py build_ext --inplace

After build, a .pyd file appears in xorzen/speed/ and the Python
wrappers automatically import it via `from . import xorzen_ext`.
"""

import os
import sys
import platform
from pathlib import Path

# ── Try to import build tools ─────────────────────────────────────
try:
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy as np
except ImportError as e:
    print(f"[setup_speed] Missing dependency: {e}")
    print("[setup_speed] Install with:  pip install cython numpy setuptools")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────
HERE       = Path(__file__).parent.resolve()
CSRC       = HERE / "csrc"
NUMPY_INC  = np.get_include()

# ── Detect compiler ───────────────────────────────────────────────
def detect_compiler():
    """Detect which compiler will be used."""
    if sys.platform == "win32":
        # Check if MSVC or MinGW
        import distutils.ccompiler
        compiler = distutils.ccompiler.new_compiler()
        if hasattr(compiler, 'compiler_type'):
            return compiler.compiler_type
        # Default to msvc on Windows
        return 'msvc'
    else:
        return 'unix'

COMPILER = detect_compiler()
print(f"[setup_speed] Detected compiler: {COMPILER}")

# ── Build flags based on compiler ─────────────────────────────────
if COMPILER == 'msvc':
    # MSVC flags
    COMPILE_ARGS = [
        '/O2',           # Optimize for speed
        '/std:c++17',    # C++17 standard
        '/fp:fast',      # Fast floating point
        '/arch:AVX2',    # AVX2 instructions
        '/openmp:experimental', # OpenMP + SIMD support (supersedes /openmp)
        '/D_USE_MATH_DEFINES',
        '/DNOMINMAX',    # Prevent min/max macros
    ]
    LINK_ARGS = []
    
    print("[setup_speed]   MSVC flags: /O2 /std:c++17 /arch:AVX2 /openmp:experimental")
    
else:
    # GCC/MinGW flags
    COMPILE_ARGS = [
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-fno-wrapv",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-fopenmp",
    ]
    LINK_ARGS = ["-fopenmp"]
    
    if sys.platform == "win32":
        COMPILE_ARGS += ["-D_WIN32_WINNT=0x0601", "-D_USE_MATH_DEFINES"]
    
    print(f"[setup_speed]   GCC flags: {' '.join(COMPILE_ARGS)}")

# ── C++ source files ──────────────────────────────────────────────
CPP_SOURCES = [
    str(CSRC / "ssm_scan.cpp"),
    str(CSRC / "attention_ops.cpp"),
    str(CSRC / "router_ops.cpp"),
    str(CSRC / "expert_dispatch.cpp"),
    str(CSRC / "math_utils.cpp"),
    str(CSRC / "fused_ops.cpp"),      # NEW: RMSNorm, SwiGLU, diagonal SSM scan
]

# ── Extension definition ──────────────────────────────────────────
# Use a flat name so --inplace drops the .pyd right here in xorzen/speed/
# regardless of which directory the script is invoked from.
ext = Extension(
    name="xorzen_ext",
    sources=[
        str(HERE / "xorzen_ext.pyx"),
        *CPP_SOURCES,
    ],
    include_dirs=[
        str(HERE),
        str(CSRC),
        NUMPY_INC,
    ],
    language="c++",
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
)

# ── Run setup ─────────────────────────────────────────────────────
print("[setup_speed] Building xorzen Lightning C++ kernels...")

setup(
    name="xorzen_speed",
    version="1.0.0",
    description="xorzen Lightning Engine — C++/AVX2/OpenMP Kernels",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
        annotate=False,
    ),
    zip_safe=False,
)

# ── Copy .pyd to the speed package directory if needed ───────────
import shutil, glob
for pyd in glob.glob(str(HERE / "xorzen_ext*.pyd")):
    dest = HERE / Path(pyd).name
    if Path(pyd).resolve() != dest.resolve():
        shutil.copy2(pyd, dest)
        print(f"[setup_speed]   Copied {Path(pyd).name} → {dest}")

print("[setup_speed] ✓ Build complete.")
print("[setup_speed]   Import with: from xorzen.speed import xorzen_ext")
