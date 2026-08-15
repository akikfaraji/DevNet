#!/usr/bin/env python3
"""Fix all __restrict__ to RESTRICT in C++ files.

This is a developer helper script. It patches the C++ kernel sources under
``xorzen/speed/csrc/`` to use a portable ``RESTRICT`` macro instead of the
MSVC-specific ``__restrict__`` keyword.

Paths are resolved relative to this script's location so the script works
from any checkout — no machine-specific absolute paths.
"""
from pathlib import Path

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"

files = [
    _CSRC_DIR / "attention_ops.cpp",
    _CSRC_DIR / "expert_dispatch.cpp",
    _CSRC_DIR / "math_utils.cpp",
    _CSRC_DIR / "router_ops.cpp",
    _CSRC_DIR / "ssm_scan.cpp",
]

macro = """
// Cross-platform restrict keyword
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif
"""

for filepath in files:
    print(f"Processing: {filepath}")
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add macro after #include "xorzen_kernels.h"
    if macro.strip() not in content:
        content = content.replace(
            '#include "xorzen_kernels.h"',
            f'#include "xorzen_kernels.h"{macro}'
        )
    
    # Replace all __restrict__
    content = content.replace('__restrict__', 'RESTRICT')
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed")

print("\nAll C++ files fixed!")
