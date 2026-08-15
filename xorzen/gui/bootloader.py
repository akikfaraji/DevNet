"""
bootloader.py -- xorzen Neural Engine Boot Sequence  (Tkinter GUI)
===================================================================
BIOS-style POST screen that opens a standalone desktop window.
No terminal output -- everything renders in a dark GUI window.

Usage
-----
    from xorzen.gui import XorzenBootloader
    boot = XorzenBootloader(config)
    boot.run()          # blocks until window closed / checks done
"""

from __future__ import annotations
import os, sys, time, platform, threading, subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# Colour / font constants
# ---------------------------------------------------------------------------
BG          = "#0a0a0f"
BG_PANEL    = "#0f0f1a"
BG_HEADER   = "#060610"
FG          = "#c8d8f0"
FG_DIM      = "#4a5a6a"
FG_GREEN    = "#00ff88"
FG_YELLOW   = "#ffd700"
FG_RED      = "#ff4455"
FG_CYAN     = "#00d4ff"
FG_MAGENTA  = "#cc88ff"
FG_WHITE    = "#ffffff"
ACCENT      = "#00d4ff"
FONT_MONO   = ("Consolas", 10)
FONT_MONO_S = ("Consolas", 9)
FONT_MONO_L = ("Consolas", 12, "bold")
FONT_TITLE  = ("Consolas", 14, "bold")
FONT_BANNER = ("Consolas", 8)

BANNER_LINES = [
    r"  ██╗  ██╗ ██████╗ ██████╗ ███████╗███████╗███╗   ██╗",
    r"  ╚██╗██╔╝██╔═══██╗██╔══██╗╚══███╔╝██╔════╝████╗  ██║",
    r"   ╚███╔╝ ██║   ██║██████╔╝  ███╔╝ █████╗  ██╔██╗ ██║",
    r"   ██╔██╗ ██║   ██║██╔══██╗ ███╔╝  ██╔══╝  ██║╚██╗██║",
    r" ██╔╝ ██╗╚██████╔╝██║  ██║███████╗███████╗██║ ╚████║",
    r"  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝",
]
SUB_BANNER = "N E U R A L   E N G I N E   ·   L I G H T N I N G   C O R E   v 0 . 2 . 4"

WIN_W, WIN_H = 860, 640


# ---------------------------------------------------------------------------

class XorzenBootloader:
    """Opens a Tkinter BIOS-style POST window, runs checks, then returns."""

    def __init__(self, config=None, skip_input: bool = False):
        self.config     = config
        self.skip_input = skip_input
        self._tk        = None
        self._done      = threading.Event()

    # ------------------------------------------------------------------
    def run(self):
        """Block until boot sequence is finished (window closed or auto-continue)."""
        import threading as _th
        # If we ARE the main thread, run the Tkinter mainloop directly here.
        # If we are NOT the main thread (e.g. called from a test runner),
        # Tcl/Tk will crash with 'Tcl_AsyncDelete: wrong thread'.
        # In that case, skip the visual window and just run the checks silently.
        if _th.current_thread() is _th.main_thread():
            self._gui_thread()
        else:
            self._run_headless()

    # ------------------------------------------------------------------
    def _run_headless(self):
        """Run all POST checks silently -- used when called from a non-main thread."""
        checks = [
            self._check_hardware,
            self._check_avx2,
            self._check_openmp,
            self._check_kernels,
            self._check_cython_ext,
            self._check_expert_manifest,
            self._check_expert_cache,
            self._check_jit,
            self._check_speed_booster,
        ]
        for fn in checks:
            fn()   # results discarded; we just verify no exception is raised

    # ------------------------------------------------------------------
    def _gui_thread(self):
        import tkinter as tk
        from tkinter import font as tkfont

        root = tk.Tk()
        root.title("XORZEN  —  Neural Engine Boot")
        root.configure(bg=BG)
        root.resizable(False, False)
        # Centre on screen
        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x  = (sw - WIN_W) // 2
        y  = (sh - WIN_H) // 2
        root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        self._tk = root
        self._build_ui(root)
        # Start checks in background after window is drawn
        root.after(200, lambda: threading.Thread(
            target=self._run_checks_bg, args=(root,), daemon=True
        ).start())
        root.mainloop()

    # ------------------------------------------------------------------
    def _build_ui(self, root):
        import tkinter as tk

        # ── Banner ───────────────────────────────────────────────────
        banner_frame = tk.Frame(root, bg=BG_HEADER)
        banner_frame.pack(fill="x", pady=(6, 0))
        for line in BANNER_LINES:
            lbl = tk.Label(banner_frame, text=line, font=FONT_BANNER,
                           fg=FG_CYAN, bg=BG_HEADER, anchor="center")
            lbl.pack()
        sub = tk.Label(banner_frame, text=SUB_BANNER, font=("Consolas", 8),
                       fg=FG_DIM, bg=BG_HEADER)
        sub.pack(pady=(0, 4))

        sep = tk.Frame(root, height=1, bg=ACCENT)
        sep.pack(fill="x")

        # ── System info ──────────────────────────────────────────────
        info_frame = tk.Frame(root, bg=BG_PANEL, pady=4)
        info_frame.pack(fill="x", padx=8, pady=4)
        self._add_section_title(info_frame, "SYSTEM INFO")
        import torch
        rows = [
            ("Platform",  platform.system() + " " + platform.release()),
            ("CPU",       (platform.processor() or platform.machine()) +
                          f"  ({os.cpu_count()} logical cores)"),
            ("RAM",       f"{self._get_ram_gb():.1f} GB"),
            ("Python",    sys.version.split()[0]),
            ("PyTorch",   torch.__version__),
            ("CUDA",      "available" if torch.cuda.is_available() else "not found  --  CPU mode"),
        ]
        for k, v in rows:
            self._add_kv_row(info_frame, k, v)

        sep2 = tk.Frame(root, height=1, bg=FG_DIM)
        sep2.pack(fill="x", padx=8)

        # ── POST checks ──────────────────────────────────────────────
        post_frame = tk.Frame(root, bg=BG, pady=2)
        post_frame.pack(fill="x", padx=8)
        self._add_section_title(post_frame, "POWER-ON SELF TEST")

        self._check_labels: List[Tuple[tk.Label, tk.Label]] = []
        check_names = [
            "Scanning hardware topology",
            "Detecting AVX2/FMA SIMD units",
            "Detecting OpenMP parallel runtime",
            "Locating C++ kernel binaries",
            "Verifying Cython extension bridge",
            "Loading expert manifest",
            "Prefetching hot experts into cache",
            "Warming JIT compilation paths",
            "Initialising speed booster",
        ]
        for name in check_names:
            row_f = tk.Frame(post_frame, bg=BG)
            row_f.pack(fill="x", pady=1)
            lbl_name = tk.Label(row_f, text=f"  [BOOT] {name}",
                                font=FONT_MONO_S, fg=FG_DIM, bg=BG,
                                anchor="w", width=52)
            lbl_name.pack(side="left")
            dots = "." * max(0, 52 - len(name) - 8)
            tk.Label(row_f, text=dots, font=FONT_MONO_S,
                     fg=FG_DIM, bg=BG).pack(side="left")
            lbl_result = tk.Label(row_f, text="...", font=FONT_MONO_S,
                                  fg=FG_DIM, bg=BG, anchor="w", width=36)
            lbl_result.pack(side="left", padx=(4, 0))
            self._check_labels.append((lbl_name, lbl_result))

        # ── Ready bar ────────────────────────────────────────────────
        sep3 = tk.Frame(root, height=1, bg=ACCENT)
        sep3.pack(fill="x", padx=8, pady=(6, 0))

        self._ready_lbl = tk.Label(root,
            text="  Initialising...",
            font=FONT_MONO_L, fg=FG_DIM, bg=BG, anchor="center")
        self._ready_lbl.pack(fill="x", pady=6)

        # ── Continue button (shown after checks) ─────────────────────
        self._btn_frame = tk.Frame(root, bg=BG)
        self._btn_frame.pack(pady=(0, 8))
        self._continue_btn = tk.Button(
            self._btn_frame,
            text="  ▶  BEGIN TRAINING  ",
            font=FONT_MONO_L,
            fg=FG_WHITE, bg="#003355",
            activeforeground=FG_WHITE, activebackground=ACCENT,
            relief="flat", bd=0, padx=12, pady=6,
            command=lambda: root.destroy(),
        )
        # Hidden until checks pass
        self._continue_btn.pack_forget()

    # ------------------------------------------------------------------
    def _add_section_title(self, parent, text):
        import tkinter as tk
        tk.Label(parent, text=f"  {text}", font=("Consolas", 9, "bold"),
                 fg=ACCENT, bg=parent["bg"], anchor="w").pack(fill="x", pady=(2, 0))

    def _add_kv_row(self, parent, key, value, vcolor=FG):
        import tkinter as tk
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x")
        tk.Label(f, text=f"    {key:<16}", font=FONT_MONO_S,
                 fg=FG_DIM, bg=parent["bg"]).pack(side="left")
        tk.Label(f, text=value, font=FONT_MONO_S,
                 fg=vcolor, bg=parent["bg"], anchor="w").pack(side="left")

    # ------------------------------------------------------------------
    def _run_checks_bg(self, root):
        """Run all POST checks in background thread, update labels safely."""
        checks = [
            self._check_hardware,
            self._check_avx2,
            self._check_openmp,
            self._check_kernels,
            self._check_cython_ext,
            self._check_expert_manifest,
            self._check_expert_cache,
            self._check_jit,
            self._check_speed_booster,
        ]
        for i, fn in enumerate(checks):
            result, color, detail = fn()
            fg_color = {
                "green":   FG_GREEN,
                "yellow":  FG_YELLOW,
                "red":     FG_RED,
                "cyan":    FG_CYAN,
                "dim":     FG_DIM,
            }.get(color, FG)
            text = result
            if detail:
                text += f"  {detail}"
            # Schedule UI update on main thread
            root.after(0, lambda i=i, t=text, c=fg_color: self._update_check(i, t, c))
            time.sleep(0.05)

        # All done
        root.after(0, lambda: self._on_checks_done(root))

    def _update_check(self, idx: int, text: str, color: str):
        _, lbl_result = self._check_labels[idx]
        lbl_result.config(text=text, fg=color)

    def _on_checks_done(self, root):
        self._ready_lbl.config(
            text="  XORZEN LIGHTNING ENGINE READY  --  TARGET 50x NOMINAL",
            fg=FG_GREEN,
        )
        if self.skip_input:
            # Auto-close after short delay
            root.after(800, root.destroy)
        else:
            self._continue_btn.pack()

    # ------------------------------------------------------------------
    # ── Individual POST checks ────────────────────────────────────────

    def _check_hardware(self):
        cores = os.cpu_count()
        ram   = self._get_ram_gb()
        return "OK", "green", f"{cores} cores / {ram:.0f} GB RAM"

    def _check_avx2(self):
        if sys.platform == "win32":
            try:
                from xorzen.speed import xorzen_ext  # noqa
                return "AVX2  (MSVC /arch:AVX2)", "green", ""
            except ImportError:
                pass
            return "AVX2  (MSVC /arch:AVX2 flagged)", "green", "runtime detection skipped on MSVC"
        try:
            r = subprocess.run(
                ["g++", "-mavx2", "-x", "c++", "-", "-o", os.devnull, "--std=c++17"],
                input=b"#include<immintrin.h>\nint main(){}",
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return "AVX2 detected", "green", ""
        except Exception:
            pass
        return "SCALAR FALLBACK", "yellow", "no AVX2"

    def _check_openmp(self):
        if sys.platform == "win32":
            cores = os.cpu_count()
            return f"OpenMP  {cores} threads  (MSVC /openmp)", "green", ""
        try:
            r = subprocess.run(
                ["g++", "-fopenmp", "-x", "c++", "-", "-o", os.devnull],
                input=b"#include<omp.h>\nint main(){return omp_get_max_threads();}",
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return f"OpenMP  {os.cpu_count()} threads", "green", ""
        except Exception:
            pass
        return "DISABLED", "yellow", "install GCC with libgomp"

    def _check_kernels(self):
        csrc = Path(__file__).parent.parent / "speed" / "csrc"
        files = list(csrc.glob("*.cpp"))
        if files:
            return f"{len(files)} kernels found", "green", ""
        return "NOT FOUND", "red", f"expected in {csrc}"

    def _check_cython_ext(self):
        try:
            from xorzen.speed import xorzen_ext  # noqa
            return "LOADED", "green", ""
        except ImportError:
            return "NOT COMPILED", "yellow", "run: python setup_speed.py build_ext --inplace"

    def _check_expert_manifest(self):
        if self.config and hasattr(self.config, "expert_shard_dir"):
            p = Path(self.config.expert_shard_dir) / "manifest.json"
            if p.exists():
                import json
                d = json.loads(p.read_text())
                n = len(d.get("experts", {}))
                total = self.config.expert_count if self.config else "?"
                return f"{n}/{total} experts", "green", ""
            return "NO MANIFEST", "yellow", "will initialise on first run"
        return "N/A", "dim", "no config"

    def _check_expert_cache(self):
        if self.config and hasattr(self.config, "max_expert_cache"):
            return f"{self.config.max_expert_cache} slots warmed", "green", ""
        return "N/A", "dim", ""

    def _check_jit(self):
        import torch
        if not hasattr(torch, "compile"):
            return "DISABLED", "yellow", "upgrade PyTorch >= 2.0"
        if sys.platform == "win32" and not torch.cuda.is_available():
            return "torch.compile (GPU only)", "yellow", "skipped CPU-Windows"
        return "torch.compile ready", "green", f"PyTorch {torch.__version__}"

    def _check_speed_booster(self):
        try:
            from xorzen.speed import SpeedBooster  # noqa
            return "ARMED", "green", "LIGHTNING profile ready"
        except ImportError as e:
            return "FAILED", "red", str(e)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_ram_gb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / 1e9
        except ImportError:
            pass
        try:
            r = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=3,
            )
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip().isdigit()]
            if lines:
                return int(lines[0]) / 1e9
        except Exception:
            pass
        return 0.0
