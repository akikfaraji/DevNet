"""
dashboard.py -- xorzen Live Training Dashboard  (Tkinter GUI)
=============================================================
Real-time training monitor as a standalone desktop window.

THREADING MODEL (critical for Windows Tkinter stability):
  - Tkinter mainloop() MUST run on the main thread.
  - Training runs in a background worker thread.
  - Worker calls dashboard.update() which uses root.after() to
    schedule all widget updates back onto the main thread safely.
  - No Tk calls ever happen from the worker thread.

Usage
-----
    from xorzen.gui import XorzenDashboard

    dash = XorzenDashboard(config, total_steps=600)

    # Option A -- training already in a thread, call from main:
    dash.start_mainloop(training_fn)   # blocks until training done + window closed

    # Option B -- manual control (training on a thread you manage):
    dash.start()                       # opens window in background (non-blocking)
    dash.update(step, metrics)         # call from any thread
    dash.stop()                        # close window
"""

from __future__ import annotations
import os, sys, time, math, threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG         = "#0a0a0f"
BG_PANEL   = "#0e0e18"
BG_CANVAS  = "#080812"
FG         = "#c8d8f0"
FG_DIM     = "#3a4a5a"
FG_GREEN   = "#00ff88"
FG_YELLOW  = "#ffd700"
FG_RED     = "#ff4455"
FG_CYAN    = "#00d4ff"
FG_MAGENTA = "#cc88ff"
FG_WHITE   = "#ffffff"
ACCENT     = "#00d4ff"

FONT_MONO  = ("Consolas", 9)
FONT_S     = ("Consolas", 8)
FONT_B     = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 11, "bold")

WIN_W, WIN_H = 900, 680

_RICH = False   # kept for compatibility


# ---------------------------------------------------------------------------
# Pure helpers (no Tk dependency)
# ---------------------------------------------------------------------------
_SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

def _sparkline(values: List[float], width: int = 32) -> str:
    if not values:
        return "\u2500" * width
    vals = [v for v in list(values)[-width:] if math.isfinite(v)]
    if not vals:
        return "?" * width
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in vals:
        idx = int((v - lo) / rng * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[min(idx, len(_SPARK_CHARS) - 1)])
    return "".join(chars).ljust(width)


def _bar(fraction: float, width: int = 20,
         filled: str = "\u2588", empty: str = "\u2591") -> str:
    n = int(max(0.0, min(1.0, fraction)) * width)
    return filled * n + empty * (width - n)


_HEAT_CHARS = " \u2591\u2592\u2593\u2588"

def _expert_heatmap(usage: Dict[int, float], n_experts: int, width: int = 64) -> str:
    if not usage:
        return "\u2591" * width
    vals = [usage.get(i, 0.0) for i in range(n_experts)]
    chunk = max(1, n_experts // width)
    blocks = []
    for i in range(0, n_experts, chunk):
        avg = sum(vals[i:i+chunk]) / max(1, len(vals[i:i+chunk]))
        idx = int(avg * (len(_HEAT_CHARS) - 1))
        blocks.append(_HEAT_CHARS[min(idx, len(_HEAT_CHARS) - 1)])
    return "".join(blocks[:width]).ljust(width)


# ---------------------------------------------------------------------------

class XorzenDashboard:
    """
    Live Tkinter training dashboard.

    All Tk widget operations are marshalled onto the Tk thread via root.after().
    update() is safe to call from any thread.
    """

    def __init__(self, config=None, total_steps: int = 0,
                 refresh_rate: float = 4.0):
        self.config       = config
        self.total_steps  = total_steps
        self._refresh_ms  = max(150, int(1000 / refresh_rate))

        # State (protected by _lock)
        self._lock         = threading.Lock()
        self._step         = 0
        self._loss_hist    = deque(maxlen=256)
        self._tok_hist     = deque(maxlen=64)
        self._metrics: Dict[str, Any] = {}
        self._expert_usage: Dict[int, float] = {}
        self._path_probs   = {"local": 0.33, "low_rank": 0.33, "ssm": 0.34}
        self._cache_hit    = 0.0
        self._start_time   = time.time()
        self._training_done = False

        # Tk handles (set by _build_ui, used only on Tk thread)
        self._root        = None
        self._widgets     = {}        # name -> widget ref

        self._tk_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()   # set when root.mainloop starts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Open the dashboard window in a background thread.
        Non-blocking — returns immediately once the window is visible.
        Training should run on the main thread or another worker.
        """
        self._start_time = time.time()
        self._ready_event.clear()
        self._tk_thread = threading.Thread(target=self._tk_main, daemon=True)
        self._tk_thread.start()
        # Wait until the window is actually open before returning
        self._ready_event.wait(timeout=5.0)

    def stop(self):
        """Close the dashboard window."""
        self._training_done = True
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def update(self, step: int, metrics: Dict[str, Any]):
        """Thread-safe metric push. Safe to call from any thread."""
        with self._lock:
            self._step = step
            self._metrics.update(metrics)
            if "train_loss" in metrics and math.isfinite(metrics["train_loss"]):
                self._loss_hist.append(metrics["train_loss"])
            if "tokens_per_sec" in metrics:
                self._tok_hist.append(metrics["tokens_per_sec"])
            if "expert_usage" in metrics:
                self._expert_usage.update(metrics["expert_usage"])
            if "path_probs" in metrics:
                self._path_probs.update(metrics["path_probs"])
            if "cache_hit_rate" in metrics:
                self._cache_hit = metrics["cache_hit_rate"]

    def start_mainloop(self, training_fn: Callable):
        """
        Run training_fn in a worker thread while running the Tkinter
        mainloop on the CURRENT (main) thread.  Blocks until both finish.

        Use this when you want the simplest integration:
            dash.start_mainloop(lambda: train_model(...))
        """
        self._start_time = time.time()
        import tkinter as tk

        # Build window on current thread
        root = tk.Tk()
        self._root = root
        self._build_ui(root)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_refresh()
        self._ready_event.set()

        # Launch training in background
        def _worker():
            try:
                training_fn()
            finally:
                self._training_done = True
                # Give user a moment to see final state, then close
                time.sleep(1.5)
                try:
                    root.after(0, root.destroy)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        root.mainloop()
        t.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal Tk thread
    # ------------------------------------------------------------------

    def _tk_main(self):
        """Runs on the background Tk thread (Option B / start())."""
        import tkinter as tk
        root = tk.Tk()
        self._root = root
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui(root)
        self._schedule_refresh()
        self._ready_event.set()
        root.mainloop()

    def _on_close(self):
        self._training_done = True
        if self._root:
            self._root.destroy()

    # ------------------------------------------------------------------
    # UI construction (must be called from Tk thread)
    # ------------------------------------------------------------------

    def _build_ui(self, root):
        import tkinter as tk

        cfg  = self.config
        name = getattr(cfg, "model_name", "xorzen").upper() if cfg else "XORZEN"

        root.title(f"XORZEN  --  {name}")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{WIN_W}x{WIN_H}+{(sw-WIN_W)//2}+{(sh-WIN_H)//2}")

        # ── Title bar ────────────────────────────────────────────────
        self._widgets["title"] = tk.Label(
            root, text=f"  {name}  --  TRAINING DASHBOARD",
            font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL, anchor="w")
        self._widgets["title"].pack(fill="x", ipady=5)
        tk.Frame(root, height=1, bg=ACCENT).pack(fill="x")

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        # ── Progress ─────────────────────────────────────────────────
        pf = self._panel(body, "PROGRESS")
        pf.pack(fill="x", pady=(0, 3))
        self._widgets["prog_lbl"] = tk.Label(
            pf, text="  Waiting...", font=FONT_MONO,
            fg=FG_CYAN, bg=BG_PANEL, anchor="w")
        self._widgets["prog_lbl"].pack(fill="x", padx=6, pady=2)
        self._widgets["prog_canvas"] = tk.Canvas(
            pf, height=14, bg=BG_CANVAS, highlightthickness=0)
        self._widgets["prog_canvas"].pack(fill="x", padx=6, pady=(0, 4))

        # ── Loss curve ───────────────────────────────────────────────
        lf = self._panel(body, "LOSS CURVE")
        lf.pack(fill="x", pady=(0, 3))
        self._widgets["loss_lbl"] = tk.Label(
            lf, text="  collecting data...", font=FONT_MONO,
            fg=FG, bg=BG_PANEL, anchor="w")
        self._widgets["loss_lbl"].pack(fill="x", padx=6, pady=2)
        self._widgets["loss_canvas"] = tk.Canvas(
            lf, height=90, bg=BG_CANVAS, highlightthickness=0)
        self._widgets["loss_canvas"].pack(fill="x", padx=6, pady=(0, 4))

        # ── Two-column: System | Routing ─────────────────────────────
        cols = tk.Frame(body, bg=BG)
        cols.pack(fill="x", pady=(0, 3))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        sf = self._panel(cols, "SYSTEM")
        sf.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self._widgets["sys_lbl"] = tk.Label(
            sf, text="", font=FONT_S, fg=FG,
            bg=BG_PANEL, anchor="nw", justify="left")
        self._widgets["sys_lbl"].pack(fill="both", padx=6, pady=4, expand=True)

        rf = self._panel(cols, "PATHWAY ROUTING")
        rf.grid(row=0, column=1, sticky="nsew")
        self._widgets["route_canvas"] = tk.Canvas(
            rf, height=74, bg=BG_CANVAS, highlightthickness=0)
        self._widgets["route_canvas"].pack(fill="x", padx=6, pady=4)

        # ── Expert heatmap ───────────────────────────────────────────
        ef = self._panel(body, "EXPERT HEATMAP")
        ef.pack(fill="x", pady=(0, 3))
        self._widgets["expert_canvas"] = tk.Canvas(
            ef, height=28, bg=BG_CANVAS, highlightthickness=0)
        self._widgets["expert_canvas"].pack(fill="x", padx=6, pady=(2, 4))

        # ── Metrics log ──────────────────────────────────────────────
        mf = self._panel(body, "METRICS")
        mf.pack(fill="x")
        self._widgets["metrics_lbl"] = tk.Label(
            mf, text="  waiting for metrics...",
            font=FONT_S, fg=FG_DIM, bg=BG_PANEL, anchor="w")
        self._widgets["metrics_lbl"].pack(fill="x", padx=6, pady=3)

        # Force geometry to be computed so canvas widths are non-zero
        root.update_idletasks()

    def _panel(self, parent, title: str):
        import tkinter as tk
        outer = tk.Frame(parent, bg=ACCENT, bd=1)
        inner = tk.Frame(outer, bg=BG_PANEL)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(inner, text=f"  {title}",
                 font=("Consolas", 8, "bold"),
                 fg=ACCENT, bg=BG_PANEL, anchor="w").pack(fill="x")
        tk.Frame(inner, height=1, bg=FG_DIM).pack(fill="x")
        return inner

    # ------------------------------------------------------------------
    # Refresh loop (always on Tk thread via after())
    # ------------------------------------------------------------------

    def _schedule_refresh(self):
        if self._root:
            self._redraw()
            self._root.after(self._refresh_ms, self._schedule_refresh)

    def _redraw(self):
        """Called on Tk thread every refresh_ms ms."""
        with self._lock:
            step    = self._step
            total   = self.total_steps
            loss    = list(self._loss_hist)
            metrics = dict(self._metrics)
            expert  = dict(self._expert_usage)
            paths   = dict(self._path_probs)
            tok_s   = list(self._tok_hist)
            cache   = self._cache_hit

        elapsed = time.time() - self._start_time
        frac    = min(1.0, step / max(total, 1))

        # ── Progress ─────────────────────────────────────────────────
        eta = ""
        if step > 0 and total > 0:
            sps = step / max(elapsed, 1)
            rem = (total - step) / max(sps, 1e-6)
            h, m, s = int(rem//3600), int((rem%3600)//60), int(rem%60)
            eta = f"   ETA {h:02d}:{m:02d}:{s:02d}"
        eh = int(elapsed // 3600); em = int((elapsed % 3600) // 60)
        self._widgets["prog_lbl"].config(
            text=(f"  Step {step:,} / {total:,}   {frac*100:.1f}%{eta}"
                  f"    elapsed {eh:02d}:{em:02d}"))
        self._draw_bar(self._widgets["prog_canvas"], frac, FG_CYAN)

        # ── Loss curve ───────────────────────────────────────────────
        cur_l   = loss[-1] if loss else float("nan")
        finite  = [v for v in loss if math.isfinite(v)]
        best_l  = min(finite) if finite else float("nan")
        lr      = metrics.get("learning_rate", 0.0)
        cur_str = f"{cur_l:.4f}" if math.isfinite(cur_l) else "nan"
        best_str= f"{best_l:.4f}" if math.isfinite(best_l) else "nan"
        going_down = len(loss) > 1 and loss[-1] < loss[-2]
        trend   = " v" if going_down else (" ^" if len(loss) > 1 else "")
        self._widgets["loss_lbl"].config(
            text=f"  loss {cur_str}{trend}    best {best_str}    lr {lr:.2e}",
            fg=FG_GREEN if going_down else FG_YELLOW)
        self._draw_loss_curve(self._widgets["loss_canvas"], loss)

        # ── System ───────────────────────────────────────────────────
        avg_tok = (sum(tok_s[-8:]) / max(len(tok_s[-8:]), 1)) if tok_s else 0
        ram_u   = self._get_ram_used()
        ram_t   = self._get_ram_total()
        cpu_p   = self._get_cpu_pct()
        grad_n  = metrics.get("grad_norm", 0.0)
        self._widgets["sys_lbl"].config(text=(
            f"  CPU       {cpu_p:.0f}%\n"
            f"  RAM       {ram_u:.1f} / {ram_t:.0f} GB\n"
            f"  tok/s     {avg_tok:,.0f}\n"
            f"  cache hit {cache*100:.1f}%\n"
            f"  grad norm {grad_n:.4f}"))

        # ── Routing ──────────────────────────────────────────────────
        self._draw_routing(self._widgets["route_canvas"], paths)

        # ── Expert heatmap ───────────────────────────────────────────
        n_exp = (self.config.expert_count
                 if self.config and hasattr(self.config, "expert_count") else 64)
        self._draw_heatmap(self._widgets["expert_canvas"], expert, n_exp)

        # ── Metrics ──────────────────────────────────────────────────
        parts = []
        for k in ("eval_loss", "eval_perplexity", "val_loss"):
            if k in metrics:
                parts.append(f"{k.replace('_',' ')}: {metrics[k]:.4f}")
        self._widgets["metrics_lbl"].config(
            text=("  " + "    ".join(parts)) if parts else "  waiting for metrics...",
            fg=FG if parts else FG_DIM)

    # ------------------------------------------------------------------
    # Canvas drawing helpers
    # ------------------------------------------------------------------

    def _draw_bar(self, canvas, frac: float, color: str):
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 10:
            return
        filled = int(frac * w)
        canvas.create_rectangle(0, 0, w, 14, fill=BG_CANVAS, outline="")
        if filled > 0:
            canvas.create_rectangle(0, 0, filled, 14, fill=color, outline="")
        canvas.create_text(w // 2, 7, text=f"{frac*100:.1f}%",
                           fill=FG_WHITE, font=FONT_S)

    def _draw_loss_curve(self, canvas, loss: list):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = 90
        if w < 10:
            return
        finite = [v for v in loss if math.isfinite(v)]
        if len(finite) < 2:
            canvas.create_text(w // 2, h // 2,
                                text="collecting data...",
                                fill=FG_DIM, font=FONT_S)
            return
        lo, hi = min(finite), max(finite)
        rng = hi - lo if hi != lo else 1.0
        pts = finite[-min(len(finite), max(w // 3, 2)):]
        n   = len(pts)
        xs  = [int(i / max(n - 1, 1) * (w - 4)) + 2 for i in range(n)]
        ys  = [int((1 - (v - lo) / rng) * (h - 16)) + 8 for v in pts]
        # Gradient fill under curve
        for i in range(n - 1):
            y_avg = (ys[i] + ys[i + 1]) // 2
            canvas.create_rectangle(xs[i], y_avg, xs[i + 1], h - 2,
                                    fill="#002218", outline="")
        # Curve line
        for i in range(n - 1):
            canvas.create_line(xs[i], ys[i], xs[i + 1], ys[i + 1],
                                fill=FG_CYAN, width=1)
        # Current value dot + label
        canvas.create_oval(xs[-1]-3, ys[-1]-3, xs[-1]+3, ys[-1]+3,
                           fill=FG_GREEN, outline="")
        canvas.create_text(min(xs[-1] + 4, w - 32), ys[-1] - 10,
                           text=f"{pts[-1]:.3f}",
                           fill=FG_GREEN, font=FONT_S, anchor="w")
        # Y-axis range labels
        canvas.create_text(3, 6,     text=f"{hi:.2f}", fill=FG_DIM,
                           font=FONT_S, anchor="w")
        canvas.create_text(3, h - 8, text=f"{lo:.2f}", fill=FG_DIM,
                           font=FONT_S, anchor="w")

    def _draw_routing(self, canvas, paths: dict):
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 20:
            return
        items = [
            ("LocalAttn", paths.get("local", 0.33),    FG_CYAN),
            ("LowRank  ", paths.get("low_rank", 0.33), FG_YELLOW),
            ("SSM      ", paths.get("ssm", 0.34),      FG_MAGENTA),
        ]
        bar_w = w - 100
        row_h = 20
        pad_y = 8
        for i, (label, frac, color) in enumerate(items):
            y = pad_y + i * row_h
            canvas.create_text(6, y + 6, text=label,
                                fill=FG_DIM, font=FONT_S, anchor="w")
            bx     = 76
            filled = int(frac * bar_w)
            canvas.create_rectangle(bx, y + 2, bx + bar_w, y + 14,
                                    fill=BG, outline=FG_DIM)
            if filled > 0:
                canvas.create_rectangle(bx, y + 2, bx + filled, y + 14,
                                        fill=color, outline="")
            canvas.create_text(bx + bar_w + 6, y + 6,
                                text=f"{frac*100:.0f}%",
                                fill=color, font=FONT_S, anchor="w")

    def _draw_heatmap(self, canvas, usage: dict, n_experts: int):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = 28
        if w < 10:
            return
        if not usage:
            canvas.create_text(w // 2, h // 2,
                                text="no expert data",
                                fill=FG_DIM, font=FONT_S)
            return
        vals   = [usage.get(i, 0.0) for i in range(n_experts)]
        cell_w = w / n_experts
        max_v  = max(vals) if max(vals) > 0 else 1.0
        for i, v in enumerate(vals):
            inten = v / max_v
            g = int(200 * inten)
            b = int(255 * inten)
            color = f"#00{g:02x}{b:02x}"
            x0 = int(i * cell_w)
            x1 = int((i + 1) * cell_w)
            canvas.create_rectangle(x0, 4, x1, h - 4, fill=color, outline="")
        active = sum(1 for v in vals if v > 0.01)
        canvas.create_text(w - 4, h // 2,
                           text=f"active {active}/{n_experts}",
                           fill=FG_WHITE, font=FONT_S, anchor="e")

    # ------------------------------------------------------------------
    # System stats
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ram_used() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except ImportError:
            return 0.0

    @staticmethod
    def _get_ram_total() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / 1e9
        except ImportError:
            return 16.0

    @staticmethod
    def _get_cpu_pct() -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
