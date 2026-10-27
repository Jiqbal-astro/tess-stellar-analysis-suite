"""Tkinter GUI for the TESS Stellar Analysis Suite.

Heavy work (TESS retrieval and every analysis) runs in worker threads; results
come back to the Tk main thread through a queue. Each analysis owns its
parameters, job token and plot, so a stale result can never overwrite a newer
one, and processing changes re-use cached raw data instead of downloading
TESS data again.
"""
from __future__ import annotations

import json
import platform
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__
from .core import (
    DETREND_METHODS, FLUX_TYPES, NORMALIZATIONS, QUALITY_MODES,
    LCConfig, LCError, LightCurveData, RawSectorData,
    bls_search, concatenate, detect_flares, discover_sectors, export_csv, export_fits,
    export_metadata, load_light_curve_file, load_raw_sector, periodogram, phase_fold,
    process_raw, summary_stats, validate_tic,
)

APP_NAME = "TESS Stellar Analysis Suite"
TABS = ("Light Curve", "Periodogram", "BLS Transit", "Phase Fold", "Flares", "Rotation / Activity", "Statistics")
ANALYSIS_TABS = {
    "periodogram": "Periodogram",
    "bls": "BLS Transit",
    "phase": "Phase Fold",
    "flares": "Flares",
    "rotation": "Rotation / Activity",
}


class PlotPanel:
    def __init__(self, parent, title=""):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        # A bare Figure (not pyplot) avoids clashes with the platform's
        # default GUI backend, e.g. the native macOS backend.
        self.fig = Figure(figsize=(9, 5.4), dpi=95)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.125, right=0.98, bottom=0.12, top=0.92)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.toolbar.update()
        self.title = title

    def draw(self):
        self.canvas.draw_idle()

    def clear(self):
        self.ax.clear()
        self.ax.set_xscale("linear")
        self.ax.axis("on")

    def message(self, text, fontsize=11):
        self.clear()
        self.ax.axis("off")
        self.ax.text(.05, .92, text, va="top", transform=self.ax.transAxes, fontsize=fontsize, wrap=True)
        self.draw()


class App(tk.Tk):
    ANALYSES = tuple(ANALYSIS_TABS)

    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{__version__}")
        self.geometry("1500x930")
        self.minsize(1180, 760)
        self.queue: queue.Queue = queue.Queue()

        # Data state. raw_sectors always belongs to `current_tic`.
        self.current_tic: int | None = None
        self.raw_sectors: dict[int, RawSectorData] = {}
        self.active_sectors: list[int] = []
        self.listed_tic: int | None = None
        self.data_source = "tess"
        self.current: LightCurveData | None = None
        self.busy_data = False
        self._last_config: dict | None = None
        self._last_range: tuple | None = None

        # Analysis state
        self._auto_job = None
        self._processing_generation = 0
        self._job_generation = {name: 0 for name in self.ANALYSES}
        self._last_params: dict = {name: None for name in self.ANALYSES}
        self.analysis_armed = {name: False for name in self.ANALYSES}
        self.analysis_results: dict = {}
        self.analysis_params: dict = {}
        self.analysis_count = {name: 0 for name in self.ANALYSES}  # completed runs

        self._build_ui()
        self.after(60, self._poll_queue)

    # ================================================================== UI
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(self)
        left_outer.grid(row=0, column=0, sticky="nsew")
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)
        self.left_canvas = tk.Canvas(left_outer, width=390, highlightthickness=0, borderwidth=0)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=self.left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky="ns")
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left = ttk.Frame(self.left_canvas, padding=10)
        left_window = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _resize_left(_event=None):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            self.left_canvas.itemconfigure(left_window, width=self.left_canvas.winfo_width())

        left.bind("<Configure>", _resize_left)
        self.left_canvas.bind("<Configure>", _resize_left)
        # Scroll the control column whenever the pointer is over it, including
        # over its child widgets (the original only scrolled over empty space).
        left_outer.bind("<Enter>", lambda _e: self._bind_wheel(True))
        left_outer.bind("<Leave>", lambda _e: self._bind_wheel(False))

        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_target(left)
        self._build_sector(left)
        self._build_processing(left)
        self._build_range(left)
        self._build_analysis_controls(left)
        self._build_export(left)

        self.status_var = tk.StringVar(value="Ready — enter a TIC ID and click Generate Light Curve.")
        ttk.Label(left, textvariable=self.status_var, wraplength=330, foreground="#1f4e79").pack(fill="x", pady=8)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")
        self.panels: dict[str, PlotPanel] = {}
        for name in TABS:
            p = PlotPanel(self.nb, name)
            self.nb.add(p.frame, text=name)
            self.panels[name] = p
        self._reset_panels()
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        results = ttk.LabelFrame(right, text="Current result", padding=6)
        results.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.result_var = tk.StringVar(value="No light curve loaded.")
        ttk.Label(results, textvariable=self.result_var, justify="left").pack(anchor="w")

    def _bind_wheel(self, on):
        if on:
            self.bind_all("<MouseWheel>", self._on_wheel)
            self.bind_all("<Button-4>", self._on_wheel)
            self.bind_all("<Button-5>", self._on_wheel)
        else:
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_all(seq)

    def _on_wheel(self, event):
        if isinstance(event.widget, tk.Listbox):
            return  # let the sector list scroll itself
        if getattr(event, "num", None) in (4, 5):
            step = -1 if event.num == 4 else 1
        elif platform.system() == "Darwin":
            step = -event.delta          # macOS reports small deltas (±1, ±2 …)
        else:
            step = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        self.left_canvas.yview_scroll(step, "units")

    def _build_target(self, left):
        target = ttk.LabelFrame(left, text="Target", padding=8)
        target.pack(fill="x", pady=(0, 8))
        target.columnconfigure(0, weight=1)
        ttk.Label(target, text="TIC ID").grid(row=0, column=0, sticky="w")
        self.tic_var = tk.StringVar()
        tic_entry = ttk.Entry(target, textvariable=self.tic_var, width=25)
        tic_entry.grid(row=1, column=0, sticky="ew", pady=3)
        tic_entry.bind("<Return>", lambda _e: self.generate_quick())
        self.generate_btn = ttk.Button(target, text="Generate Light Curve", command=self.generate_quick)
        self.generate_btn.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        self.all_sectors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(target, text="Retrieve all available TESS sectors",
                        variable=self.all_sectors_var).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(target, text="Off = fastest recent sector. On = all sectors found for this TIC.",
                  wraplength=300).grid(row=4, column=0, sticky="w")
        self.discover_btn = ttk.Button(target, text="Find Other TESS Sectors", command=self.discover)
        self.discover_btn.grid(row=5, column=0, sticky="ew", pady=2)
        self.open_btn = ttk.Button(target, text="Open Local Light Curve…", command=self.open_file)
        self.open_btn.grid(row=6, column=0, sticky="ew", pady=2)

    def _build_sector(self, left):
        sec = ttk.LabelFrame(left, text="Sectors", padding=8)
        sec.pack(fill="x", pady=8)
        self.sector_list = tk.Listbox(sec, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        self.sector_list.pack(fill="x")
        b = ttk.Frame(sec)
        b.pack(fill="x", pady=(5, 0))
        ttk.Button(b, text="Select all", command=self.select_all).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.load_btn = ttk.Button(b, text="Load selected", command=self.load_selected)
        self.load_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

    def _build_processing(self, left):
        proc = ttk.LabelFrame(left, text="Light-curve processing (automatic)", padding=8)
        proc.pack(fill="x", pady=8)
        self.flux_var = tk.StringVar(value="Corrected Flux")
        self.quality_var = tk.StringVar(value="Strict (quality == 0)")
        self.norm_var = tk.StringVar(value="Median")
        self.detrend_var = tk.StringVar(value="None")
        self.window_var = tk.StringVar(value="101")
        self.clip_var = tk.StringVar(value="0.0")
        combos = [("Flux", self.flux_var, FLUX_TYPES), ("Quality", self.quality_var, QUALITY_MODES),
                  ("Normalize", self.norm_var, NORMALIZATIONS), ("Detrend", self.detrend_var, DETREND_METHODS)]
        for i, (label, var, vals) in enumerate(combos):
            self._combo(proc, label, var, vals, i).bind("<<ComboboxSelected>>", self._auto_from_event)
        for row, (label, var) in enumerate((("Window [cadences]", self.window_var),
                                            ("Sigma clip (0 = off)", self.clip_var)), start=4):
            ttk.Label(proc, text=label).grid(row=row, column=0, sticky="w", pady=2)
            e = ttk.Entry(proc, textvariable=var, width=12)
            e.grid(row=row, column=1, sticky="ew", pady=2)
            e.bind("<Return>", self._auto_from_event)
            e.bind("<FocusOut>", self._auto_from_event)
        proc.columnconfigure(1, weight=1)
        ttk.Label(proc, text="Settings reprocess cached data. TESS is not downloaded again.",
                  wraplength=320).grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def _build_range(self, left):
        rng = ttk.LabelFrame(left, text="Light-curve range (BJD − 2457000)", padding=8)
        rng.pack(fill="x", pady=8)
        self.bjd_min, self.bjd_max = tk.StringVar(), tk.StringVar()
        for row, (label, var) in enumerate((("BJD min", self.bjd_min), ("BJD max", self.bjd_max))):
            ttk.Label(rng, text=label).grid(row=row, column=0, sticky="w")
            e = ttk.Entry(rng, textvariable=var, width=13)
            e.grid(row=row, column=1, sticky="ew")
            e.bind("<Return>", self._refresh_range)
            e.bind("<FocusOut>", self._refresh_range)
        ttk.Button(rng, text="Full range", command=self.reset_range).grid(row=2, column=0, columnspan=2,
                                                                          sticky="ew", pady=(5, 0))
        rng.columnconfigure(1, weight=1)

    def _build_analysis_controls(self, left):
        an = ttk.LabelFrame(left, text="Independent analysis", padding=8)
        an.pack(fill="x", pady=8)
        self.pmin_var, self.pmax_var = tk.StringVar(value="0.05"), tk.StringVar(value="20")
        self.blsmin_var, self.blsmax_var = tk.StringVar(value="0.5"), tk.StringVar(value="20")
        self.phase_period_var, self.phase_epoch_var = tk.StringVar(), tk.StringVar()
        self.phase_bin_var = tk.StringVar(value="0.01")
        self.flare_thr_var, self.flare_min_var = tk.StringVar(value="5.0"), tk.StringVar(value="2")
        self.rotmin_var, self.rotmax_var = tk.StringVar(value="0.2"), tk.StringVar(value="30")

        rows = [
            ("LS period [d]", (self.pmin_var, self.pmax_var), "periodogram"),
            ("BLS period [d]", (self.blsmin_var, self.blsmax_var), "bls"),
            ("Phase period [d]", (self.phase_period_var,), "phase"),
            ("Phase epoch [BTJD]", (self.phase_epoch_var,), "phase"),
            ("Phase bin [cycles]", (self.phase_bin_var,), "phase"),
            ("Flare threshold [σ]", (self.flare_thr_var,), "flares"),
            ("Flare min points", (self.flare_min_var,), "flares"),
            ("Rotation period [d]", (self.rotmin_var, self.rotmax_var), "rotation"),
        ]
        for r, (label, variables, name) in enumerate(rows):
            ttk.Label(an, text=label).grid(row=r, column=0, sticky="w", pady=1)
            for c, var in enumerate(variables, start=1):
                width = 7 if len(variables) == 2 else 16
                e = ttk.Entry(an, textvariable=var, width=width)
                e.grid(row=r, column=c, columnspan=1 if len(variables) == 2 else 2, sticky="w", padx=1, pady=1)
                e.bind("<Return>", lambda _e, n=name: self._param_event(n))
                e.bind("<FocusOut>", lambda _e, n=name: self._param_event(n))
        buttons = ttk.Frame(an)
        buttons.grid(row=len(rows), column=0, columnspan=3, sticky="ew", pady=(7, 2))
        for text, cmd in (("Run Periodogram", self.run_periodogram), ("Run BLS", self.run_bls),
                          ("Phase Fold", self.run_phase), ("Screen Flares", self.run_flares),
                          ("Rotation / Activity", self.run_rotation)):
            ttk.Button(buttons, text=text, command=cmd).pack(fill="x", pady=1)
        ttk.Label(an, text="Each analysis runs when its tab is opened. After the first run, changing its "
                           "parameters, the range or the processing re-runs it automatically.",
                  wraplength=320).grid(row=len(rows) + 1, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def _build_export(self, left):
        exp = ttk.LabelFrame(left, text="Export", padding=8)
        exp.pack(fill="x", pady=8)
        for txt, cmd in (("Export CSV", self.save_csv), ("Export FITS", self.save_fits),
                         ("Export metadata JSON", self.save_metadata), ("Export analysis JSON", self.save_analysis),
                         ("Save current plot", self.save_current_plot)):
            ttk.Button(exp, text=txt, command=cmd).pack(fill="x", pady=2)

    @staticmethod
    def _combo(parent, label, var, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        c = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=22)
        c.grid(row=row, column=1, sticky="ew", pady=2)
        return c

    def _reset_panels(self):
        self.panels["Light Curve"].message("No light curve loaded.\n\nEnter a TIC ID and click Generate Light Curve,\n"
                                           "or open a local light-curve file.")
        for name, tab in ANALYSIS_TABS.items():
            self.panels[tab].message("Open this tab after loading a light curve; it calculates automatically.")
        self.panels["Statistics"].message("No light curve loaded.")

    # ============================================================ threading
    @staticmethod
    def _thread(fn):
        threading.Thread(target=fn, daemon=True).start()

    def _set_data_busy(self, busy):
        self.busy_data = busy
        state = "disabled" if busy else "normal"
        for btn in (self.generate_btn, self.discover_btn, self.load_btn, self.open_btn):
            btn.configure(state=state)

    def _poll_queue(self):
        # Every message is handled independently and the poll is always
        # rescheduled, so one failing handler can never freeze the GUI.
        try:
            while True:
                try:
                    kind, payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(kind, payload)
                except Exception as exc:  # pragma: no cover - defensive
                    self.status_var.set(f"Display error: {exc}")
        finally:
            self.after(60, self._poll_queue)

    def _handle(self, kind, payload):
        if kind == "status":
            self.status_var.set(payload)
        elif kind == "done":
            self._set_data_busy(False)
            self.status_var.set(payload)
        elif kind == "error":
            self._set_data_busy(False)
            self.status_var.set("Operation failed.")
            messagebox.showerror(APP_NAME, payload)
        elif kind == "warning":
            messagebox.showwarning(APP_NAME, payload)
        elif kind == "loaded":
            self._apply_loaded(payload, new_target=True)
        elif kind == "reprocessed":
            token, data = payload
            if token == self._processing_generation:
                self._apply_loaded(data, new_target=False)
        elif kind == "sectors":
            tic, sectors = payload
            self.listed_tic = tic
            self.sector_list.delete(0, tk.END)
            for s in sectors:
                self.sector_list.insert(tk.END, str(s))
            self._mark_active_sectors()
        elif kind == "analysis_status":
            name, text = payload
            self.status_var.set(f"{name}: {text}")
        elif kind == "analysis_error":
            name, token, select, text = payload
            if token == self._job_generation[name]:
                self._show_analysis_error(name, text, select)
        elif kind == "analysis":
            name, token, select, result = payload
            if token == self._job_generation[name]:
                getattr(self, f"_show_{name}")(result, select)
                self.analysis_count[name] += 1

    # ================================================================= data
    def _config(self):
        try:
            window, clip = int(self.window_var.get()), float(self.clip_var.get())
        except ValueError as exc:
            raise LCError("Detrend window must be a whole number and sigma clip must be numeric.") from exc
        if window < 5:
            raise LCError("Detrend window must be at least 5 cadences.")
        if clip < 0:
            raise LCError("Sigma clip must be 0 (off) or positive.")
        return LCConfig(flux_type=self.flux_var.get(), quality_mode=self.quality_var.get(),
                        normalize=self.norm_var.get(), detrend=self.detrend_var.get(),
                        detrend_window=window, sigma_clip=clip)

    def _input_error(self, exc):
        messagebox.showerror("Input Error", str(exc))

    @staticmethod
    def _load_many(tic, sectors, config, cached, report):
        """Load sectors (re-using cache), skipping sectors that fail."""
        raw_map, failed = {}, []
        for i, sec in enumerate(sectors, 1):
            have = cached.get(sec)
            if have is not None and not (config.flux_type == "PCA Flux" and have.pca_flux is None):
                raw_map[sec] = have
                continue
            report(f"Retrieving TIC {tic}: Sector {sec} ({i}/{len(sectors)})…")
            try:
                raw_map[sec] = load_raw_sector(tic, sec, config)
            except LCError as exc:
                failed.append(f"Sector {sec}: {exc}")
        return raw_map, failed

    @staticmethod
    def _process_all(raw_map, sectors, config):
        processed, failed = [], []
        for s in sectors:
            try:
                processed.append(process_raw(raw_map[s], config))
            except LCError as exc:
                failed.append(f"Sector {s}: {exc}")
        if not processed:
            raise LCError("No sector could be processed:\n" + "\n".join(failed))
        return concatenate(processed), failed

    def generate_quick(self):
        if self.busy_data:
            return
        try:
            tic, config = validate_tic(self.tic_var.get()), self._config()
        except LCError as exc:
            self._input_error(exc)
            return
        self._set_data_busy(True)
        all_sectors = bool(self.all_sectors_var.get())
        cached = dict(self.raw_sectors) if tic == self.current_tic and self.data_source == "tess" else {}

        def work():
            report = lambda text: self.queue.put(("status", text))  # noqa: E731
            try:
                if all_sectors:
                    report(f"Finding all TESS sectors containing TIC {tic}…")
                    sectors = discover_sectors(tic)
                    self.queue.put(("sectors", (tic, sectors)))
                    raw_map, failed = self._load_many(tic, sectors, config, cached, report)
                    if not raw_map:
                        raise LCError("No sector could be retrieved:\n" + "\n".join(failed))
                    sectors = sorted(raw_map)
                else:
                    report(f"Retrieving TIC {tic} with Eleanor (most recent sector)…")
                    raw = load_raw_sector(tic, None, config)
                    raw_map, sectors, failed = {raw.sector: raw}, [raw.sector], []
                current, failed_proc = self._process_all(raw_map, sectors, config)
                self.queue.put(("loaded", dict(tic=tic, raw=raw_map, active=sectors, current=current,
                                               config=config, source="tess")))
                problems = failed + failed_proc
                if problems:
                    self.queue.put(("warning", "Some sectors were skipped:\n\n" + "\n".join(problems)))
                self.queue.put(("done", f"Loaded TIC {tic}: sector(s) {sectors}, "
                                        f"{len(current.time):,} usable cadences."))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        self._thread(work)

    def discover(self):
        if self.busy_data:
            return
        try:
            tic = validate_tic(self.tic_var.get())
        except LCError as exc:
            self._input_error(exc)
            return
        self._set_data_busy(True)

        def work():
            try:
                self.queue.put(("status", f"Finding available sectors for TIC {tic}…"))
                sectors = discover_sectors(tic)
                self.queue.put(("sectors", (tic, sectors)))
                self.queue.put(("done", f"Found {len(sectors)} sector(s) for TIC {tic}. Select and load what you need."))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        self._thread(work)

    def select_all(self):
        self.sector_list.selection_set(0, tk.END)

    def _mark_active_sectors(self):
        if self.listed_tic != self.current_tic:
            return
        self.sector_list.selection_clear(0, tk.END)
        for i in range(self.sector_list.size()):
            if int(self.sector_list.get(i)) in self.active_sectors:
                self.sector_list.selection_set(i)

    def load_selected(self):
        if self.busy_data:
            return
        try:
            tic, config = validate_tic(self.tic_var.get()), self._config()
            sectors = sorted(int(self.sector_list.get(i)) for i in self.sector_list.curselection())
            if not sectors:
                raise LCError("Select at least one sector (use Find Other TESS Sectors first).")
            if self.listed_tic is not None and self.listed_tic != tic:
                raise LCError(f"The sector list belongs to TIC {self.listed_tic}. "
                              f"Click Find Other TESS Sectors to list sectors for TIC {tic}.")
        except LCError as exc:
            self._input_error(exc)
            return
        self._set_data_busy(True)
        cached = dict(self.raw_sectors) if tic == self.current_tic and self.data_source == "tess" else {}

        def work():
            report = lambda text: self.queue.put(("status", text))  # noqa: E731
            try:
                raw_map, failed = self._load_many(tic, sectors, config, cached, report)
                if not raw_map:
                    raise LCError("No sector could be retrieved:\n" + "\n".join(failed))
                loaded = sorted(raw_map)
                current, failed_proc = self._process_all(raw_map, loaded, config)
                all_raw = {**cached, **raw_map}
                self.queue.put(("loaded", dict(tic=tic, raw=all_raw, active=loaded, current=current,
                                               config=config, source="tess")))
                if failed + failed_proc:
                    self.queue.put(("warning", "Some sectors were skipped:\n\n" + "\n".join(failed + failed_proc)))
                self.queue.put(("done", f"Loaded TIC {tic}: sector(s) {loaded}, {len(current.time):,} usable cadences."))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        self._thread(work)

    def open_file(self, path=None):
        if self.busy_data:
            return
        try:
            config = self._config()
        except LCError as exc:
            self._input_error(exc)
            return
        path = path or filedialog.askopenfilename(
            title="Open light curve",
            filetypes=[("Light curves", "*.fits *.fit *.csv *.txt *.dat"), ("All files", "*.*")])
        if not path:
            return
        if config.flux_type == "PCA Flux":
            self.flux_var.set("Corrected Flux")
            config.flux_type = "Corrected Flux"
        self._set_data_busy(True)

        def work():
            try:
                self.queue.put(("status", f"Reading {Path(path).name}…"))
                raw = load_light_curve_file(path)
                current = process_raw(raw, config)
                self.queue.put(("loaded", dict(tic=raw.tic, raw={raw.sector: raw}, active=[raw.sector],
                                               current=current, config=config, source="file")))
                self.queue.put(("done", f"Opened {Path(path).name}: {len(current.time):,} usable cadences."))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        self._thread(work)

    def _apply_loaded(self, data, new_target):
        self.raw_sectors = data["raw"]
        self.active_sectors = list(data["active"])
        self.current = data["current"]
        self.current_tic = data["tic"]
        self.data_source = data.get("source", self.data_source)
        self._last_config = asdict(data["config"])
        if new_target:
            if data["tic"]:
                self.tic_var.set(str(data["tic"]))
            self._after_new_target()
        else:
            self._after_reprocess()
        self._mark_active_sectors()

    # =================================================== automatic processing
    def _auto_from_event(self, _event=None):
        if self.current is None or self.busy_data:
            return
        if self._auto_job:
            self.after_cancel(self._auto_job)
        self._auto_job = self.after(250, self._apply_processing)

    def _apply_processing(self, force=False):
        self._auto_job = None
        try:
            config = self._config()
        except LCError as exc:
            self.status_var.set(str(exc))
            return
        if not self.raw_sectors or (not force and asdict(config) == self._last_config):
            return
        if config.flux_type == "PCA Flux" and self.data_source == "file" and \
                any(self.raw_sectors[s].pca_flux is None for s in self.active_sectors):
            self.flux_var.set(self._last_config["flux_type"])
            messagebox.showinfo(APP_NAME, "PCA Flux is only available for data retrieved from TESS with Eleanor.")
            return
        self._processing_generation += 1
        token = self._processing_generation
        tic, raw_map, active = self.current_tic, dict(self.raw_sectors), list(self.active_sectors)
        self.status_var.set("Reprocessing cached data…")
        need_pca = config.flux_type == "PCA Flux" and any(raw_map[s].pca_flux is None for s in active)
        if need_pca:
            self._set_data_busy(True)

        def work():
            try:
                rmap = raw_map
                if need_pca:
                    self.queue.put(("status", "PCA selected: retrieving PCA products once…"))
                    fresh, failed = self._load_many(tic, active, config, {}, lambda s: self.queue.put(("status", s)))
                    if failed:
                        raise LCError("\n".join(failed))
                    rmap = {**raw_map, **fresh}
                current, failed = self._process_all(rmap, active, config)
                self.queue.put(("reprocessed", (token, dict(tic=tic, raw=rmap, active=active, current=current,
                                                            config=config, source=self.data_source))))
                if failed:
                    self.queue.put(("warning", "Some sectors could not be processed:\n\n" + "\n".join(failed)))
                if need_pca:
                    self.queue.put(("done", "PCA flux loaded."))
                elif token == self._processing_generation:
                    self.queue.put(("status", "Processing updated automatically from cached data."))
            except Exception as exc:
                if token == self._processing_generation:
                    self.queue.put(("error", str(exc)))

        self._thread(work)

    # ============================================================== plotting
    def _range_values(self):
        if self.current is None:
            raise LCError("Generate or open a light curve first.")
        try:
            a = float(self.bjd_min.get()) if self.bjd_min.get().strip() else float(self.current.time.min())
            b = float(self.bjd_max.get()) if self.bjd_max.get().strip() else float(self.current.time.max())
        except ValueError as exc:
            raise LCError("BJD limits must be numeric.") from exc
        if a >= b:
            raise LCError("BJD minimum must be less than BJD maximum.")
        return a, b

    def _range_mask(self):
        a, b = self._range_values()
        m = (self.current.time >= a) & (self.current.time <= b)
        if not m.any():
            raise LCError("No data points fall inside the selected BJD range.")
        return m

    @staticmethod
    def _display_slice(x, y, max_points=45000):
        """Decimate for display and break the line at data gaps (> 0.5 d)."""
        if len(x) > max_points:
            step = int(np.ceil(len(x) / max_points))
            x, y = x[::step], y[::step]
        gaps = np.where(np.diff(x) > 0.5)[0] + 1
        if len(gaps):
            x = np.insert(np.asarray(x, dtype=float), gaps, np.nan)
            y = np.insert(np.asarray(y, dtype=float), gaps, np.nan)
        return x, y

    def _flux_label(self):
        if self.norm_var.get() == "Percent deviation":
            return "Flux deviation [%]"
        if self.norm_var.get() == "None" and self.detrend_var.get() == "None":
            return "Flux [e⁻/s]"
        return "Relative flux"

    def _sector_boundaries(self, ax, m):
        lo, hi = float(self.current.time[m].min()), float(self.current.time[m].max())
        for sec in self.active_sectors:
            raw = self.raw_sectors.get(int(sec))
            if raw is None:
                continue
            for value in (np.nanmin(raw.time), np.nanmax(raw.time)):
                if lo < value < hi:
                    ax.axvline(value, ls=":", alpha=.35, color="k")

    def plot_lightcurve(self, preserve_view=False):
        if self.current is None:
            return
        try:
            m = self._range_mask()
        except LCError as exc:
            self.status_var.set(str(exc))
            return
        p = self.panels["Light Curve"]
        had_data = preserve_view and p.ax.has_data()
        oldx, oldy = (p.ax.get_xlim(), p.ax.get_ylim()) if had_data else (None, None)
        p.clear()
        x, y = self._display_slice(self.current.time[m], self.current.flux[m])
        p.ax.plot(x, y, lw=.65, rasterized=True)
        self._sector_boundaries(p.ax, m)
        p.ax.set_xlabel("Time [BJD − 2457000]")
        p.ax.set_ylabel(self._flux_label())
        label = f"TIC {self.current.tic}" if self.current.tic else self.current.metadata.get("file", "Light curve")
        p.ax.set_title(f"{label} | Sector(s): {', '.join(map(str, self.active_sectors))}")
        p.ax.grid(alpha=.18)
        p.ax.ticklabel_format(useOffset=False, axis="x")
        if oldx is not None:
            p.ax.set_xlim(oldx)
            p.ax.set_ylim(oldy)
        p.draw()
        self.update_summary()

    def _set_full_range(self):
        self.bjd_min.set(f"{float(self.current.time.min()):.6f}")
        self.bjd_max.set(f"{float(self.current.time.max()):.6f}")
        self._last_range = self._range_values()

    def _after_new_target(self):
        self._set_full_range()
        for name in self.ANALYSES:
            self.analysis_armed[name] = False
            self._job_generation[name] += 1
            self._last_params[name] = None
        self.analysis_results.clear()
        self.analysis_params.clear()
        self.phase_period_var.set("")
        self.phase_epoch_var.set("")
        for tab in ANALYSIS_TABS.values():
            self.panels[tab].message("Open this tab to calculate this analysis for the new light curve.")
        self.panels["Light Curve"].clear()
        self.nb.select(self.panels["Light Curve"].frame)
        self.plot_lightcurve(False)
        self.plot_statistics()
        self.result_var.set("Light curve loaded. Open an analysis tab; it will calculate automatically.")

    def _after_reprocess(self):
        # Keep the user's range if it still overlaps the data; otherwise reset.
        try:
            self._range_mask()
        except LCError:
            self._set_full_range()
        self.plot_lightcurve(True)
        self.plot_statistics()
        if any(self.analysis_armed.values()):
            self.result_var.set("Processing updated. Analyses already used are recalculating.")
        self._rerun_armed_analyses(force=True)

    def _refresh_range(self, _event=None):
        if self.current is None:
            return
        try:
            rng = self._range_values()
        except LCError as exc:
            self.status_var.set(str(exc))
            return
        if rng == self._last_range:
            return
        self._last_range = rng
        self.plot_lightcurve(False)
        self.plot_statistics()
        self._rerun_armed_analyses(force=True)

    def reset_range(self):
        if self.current is None:
            return
        self._set_full_range()
        self.plot_lightcurve(False)
        self.plot_statistics()
        self._rerun_armed_analyses(force=True)

    def reset_zoom(self):
        self.plot_lightcurve(False)

    def _window_stats(self):
        m = self._range_mask()
        c = self.current
        lc = LightCurveData(c.tic, c.sector, c.time[m], c.flux[m], c.raw_flux[m], c.corr_flux[m],
                            None, c.quality[m], None, c.metadata)
        return summary_stats(lc)

    def update_summary(self):
        if self.current is None:
            return
        s = self._window_stats()
        self.result_var.set(f"TIC {self.current.tic} | Sectors: {self.active_sectors} | Cadences: {s.get('n', 0):,} "
                            f"| RMS: {s.get('rms_ppm', np.nan):,.0f} ppm | point-to-point: "
                            f"{s.get('point_to_point_ppm', np.nan):,.0f} ppm")

    def plot_statistics(self):
        if self.current is None:
            return
        try:
            s = self._window_stats()
        except LCError as exc:
            self.panels["Statistics"].message(str(exc))
            return
        cfg = self.current.metadata.get("config", {})
        lines = [
            ("Target", f"TIC {self.current.tic}"),
            ("Sectors", ", ".join(map(str, self.active_sectors))),
            ("Time range [BTJD]", f"{s['time_start']:.4f} – {s['time_end']:.4f}"),
            ("Time span [d]", f"{s['time_span_days']:.3f}"),
            ("Cadence [min]", f"{s['cadence_minutes']:.2f}"),
            ("Cadences", f"{s['n']:,}"),
            ("", ""),
            ("Median flux", f"{s['median']:.7g}"),
            ("Mean flux", f"{s['mean']:.7g}"),
            ("Std", f"{s['std']:.7g}"),
            ("MAD", f"{s['mad']:.7g}"),
            ("Min / Max", f"{s['min']:.7g} / {s['max']:.7g}"),
            ("", ""),
            ("RMS [ppm]", f"{s['rms_ppm']:,.1f}"),
            ("Robust RMS [ppm]", f"{s['robust_rms_ppm']:,.1f}"),
            ("Point-to-point [ppm]", f"{s['point_to_point_ppm']:,.1f}"),
            ("", ""),
            ("Processing", f"{cfg.get('flux_type')}, {cfg.get('quality_mode')}"),
            ("", f"normalize={cfg.get('normalize')}, detrend={cfg.get('detrend')} "
                 f"({cfg.get('detrend_window')}), clip={cfg.get('sigma_clip')}"),
        ]
        text = "\n".join(f"{k:<22}{v}" for k, v in lines)
        p = self.panels["Statistics"]
        p.clear()
        p.ax.axis("off")
        p.ax.text(.04, .95, text, va="top", family="monospace", fontsize=10.5, transform=p.ax.transAxes)
        p.ax.text(.04, .02, "Statistics refer to the selected BJD range. Point-to-point scatter is a robust "
                            "estimate of white noise per cadence.", fontsize=8.5, color="0.4", transform=p.ax.transAxes)
        p.ax.set_title("Light-curve statistics")
        p.draw()

    # ======================================================== analysis helpers
    def _analysis_window(self):
        if self.current is None:
            raise LCError("Generate or open a light curve first.")
        m = self._range_mask()
        t = np.asarray(self.current.time[m], dtype=float)
        f = np.asarray(self.current.flux[m], dtype=float)
        good = np.isfinite(t) & np.isfinite(f)
        t, f = t[good], f[good]
        if len(t) < 20:
            raise LCError("Not enough finite cadences in the selected range.")
        order = np.argsort(t)
        t, f = t[order], f[order]
        keep = np.r_[True, np.diff(t) > 0]
        return t[keep], f[keep]

    def _params(self, name):
        """Read and validate the parameters of one analysis."""
        try:
            if name == "periodogram":
                a, b = float(self.pmin_var.get()), float(self.pmax_var.get())
                if a <= 0 or b <= a:
                    raise LCError("LS period limits must satisfy 0 < minimum < maximum.")
                return {"min_period": a, "max_period": b}
            if name == "bls":
                a, b = float(self.blsmin_var.get()), float(self.blsmax_var.get())
                if a <= 0 or b <= a:
                    raise LCError("BLS period limits must satisfy 0 < minimum < maximum.")
                return {"min_period": a, "max_period": b}
            if name == "rotation":
                a, b = float(self.rotmin_var.get()), float(self.rotmax_var.get())
                if a <= 0 or b <= a:
                    raise LCError("Rotation period limits must satisfy 0 < minimum < maximum.")
                return {"min_period": a, "max_period": b}
            if name == "phase":
                text = self.phase_period_var.get().strip()
                if not text:
                    raise LCError("Enter a phase period, or run BLS first to transfer its candidate.")
                period = float(text)
                epoch_text, bin_text = self.phase_epoch_var.get().strip(), self.phase_bin_var.get().strip()
                epoch = float(epoch_text) if epoch_text else None
                bin_size = float(bin_text) if bin_text else None
                if period <= 0:
                    raise LCError("Phase period must be positive.")
                if bin_size is not None and not 0 < bin_size < 0.5:
                    raise LCError("Phase bin must be between 0 and 0.5 cycles (leave empty for no binning).")
                return {"period": period, "epoch": epoch, "bin_size": bin_size}
            if name == "flares":
                threshold, min_points = float(self.flare_thr_var.get()), int(self.flare_min_var.get())
                if threshold <= 0 or min_points < 1:
                    raise LCError("Flare threshold must be > 0 and minimum points must be ≥ 1.")
                return {"threshold": threshold, "min_points": min_points}
        except ValueError as exc:
            raise LCError(f"{ANALYSIS_TABS[name]}: parameters must be numeric.") from exc
        raise KeyError(name)

    def _run(self, name, compute, auto=False, select=None, force=False):
        """Validate, then compute `name` in a worker thread."""
        if select is None:
            select = not auto
        try:
            params = self._params(name)
            t, f = self._analysis_window()
        except LCError as exc:
            if not auto:
                self._show_analysis_error(name, str(exc), select=True)
            return
        key = (params, self._range_values(), self._processing_generation)
        if auto and not force and self.analysis_armed[name] and key == self._last_params[name]:
            return
        self._last_params[name] = key
        self.analysis_armed[name] = True
        self._job_generation[name] += 1
        token = self._job_generation[name]
        label = ANALYSIS_TABS[name]
        if select:
            self.nb.select(self.panels[label].frame)
        self.panels[label].message(f"Calculating {label}…")
        self.status_var.set(f"{label}: calculating…")

        def work():
            try:
                result = compute(t, f, params)
                self.queue.put(("analysis", (name, token, select, (params, result))))
                self.queue.put(("analysis_status", (label, "complete")))
            except Exception as exc:
                self.queue.put(("analysis_error", (name, token, select, str(exc))))

        self._thread(work)

    def _show_analysis_error(self, name, text, select=False):
        tab = ANALYSIS_TABS[name]
        self.panels[tab].message(f"Analysis could not be completed\n\n{text}", fontsize=10)
        if select:
            self.nb.select(self.panels[tab].frame)
        self.status_var.set(f"{tab}: {text.splitlines()[0]}")

    def _on_tab_changed(self, _event=None):
        if self.current is None:
            return
        try:
            tab = self.nb.tab(self.nb.select(), "text")
        except tk.TclError:
            return
        name = next((n for n, t in ANALYSIS_TABS.items() if t == tab), None)
        if name is None or self.analysis_armed[name]:
            return
        if name == "phase":
            if not self.phase_period_var.get().strip() and "bls" in self.analysis_results:
                bls = self.analysis_results["bls"]
                self.phase_period_var.set(f"{bls['period']:.10g}")
                self.phase_epoch_var.set(f"{bls['transit_time']:.8f}")
            if not self.phase_period_var.get().strip():
                self.panels["Phase Fold"].message("Enter a phase period, or run BLS first.")
                return
        self._dispatch(name, auto=True, select=False)

    def _dispatch(self, name, **kw):
        {"periodogram": self.run_periodogram, "bls": self.run_bls, "phase": self.run_phase,
         "flares": self.run_flares, "rotation": self.run_rotation}[name](**kw)

    def _rerun_armed_analyses(self, force=False):
        for name in self.ANALYSES:
            if self.analysis_armed[name]:
                self._dispatch(name, auto=True, select=False, force=force)

    def _param_event(self, name):
        """A parameter field changed: re-run only that analysis, only if used and changed."""
        if self.current is not None and self.analysis_armed[name]:
            self._dispatch(name, auto=True, select=False)

    @staticmethod
    def _add_peak_table(panel, peaks, title):
        if not peaks:
            return
        lines = [title] + [f"{i}. {per:.6g} d  {pw:.4g}" for i, (per, pw) in enumerate(peaks, 1)]
        panel.ax.text(.985, .03, "\n".join(lines), transform=panel.ax.transAxes, ha="right", va="bottom",
                      fontsize=8.5, family="monospace", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.85))

    def _finish(self, name, select, result_text):
        self.result_var.set(result_text)
        if select:
            self.nb.select(self.panels[ANALYSIS_TABS[name]].frame)

    # ============================================================ periodogram
    def run_periodogram(self, auto=False, select=None, force=False):
        self._run("periodogram", lambda t, f, p: periodogram(t, f, p["min_period"], p["max_period"], top_n=6),
                  auto, select, force)

    def _show_periodogram(self, payload, select):
        params, (periods, power, best, best_power, fap, peaks) = payload
        p = self.panels["Periodogram"]
        p.clear()
        p.ax.plot(periods, power, lw=.8)
        p.ax.axvline(best, ls="--", alpha=.7, color="tab:red")
        p.ax.plot([x for x, _ in peaks], [y for _, y in peaks], "o", ms=4, color="tab:orange")
        p.ax.set_xscale("log")
        p.ax.set_xlabel("Period [days]")
        p.ax.set_ylabel("Lomb–Scargle power")
        p.ax.set_title(f"Best period = {best:.6g} d | power = {best_power:.4g} | FAP = {fap:.3g}")
        p.ax.grid(alpha=.18, which="both")
        self._add_peak_table(p, peaks, "Top LS peaks")
        p.draw()
        self.analysis_results["periodogram"] = {"best_period": best, "power": best_power, "fap": fap, "peaks": peaks}
        self.analysis_params["periodogram"] = params
        self._finish("periodogram", select, f"Lomb–Scargle: P = {best:.6g} d | power = {best_power:.5g} | "
                                            f"FAP = {fap:.4g} | top peaks = {len(peaks)}")

    # ==================================================================== BLS
    def run_bls(self, auto=False, select=None, force=False):
        self._run("bls", lambda t, f, p: bls_search(t, f, p["min_period"], p["max_period"]), auto, select, force)

    def _show_bls(self, payload, select):
        params, (result, s) = payload
        p = self.panels["BLS Transit"]
        p.clear()
        p.ax.plot(result.period, result.power, lw=.8)
        p.ax.axvline(s["period"], ls="--", alpha=.7, color="tab:red")
        for h in (0.5, 2.0):
            if result.period.min() <= s["period"] * h <= result.period.max():
                p.ax.axvline(s["period"] * h, ls=":", alpha=.5, color="tab:red")
        p.ax.set_xlabel("Period [days]")
        p.ax.set_ylabel("BLS log-likelihood")
        p.ax.set_title(f"Candidate P = {s['period']:.6g} d | depth SNR = {s['depth_snr']:.1f}")
        p.ax.grid(alpha=.18)
        depth_ppm = s["depth"] * (1e4 if self.norm_var.get() == "Percent deviation" else 1e6)
        p.ax.text(.02, .97,
                  f"Duration   {s['duration_days'] * 24:.2f} h\n"
                  f"Depth      {depth_ppm:,.0f} ppm\n"
                  f"Epoch      {s['transit_time']:.6f} BTJD\n"
                  f"Transits   {s['n_transits']}\n"
                  f"Odd/even   {s['odd_even_sigma']:.1f} σ difference\n"
                  f"Candidate only — vet before interpreting",
                  transform=p.ax.transAxes, va="top", fontsize=9, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.85))
        p.draw()
        self.analysis_results["bls"] = s
        self.analysis_params["bls"] = params
        self.phase_period_var.set(f"{s['period']:.10g}")
        self.phase_epoch_var.set(f"{s['transit_time']:.8f}")
        flag = " | odd/even mismatch!" if s["odd_even_sigma"] > 3 else ""
        self._finish("bls", select, f"BLS candidate: P = {s['period']:.6g} d | depth = {depth_ppm:,.0f} ppm | "
                                    f"duration = {s['duration_days'] * 24:.2f} h | SNR = {s['depth_snr']:.1f}{flag} "
                                    f"| candidate only — period sent to Phase Fold")
        # Transfer the candidate to Phase Fold without switching tabs
        self.run_phase(auto=True, select=False, force=True)

    # ============================================================= phase fold
    def run_phase(self, auto=False, select=None, force=False):
        self._run("phase", lambda t, f, p: phase_fold(t, f, p["period"], p["epoch"], p["bin_size"]),
                  auto, select, force)

    def _show_phase(self, payload, select):
        params, (phase, flux, epoch, binned) = payload
        p = self.panels["Phase Fold"]
        p.clear()
        p.ax.plot(phase, flux, ".", ms=1.6, alpha=.45, color="tab:blue", rasterized=True, label="Data")
        if binned is not None and len(binned[0]):
            p.ax.plot(binned[0], binned[1], "-o", lw=1.4, ms=2.5, color="tab:orange", label="Binned median")
        p.ax.set_xlabel("Phase")
        p.ax.set_ylabel(self._flux_label())
        p.ax.set_title(f"Phase-folded | P = {params['period']:.8g} d | epoch = {epoch:.6f}")
        p.ax.grid(alpha=.18)
        p.ax.legend(loc="lower right", fontsize=8)
        p.draw()
        self.analysis_results["phase"] = {"period": params["period"], "epoch": epoch,
                                          "binned_phase": None if binned is None else binned[0],
                                          "binned_flux": None if binned is None else binned[1]}
        self.analysis_params["phase"] = params
        if select or self.nb.tab(self.nb.select(), "text") == "Phase Fold":
            self._finish("phase", select, f"Phase fold | P = {params['period']:.8g} d | epoch = {epoch:.7f} | "
                                          f"binned points = {0 if binned is None else len(binned[0])}")

    # ================================================================= flares
    def run_flares(self, auto=False, select=None, force=False):
        percent = self.norm_var.get() == "Percent deviation"

        def compute(t, f, p):
            rel = 1.0 + f / 100.0 if percent else f
            events = detect_flares(t, rel, p["threshold"], p["min_points"])
            for e in events:
                e["plot_flux"] = (e["peak_flux"] - 1.0) * 100.0 if percent else e["peak_flux"]
            return events

        self._run("flares", compute, auto, select, force)

    def _show_flares(self, payload, select):
        params, events = payload
        p = self.panels["Flares"]
        p.clear()
        t, f = self._analysis_window()
        x, y = self._display_slice(t, f)
        p.ax.plot(x, y, lw=.6, rasterized=True)
        if events:
            p.ax.scatter([e["time"] for e in events], [e["plot_flux"] for e in events], s=36,
                         facecolors="none", edgecolors="tab:red", lw=1.3, zorder=3)
            total_ed = sum(e["equivalent_duration_days"] for e in events) * 86400
            p.ax.text(.02, .97, f"Candidates = {len(events)}\nTotal equivalent duration = {total_ed:,.1f} s",
                      transform=p.ax.transAxes, va="top", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.85))
        p.ax.set_title(f"Flare candidates: {len(events)} (≥ {params['threshold']:g} σ, "
                       f"≥ {params['min_points']} points)")
        p.ax.set_xlabel("Time [BJD − 2457000]")
        p.ax.set_ylabel(self._flux_label())
        p.ax.grid(alpha=.18)
        p.ax.ticklabel_format(useOffset=False, axis="x")
        p.draw()
        self.analysis_results["flares"] = [{k: v for k, v in e.items() if k != "plot_flux"} for e in events]
        self.analysis_params["flares"] = params
        self._finish("flares", select, f"Flare candidates: {len(events)} | threshold = {params['threshold']:.2f} σ "
                                       f"| minimum points = {params['min_points']}")

    # =============================================================== rotation
    def run_rotation(self, auto=False, select=None, force=False):
        self._run("rotation", lambda t, f, p: periodogram(t, f, p["min_period"], p["max_period"], top_n=8),
                  auto, select, force)

    def _show_rotation(self, payload, select):
        params, (periods, power, best, best_power, fap, peaks) = payload
        p = self.panels["Rotation / Activity"]
        p.clear()
        p.ax.plot(periods, power, lw=.8)
        p.ax.axvline(best, ls="--", alpha=.7, color="tab:red", label=f"P = {best:.4g} d")
        for h, lab in ((0.5, "P/2"), (2.0, "2P")):
            if periods.min() <= best * h <= periods.max():
                p.ax.axvline(best * h, ls=":", alpha=.6, color="tab:purple", label=lab)
        p.ax.set_xscale("log")
        p.ax.set_xlabel("Period [days]")
        p.ax.set_ylabel("Lomb–Scargle power")
        p.ax.set_title(f"Rotation/activity candidate = {best:.6g} d | FAP = {fap:.3g}")
        p.ax.grid(alpha=.18, which="both")
        p.ax.legend(loc="upper left", fontsize=8)
        self._add_peak_table(p, peaks, "Top activity peaks")
        p.draw()
        self.analysis_results["rotation"] = {"best_period": best, "power": best_power, "fap": fap, "peaks": peaks}
        self.analysis_params["rotation"] = params
        self._finish("rotation", select, f"Rotation/activity candidate: P = {best:.6g} d | FAP = {fap:.4g} | "
                                         f"check P/2 and 2P aliases")

    # ================================================================= export
    def _default_name(self, suffix):
        if self.current is None:
            return f"lightcurve{suffix}"
        return f"TIC{self.current.tic}_S{'-'.join(map(str, self.active_sectors))}{suffix}"

    def _export(self, what, suffix, filetypes, writer):
        if self.current is None:
            messagebox.showinfo(APP_NAME, "Generate or open a light curve first.")
            return None
        ext = filetypes[0][1].lstrip("*")
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=filetypes,
                                            initialfile=self._default_name(suffix), title=f"Save {what}")
        if not path:
            return None
        try:
            writer(path)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not save {what}:\n\n{exc}")
            return None
        self.status_var.set(f"Saved {Path(path).name}")
        return path

    def save_csv(self):
        return self._export("CSV", ".csv", [("CSV", "*.csv")], lambda p: export_csv(self.current, p))

    def save_fits(self):
        return self._export("FITS", ".fits", [("FITS", "*.fits")], lambda p: export_fits(self.current, p))

    def save_metadata(self):
        return self._export("metadata", "_metadata.json", [("JSON", "*.json")],
                            lambda p: export_metadata(self.current, p))

    def analysis_payload(self):
        def clean(obj):
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            if isinstance(obj, float) and not np.isfinite(obj):
                return None
            if isinstance(obj, np.ndarray):
                return clean(obj.tolist())
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            return obj

        try:
            rng = self._range_values()
        except LCError:
            rng = (None, None)
        return clean({
            "software": f"{APP_NAME} v{__version__}",
            "tic": self.current.tic,
            "sectors": self.active_sectors,
            "data_source": self.data_source,
            "time_system": "BTJD = BJD_TDB - 2457000",
            "analysis_range_btjd": list(rng),
            "light_curve_config": self.current.metadata.get("config", {}),
            "parameters": self.analysis_params,
            "results": self.analysis_results,
        })

    def save_analysis(self):
        return self._export("analysis results", "_analysis.json", [("JSON", "*.json")],
                            lambda p: Path(p).write_text(json.dumps(self.analysis_payload(), indent=2),
                                                         encoding="utf-8"))

    def save_current_plot(self):
        tab = self.nb.tab(self.nb.select(), "text")
        p = self.panels.get(tab)
        if p is None or self.current is None:
            messagebox.showinfo(APP_NAME, "Generate or open a light curve first.")
            return None
        slug = tab.lower().replace(" / ", "_").replace(" ", "_")
        return self._export("plot", f"_{slug}.png", [("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                            lambda path: p.fig.savefig(path, dpi=300, bbox_inches="tight"))


if __name__ == "__main__":
    App().mainloop()
