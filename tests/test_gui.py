"""End-to-end GUI test: drives the real Tk application by invoking its
buttons, with the offline Eleanor test double and patched dialogs.

Run on a machine without a display with:  xvfb-run -a pytest tests/test_gui.py
"""
import json
import os
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

import eleanor  # test double
from tess_lc_generator import app as app_module

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (use xvfb-run)")

TIC = "261136679"


class Dialogs:
    """Records message boxes instead of blocking; answers file dialogs."""

    def __init__(self):
        self.messages = []
        self.save_path = None
        self.open_path = None

    def install(self, monkeypatch):
        for kind in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(app_module.messagebox, kind,
                                lambda title, msg, kind=kind: self.messages.append((kind, title, msg)))
        monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: self.save_path)
        monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: self.open_path)

    def errors(self):
        return [m for m in self.messages if m[0] == "showerror"]


@pytest.fixture
def gui(monkeypatch):
    dialogs = Dialogs()
    dialogs.install(monkeypatch)
    eleanor.CALLS.clear()
    application = app_module.App()
    application.dialogs = dialogs
    yield application
    application.destroy()


def buttons(root):
    found = {}
    stack = [root]
    while stack:
        w = stack.pop()
        if isinstance(w, ttk.Button):
            found[w.cget("text")] = w
        stack.extend(w.winfo_children())
    return found


def click(app, text):
    btn = buttons(app)[text]
    assert str(btn.cget("state")) != "disabled", f"button '{text}' is disabled"
    btn.invoke()


def pump(app, until, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if until():
            for _ in range(5):
                app.update()
                time.sleep(0.02)
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the GUI")


def idle(app):
    pump(app, lambda: not app.busy_data)


def current_tab(app):
    return app.nb.tab(app.nb.select(), "text")


def generate(app, tic=TIC, all_sectors=False):
    app.tic_var.set(tic)
    app.all_sectors_var.set(all_sectors)
    click(app, "Generate Light Curve")
    idle(app)


def wait_result(app, name, after=0):
    pump(app, lambda: app.analysis_count[name] > after)


def count(app, name):
    return app.analysis_count[name]


def set_combo(app, var, value):
    var.set(value)
    app._auto_from_event()
    gen = app._processing_generation
    pump(app, lambda: app._processing_generation > gen or app._auto_job is None)
    pump(app, lambda: app.current.metadata["config"][_config_key(var, app)] == value)


def _config_key(var, app):
    return {str(app.flux_var): "flux_type", str(app.quality_var): "quality_mode",
            str(app.norm_var): "normalize", str(app.detrend_var): "detrend"}[str(var)]


# --------------------------------------------------------------------------
def test_every_button_exists(gui):
    expected = {"Generate Light Curve", "Find Other TESS Sectors", "Open Local Light Curve…", "Select all",
                "Load selected", "Full range", "Run Periodogram", "Run BLS", "Phase Fold", "Screen Flares",
                "Rotation / Activity", "Export CSV", "Export FITS", "Export metadata JSON",
                "Export analysis JSON", "Save current plot"}
    assert expected <= set(buttons(gui))


def test_invalid_tic_shows_error(gui):
    gui.tic_var.set("abc")
    click(gui, "Generate Light Curve")
    assert gui.dialogs.errors()[-1][1] == "Input Error"


def test_unknown_target_reports_error_and_recovers(gui):
    generate(gui, "404")
    assert "No TESS data" in gui.dialogs.errors()[-1][2]
    generate(gui)                       # buttons re-enabled, next load works
    assert gui.current is not None


def test_full_workflow_buttons(gui, tmp_path):
    generate(gui)
    assert gui.active_sectors == [54]
    assert gui.panels["Light Curve"].ax.has_data()
    assert current_tab(gui) == "Light Curve"

    # Rotation on undetrended light curve finds the 5.3-d spot signal
    click(gui, "Rotation / Activity")
    wait_result(gui, "rotation")
    assert gui.analysis_results["rotation"]["best_period"] == pytest.approx(5.3, rel=0.05)
    assert current_tab(gui) == "Rotation / Activity"

    click(gui, "Run Periodogram")
    wait_result(gui, "periodogram")
    assert current_tab(gui) == "Periodogram"

    # Detrend for the transit search; armed analyses re-run without tab jumping
    n = count(gui, "periodogram")
    set_combo(gui, gui.detrend_var, "Median filter")
    wait_result(gui, "periodogram", n)
    assert current_tab(gui) == "Periodogram"

    click(gui, "Run BLS")
    wait_result(gui, "bls")
    bls = gui.analysis_results["bls"]
    assert bls["period"] == pytest.approx(3.1, rel=0.005)
    assert float(gui.phase_period_var.get()) == pytest.approx(bls["period"])
    wait_result(gui, "phase")                      # candidate transferred to Phase Fold
    assert current_tab(gui) == "BLS Transit"       # ...without jumping away from BLS

    click(gui, "Phase Fold")
    pump(gui, lambda: current_tab(gui) == "Phase Fold")
    assert gui.analysis_results["phase"]["period"] == pytest.approx(bls["period"])

    click(gui, "Screen Flares")
    wait_result(gui, "flares")
    assert len(gui.analysis_results["flares"]) == 3
    assert current_tab(gui) == "Flares"

    # Range: narrower window re-runs analyses; Full range restores it
    gui.bjd_min.set("2775.0")
    gui.bjd_max.set("2790.0")
    n = count(gui, "flares")
    gui._refresh_range()
    wait_result(gui, "flares", n)
    assert len(gui.analysis_results["flares"]) == 1      # only the flare at +11.7 d
    click(gui, "Full range")
    assert float(gui.bjd_min.get()) == pytest.approx(gui.current.time.min())

    # Range survives a processing change (original reset it)
    gui.bjd_min.set("2775.0")
    gui._refresh_range()
    n = count(gui, "flares")
    set_combo(gui, gui.norm_var, "Percent deviation")
    assert gui.bjd_min.get() == "2775.0"
    wait_result(gui, "flares", n)
    assert len(gui.analysis_results["flares"]) == 2      # flares at +11.7 d and +21.3 d, in % units
    click(gui, "Full range")
    set_combo(gui, gui.norm_var, "Median")

    # Every export button writes a valid file
    for text, name in (("Export CSV", "lc.csv"), ("Export FITS", "lc.fits"),
                       ("Export metadata JSON", "meta.json"), ("Export analysis JSON", "analysis.json"),
                       ("Save current plot", "plot.png")):
        gui.dialogs.save_path = str(tmp_path / name)
        click(gui, text)
        assert (tmp_path / name).stat().st_size > 0, text
    payload = json.loads((tmp_path / "analysis.json").read_text())
    assert set(payload["results"]) >= {"periodogram", "bls", "phase", "flares", "rotation"}
    assert payload["parameters"]["bls"]["min_period"] == 0.5
    assert json.loads((tmp_path / "meta.json").read_text())["tic"] == int(TIC)

    # Re-open the exported CSV with the new Open button
    gui.dialogs.open_path = str(tmp_path / "lc.csv")
    click(gui, "Open Local Light Curve…")
    idle(gui)
    assert gui.data_source == "file"
    assert len(gui.current.time) > 1000
    assert not gui.dialogs.errors()


def test_sector_buttons_and_cache(gui):
    gui.tic_var.set(TIC)
    click(gui, "Find Other TESS Sectors")
    idle(gui)
    assert gui.sector_list.get(0, tk.END) == ("14", "41", "54")
    click(gui, "Select all")
    assert len(gui.sector_list.curselection()) == 3
    click(gui, "Load selected")
    idle(gui)
    assert gui.active_sectors == [14, 41, 54]
    assert gui.current.metadata["sectors"] == [14, 41, 54]

    # Load a subset: cached data are re-used, no new download
    calls = len(eleanor.CALLS)
    gui.sector_list.selection_clear(0, tk.END)
    gui.sector_list.selection_set(0, 1)
    click(gui, "Load selected")
    idle(gui)
    assert gui.active_sectors == [14, 41]
    assert len(eleanor.CALLS) == calls

    # BUG FIX: deselected sector must not come back after a processing change
    set_combo(gui, gui.quality_var, "Remove common artifacts")
    assert gui.current.metadata["sectors"] == [14, 41]


def test_new_tic_never_uses_previous_targets_cache(gui):
    gui.tic_var.set(TIC)
    click(gui, "Find Other TESS Sectors")
    idle(gui)
    gui.sector_list.selection_set(1)
    click(gui, "Load selected")
    idle(gui)
    first = gui.current.flux.copy()

    # BUG FIX: original re-used Sector 41 of the previous TIC here
    gui.tic_var.set("123456")
    click(gui, "Load selected")
    assert "belongs to TIC" in gui.dialogs.errors()[-1][2]
    click(gui, "Find Other TESS Sectors")
    idle(gui)
    gui.sector_list.selection_set(1)
    click(gui, "Load selected")
    idle(gui)
    assert gui.current.tic == 123456
    assert ("TargetData", 123456, 41, False) in eleanor.CALLS
    assert not np.array_equal(first, gui.current.flux)


def test_all_sectors_checkbox(gui):
    generate(gui, all_sectors=True)
    assert gui.active_sectors == [14, 41, 54]


def test_new_target_clears_old_results(gui):
    generate(gui)
    click(gui, "Run BLS")
    wait_result(gui, "bls")
    generate(gui, "555")
    assert gui.analysis_results == {}
    assert gui.phase_period_var.get() == ""
    assert current_tab(gui) == "Light Curve"


def test_opening_a_tab_runs_its_analysis(gui):
    generate(gui)
    gui.nb.select(gui.panels["Periodogram"].frame)
    wait_result(gui, "periodogram")
    gui.nb.select(gui.panels["Phase Fold"].frame)       # no period yet: friendly message
    gui.update()
    assert "phase" not in gui.analysis_results
    gui.nb.select(gui.panels["Statistics"].frame)
    gui.update()
    assert gui.panels["Statistics"].ax.texts


def test_pca_flux_downloads_once(gui):
    generate(gui)
    set_combo(gui, gui.flux_var, "PCA Flux")
    idle(gui)
    assert gui.current.pca_flux is not None
    assert ("TargetData", int(TIC), 54, True) in eleanor.CALLS


def test_bad_analysis_parameters_show_message_not_crash(gui):
    generate(gui)
    gui.blsmin_var.set("5")
    gui.blsmax_var.set("2")
    click(gui, "Run BLS")
    texts = " ".join(t.get_text() for t in gui.panels["BLS Transit"].ax.texts)
    assert "minimum < maximum" in texts
    gui.phase_period_var.set("")
    click(gui, "Phase Fold")
    assert "Enter a phase period" in " ".join(t.get_text() for t in gui.panels["Phase Fold"].ax.texts)
    gui.window_var.set("abc")
    gui._apply_processing()
    assert "whole number" in gui.status_var.get()


def test_display_error_does_not_freeze_gui(gui):
    generate(gui)
    token = gui._job_generation["periodogram"]
    gui.queue.put(("analysis", ("periodogram", token, False, ("broken", None))))
    pump(gui, lambda: gui.status_var.get().startswith("Display error"))
    n = count(gui, "periodogram")
    click(gui, "Run Periodogram")                      # app still responds
    wait_result(gui, "periodogram", n)


def test_export_without_data_is_friendly(gui):
    click(gui, "Export CSV")
    assert gui.dialogs.messages[-1][0] == "showinfo"
