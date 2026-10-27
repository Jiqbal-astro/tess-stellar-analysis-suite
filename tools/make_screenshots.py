"""Regenerate the README screenshots from the running GUI.

Uses the offline Eleanor test double (synthetic star with a 3.1-d planet,
5.3-d spot rotation and flares), so it needs no network access.

    xvfb-run -a -s "-screen 0 1500x930x24" python tools/make_screenshots.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests" / "fake_eleanor")]

from tess_lc_generator import app as A  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
for kind in ("showerror", "showwarning", "showinfo"):
    setattr(A.messagebox, kind, lambda *a, **k: None)

app = A.App()
app.geometry("1500x930+0+0")


def pump(cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            for _ in range(10):
                app.update()
                time.sleep(0.03)
            return
        time.sleep(0.02)
    raise RuntimeError("timeout")


def shot(name):
    app.update()
    time.sleep(0.3)
    app.update()
    subprocess.run(["import", "-window", "root", "-crop", "1500x930+0+0", str(OUT / name)], check=True)
    print("saved", name)


def run(name, fn):
    n = app.analysis_count[name]
    fn()
    pump(lambda: app.analysis_count[name] > n)


app.tic_var.set("261136679")
app.all_sectors_var.set(False)
app.generate_quick()
pump(lambda: app.current is not None and not app.busy_data)
shot("01_light_curve.png")

run("rotation", app.run_rotation)
shot("02_rotation.png")

app.detrend_var.set("Median filter")
app._apply_processing()
pump(lambda: app.current.metadata["config"]["detrend"] == "Median filter")
app.nb.select(app.panels["Light Curve"].frame)
pump(lambda: True)
shot("03_light_curve_detrended.png")

run("periodogram", app.run_periodogram)
shot("04_periodogram.png")
run("bls", app.run_bls)
pump(lambda: "phase" in app.analysis_results)
shot("05_bls.png")
run("phase", app.run_phase)
shot("06_phase_fold.png")
run("flares", app.run_flares)
shot("07_flares.png")
app.nb.select(app.panels["Statistics"].frame)
pump(lambda: True)
shot("08_statistics.png")
app.destroy()
