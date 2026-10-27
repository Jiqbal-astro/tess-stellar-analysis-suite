"""Offline stand-in for the `eleanor` package, used only by the test-suite.

It mimics the parts of the Eleanor API that the suite uses (Source,
TargetData, multi_sectors) and returns synthetic TESS-like light curves with
known signals, so the full GUI workflow can be tested without network access.

Injected signals (all sectors):
    planet transit  P = 3.1 d, depth = 3000 ppm, duration = 0.12 d, T0 = 2000.30
    rotation        P = 5.3 d, semi-amplitude = 0.8 %
    flares          three per sector
    systematics     slow ramp, a mid-sector gap, momentum dumps flagged in QUALITY
"""
import numpy as np

SECTORS = (14, 41, 54)
SECTOR_START = {14: 1683.4, 41: 2419.0, 54: 2770.4}
PLANET = dict(period=3.1, depth=3000e-6, duration=0.12, t0=2000.30)
ROTATION = dict(period=5.3, amplitude=0.008)
CALLS = []


def _sector_for(tic, sector):
    if sector is None:
        return SECTORS[-1]
    if int(sector) not in SECTORS:
        raise ValueError(f"TIC {tic} was not observed in Sector {sector}")
    return int(sector)


class Source:
    def __init__(self, tic=None, sector=None, **kwargs):
        if tic is None or int(tic) == 404:
            raise ValueError("No TESS data found for this target")
        self.tic = int(tic)
        self.sector = _sector_for(tic, sector)
        self.gaia = 1234567890123456789
        self.tess_mag = 10.3
        self.coords = (68.3, -64.7)
        self.camera, self.chip = 3, 2
        CALLS.append(("Source", self.tic, self.sector))


def synthetic_sector(tic, sector):
    rng = np.random.default_rng(1000 * int(tic) % 99991 + sector)
    start = SECTOR_START[sector]
    t = np.arange(start, start + 27.0, 10.0 / 1440.0)
    gap = (t > start + 13.0) & (t < start + 14.2)  # orbit downlink gap
    t = t[~gap]
    phase = ((t - PLANET["t0"]) / PLANET["period"] + 0.5) % 1.0 - 0.5
    transit = np.abs(phase * PLANET["period"]) < PLANET["duration"] / 2
    star = 1.0 + ROTATION["amplitude"] * np.sin(2 * np.pi * t / ROTATION["period"])
    star -= PLANET["depth"] * transit
    for tf in (start + 4.2, start + 11.7, start + 21.3):        # flares
        after = t >= tf
        star[after] += 0.02 * np.exp(-(t[after] - tf) / 0.01)
    ramp = 1.0 + 0.004 * (t - start) / 27.0
    corr = 50000.0 * star * (1 + rng.normal(0, 600e-6, t.size))
    raw = corr * ramp + rng.normal(0, 20, t.size)
    quality = np.zeros(t.size, dtype=np.int64)
    dumps = np.arange(300, t.size, 450)
    quality[dumps] = 32                       # momentum dumps
    raw[dumps] *= 0.97
    corr[dumps] *= 0.985
    quality[5:8] = 4                          # coarse point
    return t, raw, corr, quality


class TargetData:
    def __init__(self, source, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False, **kw):
        t, raw, corr, q = synthetic_sector(source.tic, source.sector)
        self.time, self.raw_flux, self.corr_flux, self.quality = t, raw, corr, q
        self.flux_err = np.full(t.size, 30.0)
        self.pca_flux = corr * 1.0001 if do_pca else None
        CALLS.append(("TargetData", source.tic, source.sector, do_pca))


def multi_sectors(sectors="all", tic=None, tc=False, **kw):
    if tic is None or int(tic) == 404:
        raise ValueError("No TESS data found for this target")
    return [Source(tic=tic, sector=s) for s in SECTORS]
