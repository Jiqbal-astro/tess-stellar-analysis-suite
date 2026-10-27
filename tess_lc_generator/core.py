"""Core data handling and analysis routines for the TESS Stellar Analysis Suite.

Everything in this module is GUI-independent so that it can be used from
scripts and tested without a display. Eleanor (and therefore TensorFlow) is
imported lazily, only when TESS data are actually retrieved.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares, LombScargle
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter

LOGGER = logging.getLogger(__name__)

BTJD_OFFSET = 2457000.0  # TESS time convention: BTJD = BJD - 2457000

FLUX_TYPES = ("Raw Flux", "Corrected Flux", "PCA Flux")
QUALITY_MODES = ("Strict (quality == 0)", "Remove common artifacts", "Allow flagged cadences")
NORMALIZATIONS = ("Median", "Mean", "Percent deviation", "None")
DETREND_METHODS = ("None", "Savitzky-Golay", "Median filter")


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class LCConfig:
    flux_type: str = "Corrected Flux"
    quality_mode: str = "Strict (quality == 0)"
    normalize: str = "Median"
    detrend: str = "None"
    detrend_window: int = 101
    sigma_clip: float = 0.0
    aperture_height: int = 15
    aperture_width: int = 15
    background_size: int = 31


@dataclass
class RawSectorData:
    tic: int
    sector: int
    time: np.ndarray
    raw_flux: np.ndarray
    corr_flux: np.ndarray
    pca_flux: Optional[np.ndarray]
    quality: np.ndarray
    flux_err: Optional[np.ndarray]
    metadata: dict


@dataclass
class LightCurveData:
    tic: int
    sector: int
    time: np.ndarray
    flux: np.ndarray
    raw_flux: np.ndarray
    corr_flux: np.ndarray
    pca_flux: Optional[np.ndarray]
    quality: np.ndarray
    flux_err: Optional[np.ndarray]
    metadata: dict


class LCError(RuntimeError):
    """Error with a message suitable for showing to the user."""


# --------------------------------------------------------------------------
# Input validation and light-curve processing
# --------------------------------------------------------------------------
def validate_tic(value: str) -> int:
    value = str(value).strip()
    if value.upper().startswith("TIC"):
        value = value[3:].strip()
    if not re.fullmatch(r"\d+", value):
        raise LCError("TIC ID must contain digits only (for example 261136679).")
    tic = int(value)
    if tic <= 0:
        raise LCError("TIC ID must be a positive integer.")
    return tic


def _finite_mask(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def quality_mask(quality: np.ndarray, mode: str) -> np.ndarray:
    q = np.asarray(quality)
    if mode == "Strict (quality == 0)":
        return q == 0
    if mode == "Allow flagged cadences":
        return np.ones(len(q), dtype=bool)
    if mode == "Remove common artifacts":
        # TESS spacecraft-event flags (Science Data Products Description Doc.):
        # 1 attitude tweak, 2 safe mode, 4 coarse point, 8 earth point,
        # 16 Argabrightening, 32 momentum dump, 128 manual exclude
        common_bits = 1 | 2 | 4 | 8 | 16 | 32 | 128
        return (q.astype(np.int64) & common_bits) == 0
    raise LCError(f"Unknown quality mode: {mode}")


def normalize_flux(flux: np.ndarray, mode: str) -> np.ndarray:
    f = np.asarray(flux, dtype=float)
    if mode == "None":
        return f.copy()
    if mode not in NORMALIZATIONS:
        raise LCError(f"Unknown normalization: {mode}")
    scale = np.nanmean(f) if mode == "Mean" else np.nanmedian(f)
    if not np.isfinite(scale) or scale == 0:
        raise LCError("Cannot normalize flux because the normalization scale is zero or invalid.")
    if mode == "Percent deviation":
        return (f / scale - 1.0) * 100.0
    return f / scale


def _odd_window(requested: int, n: int) -> int:
    w = max(5, int(requested))
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 5:
        raise LCError("Not enough data points for detrending.")
    return w


def _segments(time, gap_days=0.5):
    """Index slices of continuous data, split at gaps longer than `gap_days`."""
    if time is None or len(time) < 2:
        return [slice(0, None if time is None else len(time))]
    breaks = np.where(np.diff(time) > gap_days)[0] + 1
    edges = np.r_[0, breaks, len(time)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def detrend_flux(flux: np.ndarray, method: str, window: int, time: Optional[np.ndarray] = None) -> np.ndarray:
    """Divide the flux by a running trend. Returns flux relative to the trend (~1).

    When `time` is given, each continuous segment (split at gaps > 0.5 d, such
    as the mid-sector downlink) is detrended on its own, so the filter never
    bridges a gap and creates artefacts at the segment edges.
    """
    flux = np.asarray(flux, dtype=float)
    if method == "None":
        return flux.copy()
    if method not in ("Savitzky-Golay", "Median filter"):
        raise LCError(f"Unknown detrending method: {method}")
    if len(flux) < 7:
        raise LCError("Not enough points for detrending.")
    trend = np.full_like(flux, np.nan)
    for seg in _segments(time):
        f = flux[seg]
        if len(f) < 7:
            trend[seg] = np.nanmedian(f)
            continue
        w = _odd_window(window, len(f))
        if method == "Savitzky-Golay":
            trend[seg] = savgol_filter(f, window_length=w, polyorder=min(2, w - 2), mode="interp")
        else:
            trend[seg] = median_filter(f, size=w, mode="nearest")
    good = np.isfinite(trend) & (np.abs(trend) > 0)
    out = np.full_like(flux, np.nan)
    out[good] = flux[good] / trend[good]
    return out


def sigma_clip_mask(flux: np.ndarray, sigma: float) -> np.ndarray:
    """Symmetric robust (MAD-based) clipping. sigma <= 0 disables clipping."""
    if sigma <= 0:
        return np.ones(len(flux), dtype=bool)
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    if not np.isfinite(mad) or mad == 0:
        return np.ones(len(flux), dtype=bool)
    return np.abs(flux - med) <= sigma * 1.4826 * mad


def choose_flux(raw: RawSectorData, flux_type: str) -> np.ndarray:
    if flux_type == "Raw Flux":
        return raw.raw_flux.copy()
    if flux_type == "Corrected Flux":
        return raw.corr_flux.copy()
    if flux_type == "PCA Flux":
        if raw.pca_flux is None:
            raise LCError("PCA Flux is not available for this data set.")
        return raw.pca_flux.copy()
    raise LCError(f"Unsupported flux type: {flux_type}")


def process_raw(raw: RawSectorData, config: LCConfig) -> LightCurveData:
    """Quality filter -> detrend -> normalize -> sigma clip.

    Detrending divides by a running trend and is therefore done on the
    positive, un-normalized flux. Doing it after a "Percent deviation"
    normalization (values around zero) would divide by numbers near zero.
    Clipping is done last so that outliers are judged against the detrended
    light curve rather than against slow stellar or instrumental trends.
    """
    chosen = choose_flux(raw, config.flux_type)
    mask = quality_mask(raw.quality, config.quality_mode)
    mask &= _finite_mask(raw.time, chosen)
    if raw.flux_err is not None:
        mask &= np.isfinite(raw.flux_err)

    def take(arr, m):
        return None if arr is None else np.asarray(arr)[m]

    time, flux = raw.time[mask], chosen[mask]
    raw_flux, corr_flux = take(raw.raw_flux, mask), take(raw.corr_flux, mask)
    pca_flux, quality, err = take(raw.pca_flux, mask), take(raw.quality, mask), take(raw.flux_err, mask)
    if len(time) < 10:
        raise LCError(f"Sector {raw.sector} has fewer than 10 usable cadences after quality filtering.")

    flux = detrend_flux(flux, config.detrend, config.detrend_window, time)
    flux = normalize_flux(flux, config.normalize)
    keep = np.isfinite(flux) & sigma_clip_mask(flux, config.sigma_clip)

    time, flux = time[keep], flux[keep]
    raw_flux, corr_flux = raw_flux[keep], corr_flux[keep]
    pca_flux, quality, err = take(pca_flux, keep), quality[keep], take(err, keep)
    if len(time) < 10:
        raise LCError(f"Sector {raw.sector} has fewer than 10 cadences left after clipping.")

    meta = dict(raw.metadata)
    meta.update({
        "sector": int(raw.sector),
        "sectors": [int(raw.sector)],
        "n_cadences": int(len(time)),
        "time_min": float(np.min(time)),
        "time_max": float(np.max(time)),
        "time_system": "BTJD = BJD_TDB - 2457000",
        "config": asdict(config),
    })
    return LightCurveData(raw.tic, raw.sector, time, flux, raw_flux, corr_flux, pca_flux, quality, err, meta)


def concatenate(lcs: list[LightCurveData]) -> LightCurveData:
    if not lcs:
        raise LCError("No light curves supplied.")
    if len(lcs) == 1:
        return lcs[0]
    order = np.argsort(np.concatenate([x.time for x in lcs]), kind="stable")

    def cat(attr):
        parts = [getattr(x, attr) for x in lcs]
        if any(p is None for p in parts):
            return None
        return np.concatenate(parts)[order]

    time = cat("time")
    meta = dict(lcs[0].metadata)
    sectors = [int(x.sector) for x in lcs]
    meta.update({
        "sector": None,
        "sectors": sectors,
        "n_cadences": int(len(time)),
        "time_min": float(time.min()),
        "time_max": float(time.max()),
    })
    return LightCurveData(lcs[0].tic, -1, time, cat("flux"), cat("raw_flux"), cat("corr_flux"),
                          cat("pca_flux"), cat("quality"), cat("flux_err"), meta)


# --------------------------------------------------------------------------
# TESS retrieval (Eleanor)
# --------------------------------------------------------------------------
def _eleanor():
    try:
        import eleanor  # noqa: WPS433 (heavy import: TensorFlow)
    except Exception as exc:  # ImportError, or photutils/TensorFlow version problems
        raise LCError(
            "Could not import Eleanor, which is needed to download TESS data.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Install the pinned environment with:  pip install -r requirements.txt\n"
            "(Local light-curve files can still be opened without Eleanor.)"
        ) from exc
    return eleanor


def _load_raw_from_target(tic: int, sector: Optional[int], config: LCConfig) -> RawSectorData:
    eleanor = _eleanor()
    label = "the most recent sector" if sector is None else f"Sector {sector}"
    try:
        source = eleanor.Source(tic=tic) if sector is None else eleanor.Source(tic=tic, sector=int(sector))
        actual_sector = getattr(source, "sector", None)
        if actual_sector is None:
            actual_sector = sector if sector is not None else -1
        data = eleanor.TargetData(source, height=config.aperture_height, width=config.aperture_width,
                                  bkg_size=config.background_size, do_psf=False,
                                  do_pca=(config.flux_type == "PCA Flux"))
    except Exception as exc:
        raise LCError(f"Could not retrieve TIC {tic}, {label}: {exc}") from exc

    def arr(name):
        value = getattr(data, name, None)
        return None if value is None else np.asarray(value, dtype=float)

    metadata = {
        "source": "eleanor",
        "tic": int(getattr(source, "tic", tic)),
        "gaia": getattr(source, "gaia", None),
        "tess_mag": getattr(source, "tess_mag", None),
        "ra_deg": float(source.coords[0]),
        "dec_deg": float(source.coords[1]),
        "sector": int(actual_sector),
        "camera": getattr(source, "camera", None),
        "ccd": getattr(source, "chip", None),
    }
    return RawSectorData(int(tic), int(actual_sector), arr("time"), arr("raw_flux"), arr("corr_flux"),
                         arr("pca_flux"), np.asarray(data.quality), arr("flux_err"), metadata)


def load_raw_sector(tic: int, sector: Optional[int], config: LCConfig, progress=None) -> RawSectorData:
    if progress:
        progress(f"Loading TIC {tic}, Sector {sector}...")
    return _load_raw_from_target(tic, sector, config)


def extract_sector(tic: int, sector: Optional[int], config: LCConfig, progress=None) -> LightCurveData:
    return process_raw(load_raw_sector(tic, sector, config, progress=progress), config)


def quick_extract(tic: int, config: LCConfig, progress=None) -> LightCurveData:
    return extract_sector(tic, None, config, progress=progress)


def discover_sectors(tic: int) -> list[int]:
    eleanor = _eleanor()
    try:
        sources = eleanor.multi_sectors("all", tic=tic, tc=True)
    except Exception as exc:
        raise LCError(f"Could not determine TESS sectors for TIC {tic}: {exc}") from exc
    sectors = sorted({int(s.sector) for s in sources if getattr(s, "sector", None) is not None})
    if not sectors:
        raise LCError(f"No TESS sectors were found for TIC {tic}.")
    return sectors


# --------------------------------------------------------------------------
# Local files
# --------------------------------------------------------------------------
def _to_btjd(time):
    t = np.asarray(time, dtype=float)
    return t - BTJD_OFFSET if np.nanmedian(t) > 2.4e6 else t


def load_light_curve_file(path: str | Path) -> RawSectorData:
    """Open a light curve saved by this suite (CSV or FITS), a TESS SPOC/QLP
    light-curve FITS file, or a generic CSV with time and flux columns."""
    path = Path(path)
    suffix = path.suffix.lower()
    tic, sector = 0, 0
    err = pca = None
    try:
        if suffix in (".fits", ".fit", ".fz"):
            with fits.open(path) as hdul:
                hdr0 = hdul[0].header
                hdu = hdul["LIGHTCURVE"] if "LIGHTCURVE" in hdul else hdul[1]
                d, hdr = hdu.data, hdu.header
                names = [n.upper() for n in d.columns.names]

                def col(*cands):
                    for c in cands:
                        if c in names:
                            return np.asarray(d[c], dtype=float)
                    return None

                time = col("TIME")
                corr = col("CORR_FLUX", "PDCSAP_FLUX", "KSPSAP_FLUX", "FLUX", "SAP_FLUX")
                raw = col("RAW_FLUX", "SAP_FLUX", "FLUX")
                pca = col("PCA_FLUX")
                err = col("FLUX_ERR", "PDCSAP_FLUX_ERR", "KSPSAP_FLUX_ERR", "SAP_FLUX_ERR")
                q = col("QUALITY")
                quality = np.zeros(len(time), dtype=np.int64) if q is None else np.nan_to_num(q).astype(np.int64)
                for h in (hdr0, hdr):
                    tic = tic or int(h.get("TICID", h.get("TIC", 0)) or 0)
                    sector = sector or int(h.get("SECTOR", 0) or 0)
        elif suffix in (".csv", ".txt", ".dat"):
            import pandas as pd
            df = pd.read_csv(path, sep=None, engine="python", comment="#")
            cols = {c.lower().strip(): c for c in df.columns}

            def col(*cands):
                for c in cands:
                    if c in cols:
                        return df[cols[c]].to_numpy(dtype=float)
                return None

            time = col("time_bjd_minus_2457000", "time", "btjd", "bjd", "bjd_tdb")
            if time is None:
                time = df.iloc[:, 0].to_numpy(dtype=float)
            corr = col("corrected_flux", "pdcsap_flux", "flux")
            raw = col("raw_flux", "sap_flux")
            pca = col("pca_flux")
            processed = col("flux_processed")
            if corr is None:
                corr = processed if processed is not None else df.iloc[:, 1].to_numpy(dtype=float)
            err = col("flux_err", "pdcsap_flux_err", "error")
            q = col("quality")
            quality = np.zeros(len(time), dtype=np.int64) if q is None else np.nan_to_num(q).astype(np.int64)
        else:
            raise LCError("Unsupported file type. Use .csv, .txt, .dat or .fits.")
    except LCError:
        raise
    except Exception as exc:
        raise LCError(f"Could not read {path.name}: {exc}") from exc

    if time is None or corr is None:
        raise LCError(f"{path.name} does not contain recognisable time and flux columns.")
    raw = corr if raw is None else raw
    time = _to_btjd(time)
    metadata = {"source": "file", "file": path.name, "tic": tic, "sector": sector}
    return RawSectorData(tic, sector, time, raw, corr, pca, quality, err, metadata)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def _clean_series(time, flux, min_points=20):
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if len(t) < min_points:
        raise LCError(f"Not enough finite cadences for this analysis (need at least {min_points}).")
    order = np.argsort(t)
    t, f = t[order], f[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, f = t[keep], f[keep]
    if len(t) < min_points:
        raise LCError(f"Not enough unique cadences for this analysis (need at least {min_points}).")
    return t, f


def robust_std(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")


def _top_peaks(x, y, count, distance):
    idx, _ = find_peaks(np.nan_to_num(y, nan=-np.inf), distance=max(1, int(distance)))
    if len(idx) == 0:
        idx = np.array([int(np.nanargmax(y))])
    idx = idx[np.argsort(y[idx])[::-1]][:count]
    return [(float(x[i]), float(y[i])) for i in idx]


def periodogram(time, flux, min_period=0.05, max_period=20.0, samples=None, top_n=5,
                oversample=5, max_samples=300_000):
    """Lomb-Scargle periodogram on a frequency grid that resolves every peak.

    Returns (periods, power, best_period, best_power, fap, peaks).
    `samples` is accepted for backward compatibility and used as a minimum
    grid size.
    """
    t, y0 = _clean_series(time, flux, 20)
    if min_period <= 0 or max_period <= min_period:
        raise LCError("Period limits must satisfy 0 < minimum < maximum.")
    baseline = float(t.max() - t.min())
    if max_period > 2.0 * baseline:
        raise LCError(f"Maximum period ({max_period:g} d) is longer than twice the time baseline "
                      f"({baseline:.2f} d). Reduce the upper period limit.")
    fmin, fmax = 1.0 / max_period, 1.0 / min_period
    df = 1.0 / (oversample * baseline)
    n = int(np.ceil((fmax - fmin) / df)) + 1
    n = int(np.clip(n, max(2000, samples or 0), max_samples))
    freq = np.linspace(fmin, fmax, n)
    y = y0 - np.nanmedian(y0)
    model = LombScargle(t, y)
    power = model.power(freq)
    periods = 1.0 / freq
    i = int(np.nanargmax(power))
    best, best_power = float(periods[i]), float(power[i])
    try:
        fap = float(model.false_alarm_probability(best_power, minimum_frequency=fmin,
                                                  maximum_frequency=fmax))
    except Exception:
        fap = float("nan")
    resolution_bins = max(1, int(round(n * df / (fmax - fmin))))  # grid points per 1/T
    peaks = _top_peaks(periods, power, top_n, distance=2 * resolution_bins)
    return periods, power, best, best_power, fap, peaks


def bls_search(time, flux, min_period=0.5, max_period=20.0, samples=None, coarse_samples=20_000):
    """Box Least Squares transit search.

    Returns (result, summary) where `result` is the astropy BoxLeastSquaresResults
    and `summary` is a dict with the best candidate and simple vetting metrics.
    """
    t, y = _clean_series(time, flux, 30)
    if min_period <= 0 or max_period <= min_period:
        raise LCError("BLS limits must satisfy 0 < minimum < maximum.")
    baseline = float(t.max() - t.min())
    if max_period >= baseline:
        raise LCError(f"Maximum BLS period ({max_period:g} d) must be shorter than the time "
                      f"baseline ({baseline:.2f} d) so that at least two transits can be seen.")

    cadence = float(np.median(np.diff(t)))
    dmin = max(0.02, 2.0 * cadence)                  # >= ~30 min, >= 2 cadences
    dmax = min(0.4, 0.25 * min_period)                # <= ~10 h; astropy needs d < P_min
    if dmax <= dmin:
        dmax = min(0.9 * min_period, dmin * 2.0)
    if dmax <= dmin:
        raise LCError("Minimum BLS period is too short for the cadence of these data.")
    durations = np.geomspace(dmin, dmax, 12)

    # Two-stage frequency grid. The coarse pass locates the strongest signal;
    # the fine pass resolves it to the precision the full baseline allows
    # (transit drift < shortest duration over the whole baseline). Without
    # refinement, periods from multi-sector data drift out of phase when folded.
    fmin, fmax = 1.0 / max_period, 1.0 / min_period
    df_fine = dmin / baseline**2
    n_needed = int(np.ceil((fmax - fmin) / df_fine)) + 1
    n_coarse = int(np.clip(n_needed, max(2000, samples or 0), coarse_samples))
    freq = np.linspace(fmin, fmax, n_coarse)
    dy = np.full_like(y, robust_std(y) or 1.0)
    model = BoxLeastSquares(t, y, dy=dy)
    result = model.power(np.sort(1.0 / freq), durations, objective="likelihood")
    i = int(np.nanargmax(result.power))
    refined = False
    if n_needed > n_coarse:
        f_best = 1.0 / float(result.period[i])
        step = (fmax - fmin) / (n_coarse - 1)
        lo, hi = max(fmin, f_best - 3 * step), min(fmax, f_best + 3 * step)
        n_fine = int(np.clip(np.ceil((hi - lo) / df_fine) + 1, 200, 20000))
        fine = model.power(np.sort(1.0 / np.linspace(lo, hi, n_fine)), durations, objective="likelihood")
        j = int(np.nanargmax(fine.power))
        refined = True
        best, duration, t0 = float(fine.period[j]), float(fine.duration[j]), float(fine.transit_time[j])
        depth, depth_err, depth_snr, power = (float(fine.depth[j]), float(fine.depth_err[j]),
                                              float(fine.depth_snr[j]), float(fine.power[j]))
    else:
        best, duration, t0 = float(result.period[i]), float(result.duration[i]), float(result.transit_time[i])
        depth, depth_err, depth_snr, power = (float(result.depth[i]), float(result.depth_err[i]),
                                              float(result.depth_snr[i]), float(result.power[i]))
    stats = model.compute_stats(best, duration, t0)
    odd, even, oe_sigma = _odd_even_test(t, y, best, duration, t0)

    summary = {
        "period": best,
        "power": power,
        "duration_days": duration,
        "depth": depth,
        "depth_err": depth_err,
        "depth_snr": depth_snr,
        "transit_time": t0,
        "n_transits": int(np.sum(np.asarray(stats["per_transit_count"]) > 0)),
        "depth_odd": odd,
        "depth_even": even,
        "odd_even_sigma": oe_sigma,
        "grid_size": int(n_coarse),
        "refined": refined,
    }
    return result, summary


def _odd_even_test(t, y, period, duration, t0):
    """Compare mean odd and even transit depths.

    The uncertainty comes from the scatter of the individual transit depths
    rather than from the per-point white noise, so residual stellar
    variability (correlated noise) does not produce false alarms.
    """
    epoch = np.round((t - t0) / period).astype(int)
    dt = t - (t0 + epoch * period)
    depths, parity = [], []
    for e in np.unique(epoch):
        m = epoch == e
        inside = m & (np.abs(dt) < 0.4 * duration)
        around = m & (np.abs(dt) > 0.75 * duration) & (np.abs(dt) < 2.5 * duration)
        if inside.sum() >= 3 and around.sum() >= 6:
            depths.append(np.median(y[around]) - np.median(y[inside]))
            parity.append(e % 2)
    depths, parity = np.asarray(depths), np.asarray(parity)
    if len(depths) < 4 or parity.min() == parity.max():
        return float("nan"), float("nan"), float("nan")
    odd, even = depths[parity == 1], depths[parity == 0]
    scatter = np.std(depths - np.where(parity == 1, odd.mean(), even.mean()), ddof=2)
    err = scatter * np.sqrt(1 / len(odd) + 1 / len(even))
    sigma = float(abs(odd.mean() - even.mean()) / err) if err > 0 else float("nan")
    return float(odd.mean()), float(even.mean()), sigma


def phase_fold(time, flux, period, epoch=None, bin_size=None):
    t, f = _clean_series(time, flux, 2)
    if period <= 0:
        raise LCError("Period must be positive.")
    if epoch is None:
        epoch = float(np.nanmedian(t))
    phase = ((t - epoch) / period + 0.5) % 1.0 - 0.5
    order = np.argsort(phase)
    phase, f = phase[order], f[order]
    binned = None
    if bin_size is not None and bin_size > 0:
        edges = np.arange(-0.5, 0.5 + bin_size, bin_size)
        idx = np.digitize(phase, edges) - 1
        bx, by = [], []
        for k in range(len(edges) - 1):
            m = idx == k
            if np.any(m):
                bx.append(float(np.mean(phase[m])))
                by.append(float(np.median(f[m])))
        binned = (np.asarray(bx), np.asarray(by))
    return phase, f, float(epoch), binned


def detect_flares(time, flux, threshold=5.0, min_points=2, baseline=None):
    """Flag runs of consecutive cadences that lie `threshold` robust sigma
    above the median. `baseline` is the flux level corresponding to the
    quiescent star (1.0 for normalized flux); if None the median is used."""
    t, f = _clean_series(time, flux, 3)
    med = float(np.nanmedian(f))
    sigma = robust_std(f)
    if not np.isfinite(sigma) or sigma == 0:
        return []
    ref = med if baseline is None else float(baseline)
    z = (f - med) / sigma
    idx = np.where(z >= threshold)[0]
    if len(idx) == 0:
        return []
    dt = float(np.nanmedian(np.diff(t)))
    # Split into events at non-consecutive indices or at data gaps
    breaks = np.where((np.diff(idx) > 1) | (np.diff(t[idx]) > 3 * dt))[0] + 1
    events = []
    for g in np.split(idx, breaks):
        if len(g) < min_points:
            continue
        i = int(g[np.argmax(z[g])])
        excess = np.maximum(f[g] - med, 0.0) / abs(ref) if ref != 0 else np.full(len(g), np.nan)
        events.append({
            "index": i,
            "time": float(t[i]),
            "peak_flux": float(f[i]),
            "sigma": float(z[i]),
            "n_points": int(len(g)),
            "start_time": float(t[g[0]]),
            "end_time": float(t[g[-1]]),
            "duration_days": float(t[g[-1]] - t[g[0]] + dt),
            "amplitude": float((f[i] - med) / abs(ref)) if ref != 0 else float("nan"),
            "equivalent_duration_days": float(np.nansum(excess) * dt),
        })
    return events


def summary_stats(lc: LightCurveData) -> dict:
    good = np.isfinite(lc.flux) & np.isfinite(lc.time)
    f, t = lc.flux[good], lc.time[good]
    if len(f) == 0:
        return {}
    med = float(np.median(f))
    std = float(np.std(f))
    mad = float(np.median(np.abs(f - med)))
    norm = lc.metadata.get("config", {}).get("normalize", "Median")
    # Convert scatter to parts-per-million of the stellar flux
    if norm == "Percent deviation":
        to_ppm = 1e4
    elif med != 0:
        to_ppm = 1e6 / abs(med)
    else:
        to_ppm = float("nan")
    p2p = float(np.median(np.abs(np.diff(f))) * 1.4826 / np.sqrt(2)) if len(f) > 2 else float("nan")
    cadence = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    return {
        "n": int(len(f)), "median": med, "mean": float(np.mean(f)), "std": std, "mad": mad,
        "min": float(np.min(f)), "max": float(np.max(f)),
        "rms_ppm": std * to_ppm, "robust_rms_ppm": 1.4826 * mad * to_ppm,
        "point_to_point_ppm": p2p * to_ppm,
        "time_start": float(t.min()), "time_end": float(t.max()),
        "time_span_days": float(t.max() - t.min()),
        "cadence_minutes": cadence * 24 * 60,
        "n_sectors": len(lc.metadata.get("sectors", [lc.sector])),
    }


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_csv(lc: LightCurveData, path: str | Path):
    cols = [lc.time, lc.flux, lc.raw_flux, lc.corr_flux, lc.quality]
    names = ["time_bjd_minus_2457000", "flux_processed", "raw_flux", "corrected_flux", "quality"]
    if lc.pca_flux is not None:
        cols.append(lc.pca_flux)
        names.append("pca_flux")
    if lc.flux_err is not None:
        cols.append(lc.flux_err)
        names.append("flux_err")
    np.savetxt(Path(path), np.column_stack(cols), delimiter=",", header=",".join(names),
               comments="", fmt="%.10g")


def _fits_value(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value[:60] if isinstance(value, str) else value
    return None


def export_fits(lc: LightCurveData, path: str | Path):
    cols = [fits.Column(name="TIME", format="D", array=lc.time, unit="d"),
            fits.Column(name="FLUX", format="D", array=lc.flux),
            fits.Column(name="RAW_FLUX", format="D", array=lc.raw_flux),
            fits.Column(name="CORR_FLUX", format="D", array=lc.corr_flux),
            fits.Column(name="QUALITY", format="K", array=np.asarray(lc.quality).astype(np.int64))]
    if lc.pca_flux is not None:
        cols.append(fits.Column(name="PCA_FLUX", format="D", array=lc.pca_flux))
    if lc.flux_err is not None:
        cols.append(fits.Column(name="FLUX_ERR", format="D", array=lc.flux_err))
    table = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    table.header["TIMEREF"] = ("BTJD", "BJD_TDB - 2457000")
    table.header["TICID"] = int(lc.tic)
    for key, value in lc.metadata.items():
        if key in ("config", "sectors"):
            continue
        v = _fits_value(value)
        if v is None:
            continue
        try:
            table.header[f"HIERARCH {key.upper()}"] = v
        except Exception:
            LOGGER.debug("Skipping FITS header key %s", key)
    table.header["SECTORS"] = ",".join(str(s) for s in lc.metadata.get("sectors", [lc.sector]))[:60]
    for key, value in lc.metadata.get("config", {}).items():
        try:
            table.header[f"HIERARCH CFG_{key.upper()}"] = _fits_value(value)
        except Exception:
            pass
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def export_metadata(lc: LightCurveData, path: str | Path):
    Path(path).write_text(json.dumps(lc.metadata, indent=2, default=str), encoding="utf-8")
