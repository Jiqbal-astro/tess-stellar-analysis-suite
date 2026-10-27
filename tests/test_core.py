"""Tests for the analysis core, run against the offline Eleanor test double."""
import json

import numpy as np
import pytest

import eleanor  # the test double from tests/fake_eleanor
from tess_lc_generator import core
from tess_lc_generator.core import LCConfig, LCError


@pytest.fixture(scope="module")
def raw():
    return core.load_raw_sector(261136679, 41, LCConfig())


@pytest.fixture(scope="module")
def detrended(raw):
    # Median filter, ~0.7 d window: the recommended setting for transit searches
    return core.process_raw(raw, LCConfig(detrend="Median filter", detrend_window=101))


# ---- input and processing ------------------------------------------------
@pytest.mark.parametrize("text,expected", [("261136679", 261136679), (" 42 ", 42), ("TIC 7", 7)])
def test_validate_tic_accepts(text, expected):
    assert core.validate_tic(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "-5", "0", "12.5"])
def test_validate_tic_rejects(text):
    with pytest.raises(LCError):
        core.validate_tic(text)


def test_strict_quality_removes_flagged(raw):
    lc = core.process_raw(raw, LCConfig())
    assert np.all(lc.quality == 0)
    assert len(lc.time) == np.sum(raw.quality == 0)


def test_common_artifacts_removes_momentum_dumps(raw):
    lc = core.process_raw(raw, LCConfig(quality_mode="Remove common artifacts"))
    assert not np.any(lc.quality & 32)


def test_allow_flagged_keeps_all(raw):
    lc = core.process_raw(raw, LCConfig(quality_mode="Allow flagged cadences"))
    assert len(lc.time) == len(raw.time)


@pytest.mark.parametrize("norm,centre", [("Median", 1.0), ("Mean", 1.0), ("Percent deviation", 0.0)])
def test_normalizations(raw, norm, centre):
    lc = core.process_raw(raw, LCConfig(normalize=norm))
    assert np.median(lc.flux) == pytest.approx(centre, abs=0.01 if centre else 0.5)


@pytest.mark.parametrize("method", ["Savitzky-Golay", "Median filter"])
@pytest.mark.parametrize("norm", ["Median", "Percent deviation", "None"])
def test_detrending_works_with_every_normalization(raw, method, norm):
    lc = core.process_raw(raw, LCConfig(detrend=method, normalize=norm))
    assert np.all(np.isfinite(lc.flux))
    centre = 0.0 if norm == "Percent deviation" else 1.0
    assert np.median(lc.flux) == pytest.approx(centre, abs=0.05)
    # rotation (0.8 %) should be largely removed
    spread = np.percentile(lc.flux, 90) - np.percentile(lc.flux, 10)
    assert spread < (0.3 if norm == "Percent deviation" else 0.003)


@pytest.mark.parametrize("method", ["Savitzky-Golay", "Median filter"])
def test_detrending_does_not_create_flares_at_gaps(raw, method):
    lc = core.process_raw(raw, LCConfig(detrend=method))
    events = core.detect_flares(lc.time, lc.flux, 5.0, 2, baseline=1.0)
    gap_end = 2419.0 + 14.2
    assert not any(abs(e["time"] - gap_end) < 0.3 for e in events)


def test_sigma_clip_reduces_points(raw):
    a = core.process_raw(raw, LCConfig(detrend="Median filter"))
    b = core.process_raw(raw, LCConfig(detrend="Median filter", sigma_clip=3))
    assert len(b.time) < len(a.time)


def test_pca_flux_requires_pca(raw):
    with pytest.raises(LCError):
        core.process_raw(raw, LCConfig(flux_type="PCA Flux"))
    pca_raw = core.load_raw_sector(261136679, 41, LCConfig(flux_type="PCA Flux"))
    assert core.process_raw(pca_raw, LCConfig(flux_type="PCA Flux")).pca_flux is not None


def test_concatenate_updates_metadata():
    lcs = [core.process_raw(core.load_raw_sector(1, s, LCConfig()), LCConfig()) for s in (14, 41)]
    lc = core.concatenate(lcs)
    assert lc.metadata["sectors"] == [14, 41]
    assert lc.metadata["time_min"] == pytest.approx(lc.time.min())
    assert lc.metadata["time_max"] == pytest.approx(lc.time.max())
    assert np.all(np.diff(lc.time) > 0)


def test_discover_sectors():
    assert core.discover_sectors(1) == [14, 41, 54]


def test_missing_target_gives_user_error():
    with pytest.raises(LCError):
        core.load_raw_sector(404, None, LCConfig())


# ---- analyses ---------------------------------------------------------------
def test_lomb_scargle_finds_rotation(raw):
    lc = core.process_raw(raw, LCConfig())
    periods, power, best, best_power, fap, peaks = core.periodogram(lc.time, lc.flux, 0.2, 20)
    assert best == pytest.approx(eleanor.ROTATION["period"], rel=0.03)
    assert fap < 1e-10
    assert len(peaks) >= 3


def test_periodogram_rejects_too_long_period(raw):
    lc = core.process_raw(raw, LCConfig())
    with pytest.raises(LCError):
        core.periodogram(lc.time, lc.flux, 0.1, 200)


def test_bls_recovers_planet(detrended):
    _, s = core.bls_search(detrended.time, detrended.flux, 0.5, 12)
    assert s["period"] == pytest.approx(eleanor.PLANET["period"], rel=0.005)
    assert s["depth"] == pytest.approx(eleanor.PLANET["depth"], rel=0.35)
    assert s["depth_snr"] > 10
    assert s["n_transits"] >= 6
    assert s["odd_even_sigma"] < 3


def test_bls_period_precise_enough_to_fold_multi_sector_data():
    # Two sectors a year apart: the period must be precise enough that the
    # transit does not drift by more than its duration across the baseline.
    rng = np.random.default_rng(20)
    t = np.r_[np.arange(0, 27, 10 / 1440), np.arange(360, 387, 10 / 1440)]
    f = 1 + rng.normal(0, 800e-6, t.size)
    period, dur = 0.71, 0.054
    phase = ((t - 0.3) / period + 0.5) % 1 - 0.5
    f[np.abs(phase * period) < dur / 2] -= 900e-6
    _, s = core.bls_search(t, f, 0.5, 12)
    drift = abs(s["period"] - period) * (t.max() - t.min()) / period
    assert s["refined"]
    assert drift < dur / 3


def test_odd_even_test_flags_eclipsing_binary_not_planet(detrended):
    _, planet = core.bls_search(detrended.time, detrended.flux, 0.5, 12)
    assert planet["odd_even_sigma"] < 3
    rng = np.random.default_rng(5)
    t = np.arange(0, 27, 10 / 1440)
    f = 1 + rng.normal(0, 600e-6, t.size)
    period = 2.4
    phase = ((t - 0.5) / period + 0.5) % 1 - 0.5
    epoch = np.round((t - 0.5) / period).astype(int)
    eclipse = np.abs(phase * period) < 0.05
    f[eclipse] -= np.where(epoch % 2 == 0, 0.006, 0.003)[eclipse]   # alternating depths
    _, eb = core.bls_search(t, f, 0.5, 12)
    assert eb["odd_even_sigma"] > 5


def test_bls_rejects_bad_limits(detrended):
    with pytest.raises(LCError):
        core.bls_search(detrended.time, detrended.flux, 5, 2)
    with pytest.raises(LCError):
        core.bls_search(detrended.time, detrended.flux, 0.5, 100)


def test_phase_fold_puts_transit_at_zero(detrended):
    phase, flux, epoch, binned = core.phase_fold(detrended.time, detrended.flux, 3.1, 2000.30, 0.01)
    assert np.all((phase >= -0.5) & (phase < 0.5))
    bx, by = binned
    assert abs(bx[np.argmin(by)]) < 0.02


def test_flares_detected(detrended):
    events = core.detect_flares(detrended.time, detrended.flux, 5.0, 2, baseline=1.0)
    assert len(events) == 3
    assert all(e["amplitude"] > 0.005 for e in events)
    assert all(e["equivalent_duration_days"] > 0 for e in events)


def test_summary_stats_ppm_consistent_across_normalization(raw):
    a = core.summary_stats(core.process_raw(raw, LCConfig(detrend="Median filter")))
    b = core.summary_stats(core.process_raw(raw, LCConfig(detrend="Median filter", normalize="Percent deviation")))
    assert a["rms_ppm"] == pytest.approx(b["rms_ppm"], rel=0.01)
    assert a["cadence_minutes"] == pytest.approx(10, rel=0.01)


# ---- export / import round trip ----------------------------------------------
def test_export_and_reopen(tmp_path, detrended):
    csv, fit, meta = tmp_path / "lc.csv", tmp_path / "lc.fits", tmp_path / "meta.json"
    core.export_csv(detrended, csv)
    core.export_fits(detrended, fit)
    core.export_metadata(detrended, meta)
    assert json.loads(meta.read_text())["tic"] == 261136679
    for path in (csv, fit):
        again = core.load_light_curve_file(path)
        assert len(again.time) == len(detrended.time)
        assert np.allclose(again.time, detrended.time)
    assert core.load_light_curve_file(fit).tic == 261136679


def test_open_spoc_style_fits(tmp_path):
    from astropy.io import fits
    t = np.linspace(1500, 1520, 500)
    cols = [fits.Column(name="TIME", format="D", array=t),
            fits.Column(name="SAP_FLUX", format="E", array=np.full(500, 100.0)),
            fits.Column(name="PDCSAP_FLUX", format="E", array=np.full(500, 101.0)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(500, int))]
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    prim = fits.PrimaryHDU()
    prim.header["TICID"], prim.header["SECTOR"] = 25155310, 7
    path = tmp_path / "spoc.fits"
    fits.HDUList([prim, hdu]).writeto(path)
    lc = core.load_light_curve_file(path)
    assert (lc.tic, lc.sector) == (25155310, 7)
    assert np.allclose(lc.corr_flux, 101.0)
