# Changelog

## 1.0.1 — code review and bug-fix release

### Fixed: data correctness
- **Wrong star shown after changing TIC.** The sector cache was keyed by sector number only, so loading
  Sector N for a new TIC silently re-used Sector N of the previous target. The cache now belongs to one TIC
  and *Load selected* refuses a sector list that was found for a different TIC.
- **Deselected sectors came back** after any processing change, because reprocessing used every cached
  sector instead of the sectors the user had loaded.
- **Detrending with "Percent deviation" normalization** divided the flux by a trend close to zero,
  producing values of ±60,000 %. Detrending now happens before normalization, then sigma clipping.
- **Detrending filters bridged data gaps**, creating spurious flare candidates and edge artefacts at the
  mid-sector downlink gap. Each continuous segment is now detrended separately.
- **RMS in ppm** was only correct for median-normalized data; it is now correct for every normalization.
- **Multi-sector metadata** (`time_min`, `time_max`) kept the values of the first sector only.
- **"Remove common artifacts"** quality mask now uses the documented TESS spacecraft-event flags
  (attitude tweak, safe mode, coarse/earth point, Argabrightening, momentum dump, manual exclude).

### Fixed: analyses
- **BLS period precision for multi-sector data.** The linear period grid found the right peak but not a
  precise enough period: for two sectors a year apart the folded transit drifted by up to 0.5 d (≈9×
  the transit duration). BLS now uses a coarse search plus a fine refinement around the best peak at the
  same speed, with transit-duration and frequency grids derived from the cadence and baseline.
- **Lomb–Scargle** now uses a uniform frequency grid that resolves every peak, the false-alarm probability
  is computed for the searched frequency range, and listed peaks are distinct local maxima.
- **Flares** are no longer split into separate events across data gaps, and amplitudes and equivalent
  durations are correct when the light curve is in percent units.
- BLS now also reports depth SNR, number of transits and an **odd/even depth test** (based on the
  scatter of individual transit depths, so residual stellar variability does not raise false alarms).

### Fixed: GUI
- **The whole GUI could freeze**: an exception in any result handler stopped the event-polling loop, after
  which no button responded. Handlers are now isolated and the loop always continues.
- **KeyError** when opening the Phase Fold tab after BLS with an empty period field.
- **Tabs jumped around**: every finished analysis selected its own tab, so a processing change flipped
  through all tabs and BLS jumped straight to Phase Fold. Only analyses the user starts now change tab.
- **BJD range was reset** on every processing change; it is now preserved.
- **Mouse-wheel scrolling** of the control panel did not work on macOS and only worked over empty space.
- Results, plots and the phase period of the **previous target remained visible** after loading a new one.
- Every focus change re-ran all analyses; analyses now re-run only when their inputs actually change, and
  LS and BLS parameters no longer trigger each other.
- *Load selected* and the new *Open* button are disabled during downloads; failed sectors are skipped
  with a warning instead of aborting the whole download.
- Export buttons now give a message when there is no data, report errors, and suggest file names.
- Embedded plots use `matplotlib.figure.Figure` instead of `pyplot`, avoiding backend conflicts on macOS.
- Window title and version numbers unified (the package reported 4.0.0, the window "v8", VERSION 1.0.0).

### Added
- **Open Local Light Curve…** — analyse the suite's own CSV/FITS exports, TESS SPOC/QLP light-curve FITS
  files, or any CSV with time and flux columns, without Eleanor or network access.
- **Full range** button, harmonic (P/2, 2P) markers on the BLS and rotation periodograms, and extended
  statistics (time span, cadence, robust and point-to-point scatter).
- Analysis JSON now records the analysis parameters, range and time system alongside the results.
- Eleanor/TensorFlow are imported only when TESS data are downloaded, with a clear message if missing.
- **Test-suite (51 tests)** including an end-to-end GUI test that clicks every button, using an offline
  Eleanor test double with injected planet, rotation and flare signals; GitHub Actions CI.

### Removed
- `maxsector.py` (unused).

## 1.0.0
- Initial release.
