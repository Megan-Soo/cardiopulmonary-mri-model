#!/usr/bin/env python3
"""
Retrospective cardiac + respiratory gating for radial UTE using self-navigating
echoes (the k-space-centre sample of every spoke), reading MRI_Raw.h5.

Pipeline
  1. Load timestamps, trajectory, DCF and k-space (all sorted by acquisition time)
  2. Navigator = k-space-centre sample of every spoke, for every coil
  3. PCA across coils -> pick a respiratory component and a cardiac component
  4. Respiration: 0.8 Hz lowpass.  Cardiac: 0.8-1.5 Hz bandpass
     (navigator is resampled to a uniform time grid using the real timestamps)
  5. Every spoke gets a cardiac phase (0..1 within its beat) and a respiratory state
  6. NUFFT (gridding, DCF from file) reconstruction of each cardiac phase / respiratory state
  7. Save .npy volumes and QC figures

Background
  Every radial spoke passes through the centre of k-space (|k| = 0), and the value
  there is the integral of the whole imaged object, i.e. it is proportional to the
  total signal seen by each coil at that moment. Breathing and the heartbeat change
  that total (lung volume / blood volume and flow), so the series of centre samples,
  one per spoke, is a free physiological "navigator" with a sampling interval equal
  to the spoke repetition time. No ECG or respiratory belt is needed.

Array conventions used throughout
  t      [spoke]                     acquisition time of each spoke (s), sorted ascending
  coord  [spoke, sample, 3]          k-space coordinates, last axis ordered (Z, Y, X)
  dcf    [spoke, sample]             density-compensation weights from the scanner
  ksp    [echo, coil, spoke, sample] complex k-space (ksp_e drops the echo axis)

Outputs (written to --out)
  qc_navigator.png      navigator traces, PC spectra and cardiac-phase histogram
  gating_signals.npz    t, resp, phase, cbin, rbin, tu, card_u, peaks, fs
  recon_cardiac.npy     [n_card, Z, Y, X] cardiac-phase-resolved magnitude images
  recon_resp.npy        [n_resp, Z, Y, X] respiratory-state-resolved magnitude images
  cardiac_phases.png, resp_states.png   central-slice montages
  recon_joint.npy       [n_card, n_resp, Z, Y, X] (only with --joint)

Usage
  python retro_gating_recon.py --file MRI_Raw.h5
Requires: numpy scipy h5py scikit-learn matplotlib sigpy
"""
import argparse
import logging
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch
from sklearn.decomposition import PCA

CARDIAC_BAND = (0.8, 1.5)   # Hz, bandpass
RESP_LOWPASS = 0.8          # Hz, lowpass


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def whiten_ksp(ksp, noise, num_coils):
    """Noise pre-whitening of multi-coil k-space.

    Receiver coils pick up correlated noise, so raw channels are not statistically
    independent. This decorrelates them (and gives each unit noise variance) so that
    later steps, in particular the PCA across coils, are not dominated by noisy or
    strongly coupled channels.

    Processing steps
      1. Find the axis of ``noise`` whose length equals ``num_coils`` and move it
         to the front, flattening everything else into one "noise sample" axis.
      2. Estimate the coil noise covariance  cov = N N^H / n_samples.
      3. Cholesky-factorise cov = L L^H and apply L^-1 to the coil axis of ``ksp``.

    Parameters
    ----------
    ksp : complex ndarray [echo, coil, spoke, sample]
    noise : complex ndarray
        Noise-only acquisition; must have exactly one axis of length ``num_coils``
        (if several axes have that length, the first is taken as the coil axis).
    num_coils : int

    Returns
    -------
    complex64 ndarray, same shape as ``ksp``, with decorrelated coil channels.
    """
    ax = [i for i, s in enumerate(noise.shape) if s == num_coils]
    noise = np.moveaxis(noise, ax[0], 0).reshape(num_coils, -1)
    cov = noise @ noise.conj().T / noise.shape[1]
    linv = np.linalg.inv(np.linalg.cholesky(cov))
    return np.einsum("ij,ejsk->eisk", linv, ksp).astype(np.complex64)


def load_raw(path):
    """Read the raw HDF5 file and return everything sorted into acquisition order.

    Expected file layout
      Gating/time (or Gating/TIME_E0)   per-spoke timestamps in seconds
      Kdata/K{Z,Y,X}_E0                 trajectory coordinates
      Kdata/KW_E0                       density-compensation weights
      Kdata/KData_E{e}_C{c}/{real,imag} k-space for echo e, coil c
      Kdata/Noise/{real,imag}           optional noise-only data for whitening

    Processing steps
      1. Read the timestamps and compute ``order = argsort(time)``. The data are
         not stored in acquisition order, so *every* array is re-indexed with this
         same ``order`` to keep spokes, coordinates and weights aligned.
      2. Stack the Z/Y/X coordinates into ``coord`` and read the DCF.
      3. Count coils and echoes by probing for dataset names.
      4. Load real + imaginary parts for every echo and coil into one complex64
         array ``[echo, coil, spoke, sample]``.
      5. If noise data exist, whiten the coils (``whiten_ksp``); otherwise just
         normalise so the largest magnitude is 1.

    Parameters
    ----------
    path : str
        Path to MRI_Raw.h5.

    Returns
    -------
    time : float64 [spoke]                      sorted timestamps (s)
    coord : float32 [spoke, sample, 3]          (Z, Y, X)
    dcf : float32 [spoke, sample]
    ksp : complex64 [echo, coil, spoke, sample]
    """
    with h5py.File(path, "r") as hf:
        try:
            time = np.squeeze(hf["Gating"]["time"])
        except (KeyError, ValueError):
            time = np.squeeze(hf["Gating"]["TIME_E0"])      # timestamps in seconds
        order = np.argsort(time)
        time = time[order].astype(np.float64)

        coord = []
        for i in ["Z", "Y", "X"]:
            logging.info(f"Loading {i} coord.")
            coord.append(hf["Kdata"][f"K{i}_E0"][0][order])
        coord = np.stack(coord, axis=-1).astype(np.float32)

        logging.info("Loading dcf")
        dcf = hf["Kdata"]["KW_E0"][0][order].astype(np.float32)

        num_coils = 0
        while f"KData_E0_C{num_coils}" in hf["Kdata"]:
            num_coils += 1
        num_echo = 0
        while f"KData_E{num_echo}_C0" in hf["Kdata"]:
            num_echo += 1
        logging.info(f"Number of coils: {num_coils}, echoes: {num_echo}")

        ksp = []
        for e in range(num_echo):
            kspc = []
            for c in range(num_coils):
                logging.info(f"Loading kspace, echo {e + 1}/{num_echo}, coil {c + 1}/{num_coils}")
                k = hf["Kdata"][f"KData_E{e}_C{c}"]
                kspc.append((k["real"][0][order] + 1j * k["imag"][0][order]).astype(np.complex64))
            ksp.append(np.stack(kspc, axis=0))
        ksp = np.stack(ksp, axis=0)                          # [echo, coil, spoke, sample]

        try:
            noise = hf["Kdata"]["Noise"]["real"][()] + 1j * hf["Kdata"]["Noise"]["imag"][()]
            logging.info("Whitening ksp.")
            ksp = whiten_ksp(ksp, noise, num_coils)
        except Exception:
            ksp /= np.abs(ksp).max()
            logging.info("No noise data.")
    return time, coord, dcf, ksp


def prepare_coord(coord, matrix):
    """Scale the trajectory to the units sigpy expects and fix the image matrix size.

    sigpy's NUFFT wants coordinates in "pixel-frequency" units, i.e. each axis
    running over roughly [-N/2, N/2] for an N-pixel image. Scanner trajectories are
    sometimes stored that way and sometimes normalised (|k| <= 1 or 0.5).

    Processing steps
      1. Find the maximum |k| along each axis.
      2. If the largest is <= 1 the coordinates are treated as normalised: a
         ``matrix`` must be given, and all axes are scaled by one common factor so
         that the largest |k| becomes matrix/2 (isotropic scaling preserves the
         trajectory geometry).
      3. The output image shape is ``matrix`` if given, otherwise
         ``2 * ceil(|k|max)`` per axis.

    Parameters
    ----------
    coord : float [spoke, sample, 3]   (Z, Y, X)
    matrix : list[int] or None
        One value (isotropic) or three values (Z, Y, X). Two values are not valid.

    Returns
    -------
    coord : float32 [spoke, sample, 3] in pixel-frequency units
    oshape : tuple[int, int, int]      reconstruction matrix (Z, Y, X)
    """
    kmax = np.abs(coord).reshape(-1, 3).max(axis=0)
    if kmax.max() <= 1.0:            # normalised units (cycles/pixel or -0.5..0.5): needs matrix
        if matrix is None:
            raise SystemExit("Coordinates look normalised (|k|<=1); please supply --matrix.")
        n = np.array(matrix * (3 // len(matrix)), int)
        coord = coord / (2 * kmax.max()) * n            # -> |k| <= n/2 (isotropic scaling)
        kmax = np.abs(coord).reshape(-1, 3).max(axis=0)
    n = np.array(matrix * (3 // len(matrix)), int) if matrix else (2 * np.ceil(kmax)).astype(int)
    logging.info(f"Reconstruction matrix (Z,Y,X): {tuple(int(x) for x in n)}")
    return coord.astype(np.float32), tuple(int(x) for x in n)


# ----------------------------------------------------------------------------
# Navigator processing
# ----------------------------------------------------------------------------
def sos_filter(x, fs, kind, cutoff, order=4):
    """Zero-phase Butterworth filter (lowpass or bandpass).

    Uses second-order sections for numerical stability and ``sosfiltfilt`` (forward
    then backward pass) so the filtered signal has no time delay.

    Parameters
    ----------
    x : 1-D array
    fs : float
        Sampling rate (Hz).
    kind : {"lowpass", "bandpass", ...}
        Any ``btype`` accepted by ``scipy.signal.butter``.
    cutoff : float or [low, high]
        Corner frequency/frequencies in Hz.
    order : int
        Butterworth order of each pass.

    Returns
    -------
    1-D array, filtered signal, same length as ``x``.
    """
    sos = butter(order, cutoff, btype=kind, fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def band_fraction(x, fs, lo, hi):
    """Fraction of a signal's spectral power lying in the band [lo, hi] Hz.

    Estimates the power spectral density with Welch's method (segments of up to
    30 s) and returns  sum(PSD in band) / sum(PSD over all frequencies).
    Used to score how "respiratory-like" or "cardiac-like" each principal component is.

    Parameters
    ----------
    x : 1-D array
    fs : float
        Sampling rate (Hz).
    lo, hi : float
        Band edges (Hz).

    Returns
    -------
    float in [0, 1].
    """
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 30)))
    band = (f >= lo) & (f <= hi)
    return p[band].sum() / p.sum()


def breath_durations(resp_s, t, min_period=1.2):
    """Time between successive respiratory peaks (s).
        min_period=1.2s caps the detectable rate at ~50 bpm
    """
    fs = 1.0 / np.median(np.diff(t))
    pk, _ = find_peaks(resp_s, distance=int(min_period * fs), prominence=0.3 * resp_s.std())
    return np.diff(t[pk]), t[pk]


def estimate_resp_rate(resp, fs, lo=0.1, hi=RESP_LOWPASS):
    """Respiratory rate (breaths/min) from the dominant spectral peak of the respiratory signal.

    Welch PSD with segments of up to 60 s, zero-padded 8x so the peak is located
    more finely than the raw 1/60 Hz resolution. The peak is searched in [lo, hi] Hz.
    """
    x = resp - np.mean(resp)
    nper = min(len(x), int(fs * 60))
    f, p = welch(x, fs=fs, nperseg=nper, nfft=8 * nper)
    band = (f >= lo) & (f <= hi)
    return 60.0 * f[band][np.argmax(p[band])]


def extract_navigator(ksp_e, coord, nav_avg):
    """Build the self-navigator: one k-space-centre value per spoke and coil.

    Processing steps
      1. Locate the centre sample: the readout index whose median |k| sample is closest to zero.
      2. Average ``nav_avg`` consecutive samples starting there (optional, to
         reduce noise).
      3. For each coil, keep the magnitude, real part and imaginary part as three
         separate features. Respiration/cardiac motion can appear in amplitude or
         in phase, and keeping all three lets the later PCA pick the best mix.

    Parameters
    ----------
    ksp_e : complex [coil, spoke, sample]   k-space of one echo
    coord : float [spoke, sample, 3]
    nav_avg : int
        Number of centre samples to average.

    Returns
    -------
    feat : float [3 * n_coils, spoke]
        Rows are |dc|, Re(dc), Im(dc) for each coil, one column per spoke.
        Still on the (possibly irregular) spoke time grid and not normalised.
    """
    kr = np.median(np.linalg.norm(coord, axis=-1), axis=0)          # [sample]
    i0 = int(np.argmin(kr))
    i0 = min(i0, ksp_e.shape[-1] - nav_avg)
    logging.info(f"Navigator sample index: {i0} (|k| = {kr[i0]:.3g}), averaging {nav_avg}")
    dc = ksp_e[:, :, i0:i0 + nav_avg].mean(-1)                      # [coil, spoke]
    feat = np.concatenate([np.abs(dc), dc.real, dc.imag], axis=0)   # [3*coil, spoke]
    return feat


def to_uniform_grid(feat, t):
    """Resample the navigator onto a uniform time grid and normalise each feature.

    Processing steps
      1. Use the median spoke-to-spoke interval ``dt`` as the grid step.
      2. Linearly interpolate every feature onto that grid.
      3. Remove a linear trend from each feature (slow signal drift/saturation
         would otherwise dominate the PCA).
      4. Divide each feature by its standard deviation, so every channel
         contributes equally; features with ~zero variance are dropped.

    Parameters
    ----------
    feat : float [n_features, spoke]
    t : float [spoke]
        Spoke times (s), ascending.

    Returns
    -------
    feat_u : float [n_kept_features, n_time]   detrended, unit-variance
    tu : float [n_time]                        uniform time grid (s)
    fs : float                                 sampling rate = 1/dt (Hz)
    """
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise SystemExit("Median timestamp spacing is zero; check the timestamps.")
    tu = np.arange(t[0], t[-1], dt)
    feat_u = np.stack([np.interp(tu, t, f) for f in feat])
    feat_u = detrend(feat_u, axis=1, type="linear")
    std = feat_u.std(axis=1, keepdims=True)
    keep = std[:, 0] > 1e-12
    return feat_u[keep] / std[keep], tu, 1.0 / dt


def get_signals(feat_u, fs, n_pcs, flip_resp):
    """Separate the respiratory and cardiac signals from the multi-coil navigator.

    Processing steps
      1. PCA across the navigator features (time x features) gives up to ``n_pcs``
         principal-component time courses ordered by explained variance. This
         combines all coils/features without choosing any coil by hand.
      2. Respiratory component: the PC with the highest fraction of its power in
         0.1 - RESP_LOWPASS Hz (``band_fraction``).
      3. Cardiac component: among the remaining PCs, the one with the highest
         power fraction in CARDIAC_BAND.
      4. Filter: respiratory PC -> zero-phase lowpass at RESP_LOWPASS;
         cardiac PC -> zero-phase bandpass over CARDIAC_BAND.
      5. Optionally invert the respiratory sign. The PCA sign is arbitrary, so
         "high" is not guaranteed to mean inspiration; use ``flip_resp`` after
         checking the images.

    Parameters
    ----------
    feat_u : float [n_features, n_time]   output of ``to_uniform_grid``
    fs : float                            sampling rate (Hz)
    n_pcs : int                           number of principal components to keep
    flip_resp : bool                      multiply the respiratory signal by -1

    Returns
    -------
    resp : float [n_time]     filtered respiratory signal
    card : float [n_time]     filtered cardiac signal
    pcs : float [n_time, n_pcs] unfiltered PC time courses (for QC plots)
    """
    pcs = PCA(n_components=min(n_pcs, feat_u.shape[0])).fit_transform(feat_u.T)   # [time, pc]
    resp_scores = [band_fraction(pcs[:, k], fs, 0.1, RESP_LOWPASS) for k in range(pcs.shape[1])]
    k_resp = int(np.argmax(resp_scores))
    card_scores = [band_fraction(pcs[:, k], fs, *CARDIAC_BAND) if k != k_resp else -1
                   for k in range(pcs.shape[1])]
    k_card = int(np.argmax(card_scores))
    logging.info(f"PC {k_resp} -> respiration (score {resp_scores[k_resp]:.2f}); "
                 f"PC {k_card} -> cardiac (score {card_scores[k_card]:.2f})")

    resp = sos_filter(pcs[:, k_resp], fs, "lowpass", RESP_LOWPASS)
    card = sos_filter(pcs[:, k_card], fs, "bandpass", list(CARDIAC_BAND))
    if flip_resp:
        resp = -resp
    return resp, card, pcs


def cardiac_phase(card_u, tu, fs, t):
    """Assign every spoke a cardiac phase between 0 and 1 (retrospective gating).

    Processing steps
      1. Detect heartbeats as peaks of the filtered cardiac signal, with a minimum
         spacing of 0.6 s (max ~100 bpm) and a prominence of 0.3 standard
         deviations to ignore small ripples. At least 3 peaks are required.
      2. Compute each beat's duration (peak-to-peak interval) and the median.
      3. Reject irregular beats: any interval differing from the median by more
         than 25% (e.g. ectopic beats or missed/extra detections) is discarded.
      4. For each spoke acquired at time ``t``, find the beat that contains it and
         set  phase = (t - t_peak_start) / beat_duration.  Phase is therefore
         relative to the beat length, which handles heart-rate variability.
      5. Spokes before the first peak, after the last, or inside a rejected beat
         get NaN and are excluded from the cardiac reconstruction.

    Note that phase 0 is the navigator-signal peak, not the ECG R-wave, so the
    cardiac cycle is correct but shifted by a constant offset.

    Parameters
    ----------
    card_u : float [n_time]   filtered cardiac signal on the uniform grid
    tu : float [n_time]       uniform time grid (s)
    fs : float                sampling rate of the grid (Hz)
    t : float [spoke]         actual acquisition time of every spoke (s)

    Returns
    -------
    phase : float [spoke]     in [0, 1), NaN where unassigned
    pk : int [n_peaks]        indices of the detected peaks in ``card_u``/``tu``
    """
    pk, _ = find_peaks(card_u, distance=int(0.6 * fs), prominence=0.3 * card_u.std())
    if len(pk) < 3:
        raise RuntimeError("Fewer than 3 cardiac peaks found; check the navigator / bands.")
    tpk = tu[pk]
    rr = np.diff(tpk)
    med = np.median(rr)
    rr_ok = np.abs(rr - med) <= 0.25 * med                 # reject irregular beats
    j = np.searchsorted(tpk, t, side="right") - 1
    valid = (j >= 0) & (j < len(tpk) - 1)
    valid[valid] &= rr_ok[j[valid]]
    phase = np.full(len(t), np.nan)
    phase[valid] = (t[valid] - tpk[j[valid]]) / rr[j[valid]]
    logging.info(f"{len(pk)} peaks, median HR {60 / med:.1f} bpm, "
                 f"{(~rr_ok).sum()} irregular beats rejected, {np.isnan(phase).sum()} spokes unassigned")
    return phase, pk, 60/med


def resp_bins(resp, n_resp):
    """Sort spokes into respiratory states by signal amplitude.

    Bin edges are quantiles of the respiratory signal, so every bin receives
    the same number of spokes. That gives each respiratory image a similar
    amount of data (and similar undersampling streaks), at the cost of bins having
    different amplitude widths.

    Parameters
    ----------
    resp : float [spoke]   respiratory signal at each spoke time
    n_resp : int           number of respiratory states

    Returns
    -------
    rbin : int [spoke]      bin index 0 .. n_resp-1 (0 = lowest signal values)
    edges : float [n_resp+1] the quantile bin edges
    """
    edges = np.quantile(resp, np.linspace(0, 1, n_resp + 1))
    return np.clip(np.digitize(resp, edges[1:-1]), 0, n_resp - 1), edges


# ----------------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------------
def recon(ksp_e, coord, dcf, spoke_idx, oshape, dev):
    """Reconstruct one 3-D image from a subset of spokes (gridding / NUFFT adjoint).

    Processing steps
      1. Select the requested spokes and flatten their coordinates and DCF weights
         to lists of points.
      2. For each coil: multiply the samples by the density-compensation weights
         (radial sampling is dense near the centre; the weights equalise this),
         then apply the adjoint non-uniform FFT to get a complex coil image.
      3. Combine coils by root-sum-of-squares of the magnitudes.

    This is plain adjoint gridding with no iterative/regularised solver, so images
    from few spokes will show undersampling streaks. An empty ``spoke_idx``
    returns a zero volume.

    Parameters
    ----------
    ksp_e : complex [coil, spoke, sample]
    coord : float [spoke, sample, 3] in pixel-frequency units
    dcf : float [spoke, sample]
    spoke_idx : int array
        Indices of the spokes to use.
    oshape : tuple[int, int, int]
        Output matrix (Z, Y, X).
    dev : sigpy.Device
        CPU or GPU device on which the NUFFT is run.

    Returns
    -------
    float32 [Z, Y, X] magnitude image.
    """
    acc = np.zeros(oshape, np.float32)
    if len(spoke_idx) == 0:
        return acc
    crd = coord[spoke_idx].reshape(-1, 3)
    w = dcf[spoke_idx].reshape(-1)
    with dev:
        crd_d = sp.to_device(crd, dev)
        for c in range(ksp_e.shape[0]):
            d = sp.to_device(ksp_e[c, spoke_idx].reshape(-1) * w, dev)
            img = sp.nufft_adjoint(d, crd_d, oshape=oshape)
            acc += np.abs(sp.to_device(img, sp.cpu_device)) ** 2
    return np.sqrt(acc)


def middle_slices(vol, axis):
    """Take the central slice of every volume in a stack along one spatial axis.

    Parameters
    ----------
    vol : [phase, Z, Y, X]
    axis : int
        Spatial axis to slice through: 0 = Z, 1 = Y, 2 = X.

    Returns
    -------
    [phase, ., .] array of 2-D slices.
    """
    return np.take(vol, vol.shape[axis + 1] // 2, axis=axis + 1)


def montage(vol, axis, title, path):
    """Save a one-row montage of the central slice of each phase/state.

    All panels share the same grey-scale window (0 to the 99.5th percentile of the
    displayed slices), so brightness changes between phases are real and not an
    artefact of per-image scaling. Panel titles are the phase/state indices.

    Parameters
    ----------
    vol : [phase, Z, Y, X]
    axis : int      spatial axis to slice through (0 = Z, 1 = Y, 2 = X)
    title : str     figure title
    path : str      output image file
    """
    sl = middle_slices(vol, axis)
    n = sl.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(2.2 * n, 2.5), squeeze=False)
    vmax = np.percentile(sl, 99.5)
    for i in range(n):
        # ax[0, i].imshow(sl[i].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        ax[0, i].imshow(sl[i], cmap="gray", vmin=0, vmax=vmax)
        ax[0, i].set_title(f"{i}", fontsize=8)
        ax[0, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def qc_plot(t, resp_s, rbin, tu, card_u, pk, phase, pcs, fs, path, window=(0.0, 20.0), resp_rate=None, card_rate=None):
    """Save a four-panel quality-control figure for the gating signals.

    Panels (top to bottom)
      1. Respiratory signal vs time, points coloured by respiratory bin.
      2. Filtered cardiac signal vs time with the detected beats marked. Panels 1
         and 2 share one x-axis, so features line up vertically.
      3. Welch power spectra of the first four principal components, with the
         respiratory (blue) and cardiac (red) bands shaded. A good run shows one
         PC peaking in each band.
      4. Histogram of cardiac phase over all assigned spokes. It should be roughly
         flat; large gaps mean some cardiac phases have little data.

    Parameters
    ----------
    t : float [spoke]          spoke times (s)
    resp_s : float [spoke]     respiratory signal at the spoke times
    rbin : int [spoke]         respiratory bin of each spoke
    tu : float [n_time]        uniform navigator time grid (s)
    card_u : float [n_time]    filtered cardiac signal on that grid
    pk : int [n_peaks]         beat peak indices into ``card_u``/``tu``
    phase : float [spoke]      cardiac phase (NaN = unassigned)
    pcs : float [n_time, n_pcs] unfiltered PC time courses
    fs : float                 navigator sampling rate (Hz)
    path : str                 output image file
    window : (float, float) or None
        Time range (s from the first spoke) shown in panels 1 and 2; y-limits are
        fitted to this window. None shows the whole scan. The default 20 s keeps
        individual heartbeats readable.
    resp_rate : float or None  estimated breaths/min, shown on panel 1
    card_rate : float or None  estimated beats/min, shown on panel 2
    """
    t0 = t[0]
    tr_, tc_ = t - t0, tu - t0
    if window is None:
        window = (0.0, max(tr_[-1], tc_[-1]))
    window = (window[0], min(window[1], max(tr_[-1], tc_[-1])))

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(4, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)          # shared x-axis -> identical time scale
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # respiratory signal, points coloured by respiratory bin
    ax0.plot(tr_, resp_s, color="0.7", lw=0.6, zorder=1)
    for b in np.unique(rbin):
        m = rbin == b
        ax0.scatter(tr_[m], resp_s[m], s=3, color=plt.cm.viridis(b / max(rbin.max(), 1)),
                    label=f"bin {b}", zorder=2)
    ax0.set_xlim(*window)
    mr = (tr_ >= window[0]) & (tr_ <= window[1])
    if mr.any():
        pad = 0.05 * np.ptp(resp_s[mr])
        ax0.set_ylim(resp_s[mr].min() - pad, resp_s[mr].max() + pad)
    ax0.set_ylabel("Respiratory signal (a.u.)")
    title = f"Respiratory signal (lowpass {RESP_LOWPASS} Hz)"
    if resp_rate is not None:
        title += f" - estimated rate {resp_rate:.1f} breaths/min"
    ax0.set_title(title)
    ax0.legend(title="Resp. bin", loc="upper left", bbox_to_anchor=(1.01, 1.0),
               markerscale=3, fontsize=8, borderaxespad=0)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # cardiac signal with detected peaks
    ax1.plot(tc_, card_u, lw=0.8)
    ax1.plot(tc_[pk], card_u[pk], "rv", label="detected peaks")
    mc = (tc_ >= window[0]) & (tc_ <= window[1])
    if mc.any():
        pad = 0.05 * np.ptp(card_u[mc])
        ax1.set_ylim(card_u[mc].min() - pad, card_u[mc].max() + pad)
    ax1.set_xlabel("Time since first spoke (s)")
    ax1.set_ylabel("Cardiac signal (a.u.)")
    title = f"Cardiac signal (bandpass {CARDIAC_BAND[0]}-{CARDIAC_BAND[1]} Hz)"
    if card_rate is not None:
        title += f" - estimated rate {card_rate:.1f} beats/min"
    ax1.set_title(title)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)

    # PC spectra
    for k in range(min(4, pcs.shape[1])):
        f, p = welch(pcs[:, k], fs=fs, nperseg=min(len(pcs), int(fs * 30)))
        ax2.semilogy(f, p, label=f"PC{k}")
    ax2.axvspan(*CARDIAC_BAND, color="r", alpha=0.15, label="cardiac band")
    ax2.axvspan(0, RESP_LOWPASS, color="b", alpha=0.08, label="respiratory band")
    ax2.set_xlim(0, 4)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power spectral density (a.u.$^2$/Hz)")
    ax2.set_title("Principal-component spectra")
    ax2.legend(fontsize=8, ncol=3)

    # cardiac phase histogram
    ax3.hist(phase[~np.isnan(phase)], bins=50)
    ax3.set_xlabel("Cardiac phase (fraction of beat)")
    ax3.set_ylabel("Number of spokes")
    ax3.set_title("Cardiac phase distribution (should be roughly flat)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    """Command-line entry point: run the whole gating + reconstruction pipeline.

    Stages
      1. Load: read and time-sort the raw data (``load_raw``), optionally keep only
         the first ``--max-spokes``, select one echo, and convert the trajectory to
         sigpy units (``prepare_coord``).
      2. Navigator: extract the k-space-centre samples (``extract_navigator``),
         resample to a uniform grid (``to_uniform_grid``), and split into
         respiratory and cardiac signals with PCA + filtering (``get_signals``).
         A warning is logged if the sampling rate is below 8 Hz.
      3. Gating: assign each spoke a cardiac phase (``cardiac_phase``) and cardiac
         bin 0..n_card-1 (-1 = rejected), interpolate the respiratory signal back
         to the spoke times and bin it by amplitude (``resp_bins``). Save the QC
         figure and ``gating_signals.npz``.
      4. Cardiac-resolved reconstruction: one image per cardiac bin, optionally
         restricted to selected respiratory bins (``--card-resp-bins``, e.g. 0 for
         one respiratory state to avoid breathing blur).
      5. Respiratory-resolved reconstruction: one image per respiratory bin, using
         all cardiac phases.
      6. Optional (``--joint``): one image per (cardiac, respiratory) pair; each
         then has far fewer spokes, so expect more streaking.

    Run ``python retro_gating_recon.py --help`` for all options.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="MRI_Raw.h5")
    ap.add_argument("--echo", type=int, default=0, help="echo used for navigator and recon")
    ap.add_argument("--matrix", type=int, nargs="+", default=None,
                    help="recon matrix: one value (isotropic) or Z Y X. Default: from |k|max")
    ap.add_argument("--nav-avg", type=int, default=1, help="centre samples averaged for the navigator")
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--n-card", type=int, default=8, help="cardiac phases")
    ap.add_argument("--n-resp", type=int, default=4, help="respiratory states")
    ap.add_argument("--flip-resp", action="store_true", help="invert respiratory signal sign")
    ap.add_argument("--card-resp-bins", type=int, nargs="*", default=None,
                    help="restrict the cardiac recon to these respiratory bins (e.g. 0)")
    ap.add_argument("--joint", action="store_true", help="also reconstruct every cardiac x respiratory bin")
    ap.add_argument("--slice-axis", type=int, default=0, choices=[0, 1, 2], help="axis for montage slice (Z,Y,X)")
    ap.add_argument("--qc-window", type=float, nargs=2, default=[None, None], metavar=("START", "STOP"),
                    help="time window (s from first spoke) shown for the respiratory and cardiac QC traces. None=full scan")
    ap.add_argument("--max-spokes", type=int, default=None, help="use only the first N spokes (quick test)")
    ap.add_argument("--device", type=int, default=0, help="-1 = CPU, >=0 = GPU id")
    ap.add_argument("--out", default="gating_out")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out, exist_ok=True)

    # ---- load ---------------------------------------------------------------
    t, coord, dcf, ksp = load_raw(args.file)
    if args.max_spokes:
        t, coord, dcf, ksp = t[:args.max_spokes], coord[:args.max_spokes], dcf[:args.max_spokes], ksp[:, :, :args.max_spokes]
    ksp_e = ksp[args.echo]                                   # [coil, spoke, sample]
    del ksp
    coord, oshape = prepare_coord(coord, args.matrix)
    logging.info(f"{len(t)} spokes, duration {t[-1] - t[0]:.1f} s, ksp {ksp_e.shape}")

    # ---- navigator signals --------------------------------------------------
    feat = extract_navigator(ksp_e, coord, args.nav_avg)
    feat_u, tu, fs = to_uniform_grid(feat, t)
    logging.info(f"Navigator sampling rate: {fs:.1f} Hz")
    if fs < 8:
        logging.warning("Navigator sampling rate is low for cardiac signals.")
    resp_u, card_u, pcs = get_signals(feat_u, fs, args.n_pcs, args.flip_resp)

    phase, pk , card_rate = cardiac_phase(card_u, tu, fs, t)
    cbin = np.where(np.isnan(phase), -1, np.floor(np.nan_to_num(phase) * args.n_card)).astype(int)
    resp_s = np.interp(t, tu, resp_u)                        # back to spoke times
    dur, tpk = breath_durations(resp_s, t) # assess successive resp peaks
    logging.info(f"Breath duration: median {np.median(dur):.2f} s, mean {dur.mean():.2f} ± {dur.std():.2f} s "
             f"({60 / np.median(dur):.1f} breaths/min)")
    resp_rate = estimate_resp_rate(resp_s, fs) # dominant breathing freq across the whole scan
    logging.info(f"Estimated dominant respiratory rate: {resp_rate:.1f} breaths/min ({resp_rate / 60:.2f} Hz)")
    rbin, _ = resp_bins(resp_s, args.n_resp)
    qc_plot(t, resp_s, rbin, tu, card_u, pk, phase, pcs, fs, os.path.join(args.out, "qc_navigator.png"),
            window=tuple(args.qc_window), resp_rate=resp_rate, card_rate=card_rate)
    np.savez(os.path.join(args.out, "gating_signals.npz"), t=t, resp=resp_s, phase=phase,
             cbin=cbin, rbin=rbin, tu=tu, card_u=card_u, peaks=pk, fs=fs)

    dev = sp.Device(args.device)
    nspokes = len(t)

    # ---- cardiac-resolved ---------------------------------------------------
    allowed = np.ones(nspokes, bool)
    if args.card_resp_bins:
        allowed = np.isin(rbin, args.card_resp_bins)
    card_vols = []
    for c in range(args.n_card):
        idx = np.where((cbin == c) & allowed)[0]
        logging.info(f"cardiac phase {c}: {len(idx)} spokes")
        card_vols.append(recon(ksp_e, coord, dcf, idx, oshape, dev))
    card_vols = np.stack(card_vols)
    np.save(os.path.join(args.out, "recon_cardiac.npy"), card_vols)
    montage(card_vols, args.slice_axis, "Cardiac phases", os.path.join(args.out, "cardiac_phases.png"))

    # ---- respiratory-resolved -----------------------------------------------
    resp_vols = []
    for r in range(args.n_resp):
        idx = np.where(rbin == r)[0]
        logging.info(f"respiratory state {r}: {len(idx)} spokes")
        resp_vols.append(recon(ksp_e, coord, dcf, idx, oshape, dev))
    resp_vols = np.stack(resp_vols)
    np.save(os.path.join(args.out, "recon_resp.npy"), resp_vols)
    montage(resp_vols, args.slice_axis, "Respiratory states",
            os.path.join(args.out, "resp_states.png"))

    # ---- optional joint cardiac x respiratory -------------------------------
    if args.joint:
        joint = np.zeros((args.n_card, args.n_resp) + oshape, np.float32)
        for c in range(args.n_card):
            for r in range(args.n_resp):
                idx = np.where((cbin == c) & (rbin == r))[0]
                joint[c, r] = recon(ksp_e, coord, dcf, idx, oshape, dev)
        np.save(os.path.join(args.out, "recon_joint.npy"), joint)

    logging.info(f"Done. Outputs in {args.out}/")


if __name__ == "__main__":
    main()