import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from astropy import units as u
from astropy.table import QTable
from astropy.time import Time

from sunpy import timeseries as ts

from sunkit_instruments.goes_xrs.goes_xrs import flux_to_flareclass

__all__ = ["find_goes_flares_asr", "find_goes_flares_plutino"]

_ASR_TABLE_COLUMNS = (
    "flare_id",
    "start_time",
    "peak_time",
    "end_time",
    "peak_flux",
    "background_flux",
    "delta_flux",
    "flux_ratio",
    "goes_class",
    "flare_class",
    "flux_integral",
    "flux_integral_corrected",
)

_PLUTINO_TABLE_COLUMNS = (
    "event_id",
    "multiple_id",
    "start_time",
    "peak_time",
    "end_time",
    "peak_flux",
    "background_flux",
    "goes_class",
    "flux_integral",
)


@u.quantity_input
def find_goes_flares_asr(goes_ts, min_delta_flux: u.W / u.m**2 = 2e-8 * u.W / u.m**2):
    """
    Detect solar flares in a GOES XRS time series.

    This implements the extrema-based flare detection algorithm described
    in Berretti et al. (2025) [1]_ for the "ASR" flare catalogue, which itself
    builds on the algorithm of Plutino et al. (2023) [2]_. Candidate flares
    are identified from consecutive local minima and maxima of the XRS-B
    (1-8 Angstrom) flux: a local minimum is a candidate flare start, and the
    following local maximum is a candidate flare peak. A candidate is
    accepted as a flare if its peak flux exceeds the local background
    (estimated at the candidate start) by at least ``min_delta_flux``. The
    flare end is the time at which the flux decays back to within
    ``min_delta_flux`` of the background, or the start of the next flare,
    whichever comes first.

    Parameters
    ----------
    goes_ts : `~sunpy.timeseries.sources.XRSTimeSeries`
        The GOES/XRS time series to search for flares. Must contain a
        ``xrsb`` column. This algorithm was designed and validated using
        1-minute averaged, quality-controlled XRS-B fluxes; results with
        other cadences have not been validated against the published
        catalogues.
    min_delta_flux : `~astropy.units.Quantity`, optional
        The minimum flux above the local background for a candidate peak to
        be registered as a flare. Defaults to 2e-8 W/m^2, the noise level
        used in both Plutino et al. (2023) and Berretti et al. (2025).

    Returns
    -------
    `~astropy.table.QTable`
        One row per detected flare, with columns:

        * ``flare_id`` : a group identifier shared by overlapping or
          homologous flares (i.e. where one flare's decay had not reached
          the background before the next flare began).
        * ``start_time``, ``peak_time``, ``end_time``
        * ``peak_flux`` : the XRS-B flux at ``peak_time``.
        * ``background_flux`` : the estimated pre-flare background flux.
        * ``delta_flux`` : ``peak_flux`` minus ``background_flux``.
        * ``flux_ratio`` : ``peak_flux`` divided by ``background_flux``.
        * ``goes_class`` : the standard NOAA flare class of ``peak_flux``.
        * ``flare_class`` : the flare class of ``delta_flux``, i.e. the
          class after removing the contribution of the background flux.
        * ``flux_integral`` : the trapezoidal time integral of the flux
          between ``start_time`` and ``end_time``.
        * ``flux_integral_corrected`` : as ``flux_integral``, but with the
          background flux subtracted first.

    Examples
    --------
    >>> from sunpy import timeseries as ts
    >>> import sunpy.data.sample  # doctest: +REMOTE_DATA
    >>> from sunkit_instruments import goes_xrs
    >>> goes_ts = ts.TimeSeries(sunpy.data.sample.GOES_XRS_TIMESERIES) # doctest: +REMOTE_DATA
    >>> goes_flare = goes_ts.truncate("2011-06-07 06:07", "2011-06-07 06:09") # doctest: +REMOTE_DATA
    >>> flares = goes_xrs.find_goes_flares_asr(goes_flare) # doctest: +REMOTE_DATA
    >>> flares["start_time", "peak_time", "end_time", "goes_class"]  # doctest: +REMOTE_DATA
    <QTable length=1>
           start_time              peak_time        ... goes_class
              Time                    Time          ...    str4
    ----------------------- ----------------------- ... ----------
    2011-06-07 06:08:31.929 2011-06-07 06:08:48.312 ...       B3.3

    Notes
    -----
    Unlike the original ASR/PLU implementations, which operate on plain
    ``pandas`` dataframes, this function works directly with a
    `~sunpy.timeseries.sources.XRSTimeSeries` and returns physically-unit
    aware quantities. It also does not attempt to correct for instrumental
    saturation, which the ASR catalogue code handles as a separate
    pre-processing step.

    Because the underlying light curve is not smoothed before searching for
    extrema, small noise-driven wiggles on the elevated flux during a strong
    flare's rise, peak, or decay are frequently registered as their own
    low-``delta_flux`` (but high ``peak_flux``) candidate events. This is a
    known characteristic of the published algorithm rather than a bug (see
    the discussion and Figure 1 of Berretti et al. 2025): such events share
    a ``flare_id`` with the larger flare they interrupt, so grouping rows by
    ``flare_id`` and keeping the maximum ``peak_flux`` in each group
    recovers a single event per group, matching how the ASR and PLU
    catalogues themselves report grouped/homologous events.

    Older, FITS-sourced GOES 8-15 time series have the operational SWPC
    scaling factor of 0.7 applied to XRS-B, unlike the "true flux" science
    data used to build the published ASR/PLU catalogues (see
    `~sunkit_instruments.goes_xrs.calculate_temperature_em` for the same
    caveat). This function does not attempt to detect or undo that scaling;
    divide ``xrsb`` by 0.7 first if you need results on the same absolute
    scale as those catalogues. When this function is run on 1-minute
    averaged, correctly-scaled XRS-B flux, its start/peak/end times and
    (peak, delta) fluxes reproduce the corresponding rows of the public ASR
    catalogue almost exactly (see the tests in ``tests/test_flares.py`` for
    a worked comparison against the 2011 June 7 M-class flare).

    References
    ----------
    .. [1] Berretti, M., Mestici, S., Giovannelli, L., et al. 2025, ApJS, 278, 9,
        DOI: 10.3847/1538-4365/adc731
    .. [2] Plutino, N., Berrilli, F., Del Moro, D., & Giovannelli, L. 2023,
        Advances in Space Research, 71, 2048, DOI: 10.1016/j.asr.2022.11.020
    """
    if not isinstance(goes_ts, ts.XRSTimeSeries):
        raise TypeError(
            f"Input time series must be a XRSTimeSeries instance, not {type(goes_ts)}"
        )
    if "xrsb" not in goes_ts.columns:
        raise ValueError("The input time series does not contain a 'xrsb' column.")

    flux = goes_ts.quantity("xrsb").to_value(u.W / u.m**2)
    time = goes_ts.time

    valid = np.isfinite(flux)
    flux = flux[valid]
    time = time[valid]

    if flux.size == 0:
        return _empty_flare_table()

    extrema_idx, extrema_kind = _find_local_extrema(flux)
    if extrema_idx.size < 3:
        return _empty_flare_table()

    seconds = (time - time[0]).to_value(u.s)
    threshold = min_delta_flux.to_value(u.W / u.m**2)

    starts, peaks, ends = [], [], []
    for j in range(extrema_idx.size - 2):
        if extrema_kind[j] != "min" or extrema_kind[j + 1] != "max":
            continue

        h = extrema_idx[j]
        k = extrema_idx[j + 1]
        background_flux = flux[h] if h == 0 else 0.5 * (flux[h] + flux[h - 1])
        peak_flux = flux[k]

        if peak_flux <= background_flux + threshold:
            continue

        end = k
        while flux[end] > background_flux + threshold:
            end += 1
            if end == flux.size - 1:
                break

        next_min = extrema_idx[j + 2]
        if end >= next_min:
            end = next_min

        starts.append(h)
        peaks.append(k)
        ends.append(end)

    if not starts:
        return _empty_flare_table()

    rows = []
    flare_id = -1
    for i, (h, k, end) in enumerate(zip(starts, peaks, ends)):
        if i == 0 or h != ends[i - 1]:
            flare_id += 1

        background_flux = flux[h] if h == 0 else 0.5 * (flux[h] + flux[h - 1])
        peak_flux = flux[k]
        delta_flux = peak_flux - background_flux

        flux_integral = np.trapezoid(flux[h:end], seconds[h:end]) * u.W * u.s / u.m**2
        flux_integral_corrected = (
            np.trapezoid(flux[h:end] - background_flux, seconds[h:end]) * u.W * u.s / u.m**2
        )

        rows.append(
            (
                flare_id,
                time[h],
                time[k],
                time[end],
                peak_flux * u.W / u.m**2,
                background_flux * u.W / u.m**2,
                delta_flux * u.W / u.m**2,
                u.Quantity(peak_flux / background_flux),
                flux_to_flareclass(peak_flux * u.W / u.m**2, decimal_places=1),
                flux_to_flareclass(delta_flux * u.W / u.m**2, decimal_places=1),
                flux_integral.to(u.J / u.m**2),
                flux_integral_corrected.to(u.J / u.m**2),
            )
        )

    columns = list(zip(*rows))
    return QTable(
        [
            columns[0],
            Time(columns[1]),
            Time(columns[2]),
            Time(columns[3]),
            u.Quantity(columns[4]),
            u.Quantity(columns[5]),
            u.Quantity(columns[6]),
            u.Quantity(columns[7]),
            columns[8],
            columns[9],
            u.Quantity(columns[10]),
            u.Quantity(columns[11]),
        ],
        names=_ASR_TABLE_COLUMNS,
    )


def _find_local_extrema(flux):
    """
    Find alternating local minima and maxima in ``flux``, starting from a
    minimum, and drop spurious extrema caused by signal noise.
    """
    minima = argrelextrema(flux, np.less_equal, order=1)[0]
    maxima = argrelextrema(flux, np.greater_equal, order=1)[0]

    idx = np.concatenate([minima, maxima])
    kind = np.array(["min"] * minima.size + ["max"] * maxima.size)
    order = np.argsort(idx, kind="stable")
    idx, kind = idx[order], kind[order]

    if kind.size and kind[0] == "max":
        idx, kind = idx[1:], kind[1:]

    return _sanitize_local_extrema(idx, kind)


def _sanitize_local_extrema(idx, kind):
    """
    Drop runs of three consecutive extrema that are separated by only two
    data points, which are indicative of signal noise rather than a true
    flare rise/decay.
    """
    if idx.size < 3:
        return idx, kind

    drop = np.zeros(idx.size, dtype=bool)
    for i in range(idx.size - 2):
        if idx[i] == idx[i + 2] - 2:
            drop[i] = True
            drop[i + 1] = True

    keep = ~drop
    return idx[keep], kind[keep]


def _empty_flare_table():
    return QTable(
        [
            np.array([], dtype=int),
            Time([], format="isot", scale="utc"),
            Time([], format="isot", scale="utc"),
            Time([], format="isot", scale="utc"),
            u.Quantity([], u.W / u.m**2),
            u.Quantity([], u.W / u.m**2),
            u.Quantity([], u.W / u.m**2),
            u.Quantity([], u.dimensionless_unscaled),
            np.array([], dtype=str),
            np.array([], dtype=str),
            u.Quantity([], u.J / u.m**2),
            u.Quantity([], u.J / u.m**2),
        ],
        names=_ASR_TABLE_COLUMNS,
    )


@u.quantity_input
def find_goes_flares_plutino(
    goes_ts,
    min_delta_flux: u.W / u.m**2 = 2e-8 * u.W / u.m**2,
    resample_cadence: u.s = 12 * u.s,
    smooth_window=21,
    spike_threshold=10,
):
    """
    Detect solar flares in a GOES XRS time series using the algorithm of
    Plutino et al. (2023) [1]_ (the "PLU" flare catalogue).

    Unlike `find_goes_flares_asr`, which follows the later, unsmoothed ASR
    algorithm, this function replicates PLU's own pre-processing: the
    XRS-B flux is first resampled to a fixed cadence, de-spiked using a
    moving-window ``max/min`` ratio test, and then boxcar-smoothed, before
    searching for local minima that mark candidate flare starts. For each
    candidate start, a pre-flare background is estimated from the
    preceding 60 seconds of (smoothed) flux, and the flare's end is the
    first later time at which the flux decays back to within
    ``min_delta_flux`` of that background. The event is only kept if the
    largest flux value between its start and end exceeds the background by
    more than ``min_delta_flux``.

    Parameters
    ----------
    goes_ts : `~sunpy.timeseries.sources.XRSTimeSeries`
        The GOES/XRS time series to search for flares. Must contain a
        ``xrsb`` column.
    min_delta_flux : `~astropy.units.Quantity`, optional
        The minimum flux above the local background for a candidate peak to
        be registered as a flare. Defaults to 2e-8 W/m^2, the noise level
        used in Plutino et al. (2023).
    resample_cadence : `~astropy.units.Quantity`, optional
        The fixed time cadence the flux is resampled (via averaging) to
        before de-spiking and smoothing. Defaults to 12 seconds, as in
        Plutino et al. (2023).
    smooth_window : int, optional
        The width, in resampled data points, of the boxcar smoothing
        window. Defaults to 21 points (about 4 minutes at the default
        12 second cadence), as in Plutino et al. (2023).
    spike_threshold : float, optional
        Points where the ratio of the maximum to the minimum flux in a
        surrounding 60 second window exceeds this value are treated as
        instrumental spikes and replaced by interpolation before smoothing.
        Defaults to 10, as in Plutino et al. (2023).

    Returns
    -------
    `~astropy.table.QTable`
        One row per detected flare, with columns:

        * ``event_id`` : a unique, sequential identifier for each row.
        * ``multiple_id`` : a group identifier shared by overlapping or
          homologous flares, i.e. where a flare's reported end had to be
          truncated to the start of the next flare because its own decay
          back to the background had not yet completed. The full duration
          of such a group can be recovered from the ``start_time`` of its
          first row and the ``end_time`` of its last row.
        * ``start_time``, ``peak_time``, ``end_time``
        * ``peak_flux`` : the largest XRS-B flux between ``start_time`` and
          ``end_time``.
        * ``background_flux`` : the estimated pre-flare background flux.
        * ``goes_class`` : the standard NOAA flare class of ``peak_flux``
          (this is not corrected for the background flux).
        * ``flux_integral`` : the trapezoidal time integral of the flux
          between ``start_time`` and ``end_time``.

    Examples
    --------
    >>> from sunpy import timeseries as ts
    >>> import sunpy.data.sample  # doctest: +REMOTE_DATA
    >>> from sunkit_instruments import goes_xrs
    >>> goes_ts = ts.TimeSeries(sunpy.data.sample.GOES_XRS_TIMESERIES) # doctest: +REMOTE_DATA
    >>> goes_flare = goes_ts.truncate("2011-06-07 06:00", "2011-06-07 07:30") # doctest: +REMOTE_DATA
    >>> flares = goes_xrs.find_goes_flares_plutino(goes_flare) # doctest: +REMOTE_DATA
    >>> flares["start_time", "peak_time", "end_time", "goes_class"]  # doctest: +REMOTE_DATA
    <QTable length=2>
              start_time                    peak_time           ... goes_class
                 Time                          Time             ...    str4
    ----------------------------- ----------------------------- ... ----------
    2011-06-07T06:03:24.000000000 2011-06-07T06:32:24.000000000 ...       M2.4
    2011-06-07T06:35:00.000000000 2011-06-07T06:40:24.000000000 ...       M2.5

    Notes
    -----
    This function does not attempt to correct for instrumental saturation,
    which the PLU catalogue code handles as a separate pre-processing step,
    and does not implement the 24-hour batching PLU uses (there is no
    equivalent limit here on how far the search for an event's end may
    extend). Older, FITS-sourced GOES 8-15 time series have the operational
    SWPC scaling factor of 0.7 applied to XRS-B; see the equivalent caveat on
    `find_goes_flares` for details. This scaling should *not* be removed from
    GOES-6 or -7 data, which was never SWPC-rescaled in the first place (see
    the GOES 1-15 Science-Quality Data Readme). Note that Plutino et al.
    (2023), Section 2, describes applying the /0.7 correction to "GOES-6 to
    GOES-15" data uniformly when building the published PLU catalogue; this
    appears to over-apply the correction to their GOES-6/7 events, inflating
    those specific catalogue rows by a factor of about 1/0.7. When run on
    raw (correctly unscaled) GOES-6 flux, this function reproduces the
    corresponding rows of the public PLU catalogue for 1986 January almost
    exactly after undoing that likely-erroneous factor (see the tests in
    ``tests/test_flares.py`` for a worked comparison); one of the smallest
    published GOES-6 events is not reproduced at all, since its corrected
    (un-inflated) peak flux does not clear the default noise threshold.

    References
    ----------
    .. [1] Plutino, N., Berrilli, F., Del Moro, D., & Giovannelli, L. 2023,
        Advances in Space Research, 71, 2048, DOI: 10.1016/j.asr.2022.11.020
    """
    if not isinstance(goes_ts, ts.XRSTimeSeries):
        raise TypeError(
            f"Input time series must be a XRSTimeSeries instance, not {type(goes_ts)}"
        )
    if "xrsb" not in goes_ts.columns:
        raise ValueError("The input time series does not contain a 'xrsb' column.")

    time, flux = _resample_flux(goes_ts, resample_cadence)
    if flux.size < smooth_window:
        return _empty_flare_table_plutino()

    window_points = max(1, round((60 * u.s / resample_cadence).decompose().value))
    flux = _remove_spikes(flux, window_points, spike_threshold)
    flux = _boxcar_smooth(flux, smooth_window)

    valid = np.isfinite(flux)
    flux, time = flux[valid], time[valid]
    if flux.size < 5:
        return _empty_flare_table_plutino()

    minima_idx = argrelextrema(flux, np.less, order=2)[0]
    if minima_idx.size == 0:
        return _empty_flare_table_plutino()

    seconds = (time - time[0]).to_value(u.s)
    noise = min_delta_flux.to_value(u.W / u.m**2)

    starts, ends, backgrounds = [], [], []
    for pos, i in enumerate(minima_idx):
        bg_start = max(0, i - window_points)
        background = flux[i] if bg_start == i else np.mean(flux[bg_start:i])

        # The end of the event is the next later local minimum whose flux has
        # decayed back to within `noise` of the background; if the decay does
        # not complete before the end of the data, use the last data point.
        end = flux.size - 1
        for later in minima_idx[pos + 1 :]:
            if flux[later] <= background + noise:
                end = later
                break

        if np.max(flux[i : end + 1]) <= background + noise:
            continue

        starts.append(i)
        ends.append(end)
        backgrounds.append(background)

    if not starts:
        return _empty_flare_table_plutino()

    n = len(starts)
    reported_end = list(ends)
    multiple_id = [0] * n
    for k in range(n - 1):
        if starts[k + 1] < ends[k]:
            reported_end[k] = starts[k + 1]
            multiple_id[k + 1] = multiple_id[k]
        else:
            multiple_id[k + 1] = multiple_id[k] + 1

    # The peak is found within each row's own (possibly truncated) window, so
    # that peak_time can never fall after end_time.
    peaks = [h + np.argmax(flux[h : e + 1]) for h, e in zip(starts, reported_end)]

    rows = []
    for event_id, (h, p, end, background) in enumerate(zip(starts, peaks, reported_end, backgrounds)):
        peak_flux = flux[p]
        flux_integral = np.trapezoid(flux[h : end + 1], seconds[h : end + 1]) * u.W * u.s / u.m**2

        rows.append(
            (
                event_id,
                multiple_id[event_id],
                time[h],
                time[p],
                time[end],
                peak_flux * u.W / u.m**2,
                background * u.W / u.m**2,
                flux_to_flareclass(peak_flux * u.W / u.m**2, decimal_places=1),
                flux_integral.to(u.J / u.m**2),
            )
        )

    columns = list(zip(*rows))
    return QTable(
        [
            columns[0],
            columns[1],
            Time(columns[2]),
            Time(columns[3]),
            Time(columns[4]),
            u.Quantity(columns[5]),
            u.Quantity(columns[6]),
            columns[7],
            u.Quantity(columns[8]),
        ],
        names=_PLUTINO_TABLE_COLUMNS,
    )


def _resample_flux(goes_ts, cadence):
    """
    Resample the ``xrsb`` column of ``goes_ts`` to a fixed time cadence by
    averaging, returning the resampled times and flux values.
    """
    df = goes_ts.to_dataframe()[["xrsb"]]
    df = df.resample(pd.Timedelta(seconds=cadence.to_value(u.s))).mean()
    return Time(df.index.to_numpy()), df["xrsb"].to_numpy()


def _remove_spikes(flux, window_points, spike_threshold):
    """
    Replace points where the ratio of the local maximum to the local
    minimum, in a centered window, exceeds ``spike_threshold`` with a
    linear interpolation between their neighbours.
    """
    series = pd.Series(flux)
    rolling_max = series.rolling(window_points, center=True, min_periods=1).max()
    rolling_min = series.rolling(window_points, center=True, min_periods=1).min()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = rolling_max / rolling_min
    is_spike = (ratio > spike_threshold).to_numpy()

    cleaned = series.copy()
    cleaned[is_spike] = np.nan
    cleaned = cleaned.interpolate(limit_direction="both").bfill().ffill()
    return cleaned.to_numpy()


def _boxcar_smooth(flux, window):
    """
    Smooth ``flux`` with a centered boxcar (moving average) filter.
    """
    return pd.Series(flux).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _empty_flare_table_plutino():
    return QTable(
        [
            np.array([], dtype=int),
            np.array([], dtype=int),
            Time([], format="isot", scale="utc"),
            Time([], format="isot", scale="utc"),
            Time([], format="isot", scale="utc"),
            u.Quantity([], u.W / u.m**2),
            u.Quantity([], u.W / u.m**2),
            np.array([], dtype=str),
            u.Quantity([], u.J / u.m**2),
        ],
        names=_PLUTINO_TABLE_COLUMNS,
    )
