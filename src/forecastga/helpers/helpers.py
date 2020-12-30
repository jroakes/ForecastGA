#! /usr/bin/env python
# coding: utf-8
#

"""Helpers"""

import pandas as pd

from seasonal.periodogram import periodogram
from scipy.signal import find_peaks
from atspy.etc.ssa import mySSA

from atspy.etc.logging import get_logger

_LOG = get_logger(__name__)


def season_list(train):
    lista = []
    for i in range(15):
        i = 1 + i
        lista.append(infer_seasonality_ssa(train, i))
    return lista


## More Diverse Selection For TBAT
def infer_seasonality_ssa(train, index=1):  ##skip the first one, normally
    ssa = mySSA(train)
    ssa.embed(embedding_dimension=36, verbose=False)
    ssa.decompose(True)
    rec = ssa.view_reconstruction(
        ssa.Xs[index], names="Seasonality", return_df=True, plot=False
    )
    peaks, _ = find_peaks(
        rec.values.reshape(
            len(rec),
        ),
        height=0,
    )
    peak_diffs = [j - i for i, j in zip(peaks[:-1], peaks[1:])]
    seasonality = max(peak_diffs, key=peak_diffs.count)
    return seasonality


## Good First Selection
def infer_seasonality(train, index=0):  ##skip the first one, normally
    interval, power = periodogram(train, min_period=4, max_period=None)
    try:
        season = int(
            pd.DataFrame([interval, power])
            .T.sort_values(1, ascending=False)
            .iloc[0, index]
        )
    except:
        _LOG.info("Welch Season failed, defaulting to  SSA solution")
        season = int(infer_seasonality_ssa(train, index=1))
    return season


def infer_periodocity(train):
    perd = pd.infer_freq(train.index)
    if perd in ["MS", "M", "BM", "BMS"]:
        periodocity = 12
    elif perd in ["BH", "H"]:
        periodocity = 24
    elif perd == "B":
        periodocity = 5
    elif perd == "D":
        periodocity = 7
    elif perd in ["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
        periodocity = 52
    elif perd in ["Q", "QS", "BQ", "BQS"]:
        periodocity = 4
    elif perd in ["A", "BA", "AS", "BAS"]:
        periodocity = 10
    elif perd in ["T", "min"]:
        periodocity = 60
    elif perd == "S":
        periodocity = 60
    elif perd in ["L", "ms"]:
        periodocity = 1000
    elif perd in ["U", "us"]:
        periodocity = 1000
    elif perd == "N":
        periodocity = 1000

    return periodocity


def select_seasonality(train, season):
    if season == "periodocity":
        seasonality = infer_periodocity(train)
    elif season == "infer_from_data":
        seasonality = infer_seasonality(train)
    return seasonality


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """
    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError(
            "no discernible frequency found to `idx`.  Specify"
            " a frequency string with `freq`."
        )
    return idx


def get_unique_N(iterable, N):
    """Yields (in order) the first N unique elements of iterable.
    Might yield less if data too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        yield e
        if len(seen) == N:
            return
