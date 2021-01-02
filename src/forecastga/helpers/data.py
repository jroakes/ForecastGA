#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Data Helpers"""

import pandas as pd
import numpy as np

from seasonal.periodogram import periodogram
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split as tts

from forecastga.helpers.ssa import mySSA
from forecastga.helpers.logging import get_logger

_LOG = get_logger(__name__)

from forecastga.models import MODELS

def print_model_info():
    _ = [print(v['name'], ":", v['description']) for k, v in MODELS.items() if v["status"] == "active"]

def constant_feature_detect(data, threshold=0.98):
    """detect features that show the same value for the
    majority/all of the observations (constant/quasi-constant features)

    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable as constant

    Returns
    -------
    list of variables names
    """

    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (
            (data_copy[feature].value_counts() / np.float(len(data_copy)))
            .sort_values(ascending=False)
            .values[0]
        )
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    _LOG.info(len(quasi_constant_feature), " variables are found to be almost constant")
    return quasi_constant_feature


# More Diverse Selection For TBAT
def infer_seasonality_ssa(train, index=1):
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


# Good First Selection
def infer_seasonality(train, index=0):  # skip the first one, normally
    interval, power = periodogram(train, min_period=4, max_period=None)
    try:
        season = int(
            pd.DataFrame([interval, power])
            .T.sort_values(1, ascending=False)
            .iloc[0, index]
        )
    except:
        _LOG.warning("Welch Season failed, defaulting to  SSA solution")
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


def parse_data(df):
    if type(df) == pd.DataFrame:
        if df.shape[1] > 1:
            raise ValueError("The dataframe should only contain one target column")
    elif type(df) == pd.Series:
        df = df.to_frame()
    else:
        raise TypeError(
            "Please supply a pandas dataframe with one column or a pandas series"
        )
    try:
        df.index.date
    except AttributeError:
        raise TypeError("The index should be a datetype")

    if df.isnull().any().values[0]:
        raise ValueError(
            "The dataframe cannot have any null values, please interpolate"
        )
    try:
        df.columns = ["Target"]
    except:
        raise ValueError("There should only be one column")

    df.index = df.index.rename("Date")
    df.index = add_freq(df.index)

    _LOG.info(
        "The data has been successfully parsed by infering a frequency, \
                and establishing a 'Date' index and 'Target' column."
    )

    return df, pd.infer_freq(df.index)


def train_test_split(df, train_proportion=0.75):

    size = int(df["Target"].shape[0] * train_proportion)
    train, test = tts(df["Target"], train_size=size, shuffle=False, stratify=None)
    _LOG.info(
        "An insample split of training size {} and testing \
                size {} has been constructed".format(
            len(train), len(test)
        )
    )
    return train, test


def season_list(train):
    lista = []
    for i in range(15):
        i = 1 + i
        lista.append(infer_seasonality_ssa(train, i))
    return lista


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
            _LOG.info(
                "The following set of plausible SSA seasonalities have been identified: {}".format(
                    seen
                )
            )
            return


# simple batcher.
def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr
