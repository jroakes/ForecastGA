#! /usr/bin/env python
# coding: utf-8
#

__version__ = "0.1.0"


"""ForecastGA: Main"""



import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

pd.plotting.register_matplotlib_converters()
import seaborn as sns

from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split as tts

from datetime import timedelta
from dateutil.relativedelta import relativedelta

from forecastga.helpers.logging import get_logger

_LOG = get_logger(__name__)


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
    print(type(df))
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


def train_models(
    train,
    models,
    forecast_len,
    full_df=None,
    seasonality="infer_from_data",
    in_sample=None,
    freq=None,
    GPU=None,
):

    seasons = select_seasonality(train, seasonality)

    periods = select_seasonality(train, "periodocity")

    models_dict = {}
    for m in models:
        if in_sample:
            _LOG.info("Model {} is being trained for in sample prediction".format(m))
        else:
            _LOG.info(
                "Model {} is being trained for out of sample prediction".format(m)
            )

    return models_dict, seasons


def forecast_models(
    models_dict, forecast_len, freq, df, in_sample=True, GPU=False
):  # test here means any df

    forecast_dict = {}
    for name, model in models_dict.items():
        if in_sample:
            _LOG.info("Model {} is being used to forcast in sample".format(name))
        else:
            _LOG.info("Model {} is being used to forcast out of sample".format(name))

        if name == "PYAF":
            forecast_dict[name] = model["Target_Forecast"][-forecast_len:].values

    return forecast_dict


def forecast_frame(test, forecast_dict):
    insample = test.to_frame()
    for name, forecast in forecast_dict.items():
        insample[name] = forecast
    return insample


def forecast_frame_insample(forecast_dict, test):
    insample = test.to_frame()
    for name, forecast in forecast_dict.items():
        insample[name] = forecast
    return insample


def forecast_frame_outsample(forecast_dict, df, forecast_len, index):
    ra = -1
    for name, forecast in forecast_dict.items():
        ra += 1
        if ra == 0:
            outsample = pd.DataFrame(data=forecast, columns=[name], index=index)
            outsample[name] = forecast
        else:
            outsample[name] = forecast
    return outsample


def insample_performance(test, forecast_dict, dict=False):
    forecasts = forecast_frame(test, forecast_dict)
    dict_perf = {}
    for col, _ in forecasts.iteritems():
        dict_perf[col] = {}
        dict_perf[col]["rmse"] = rmse(forecasts["Target"], forecasts[col])
        dict_perf[col]["mse"] = dict_perf[col]["rmse"] ** 2
        dict_perf[col]["mean"] = forecasts[col].mean()
    if dict:
        return dict_perf
    else:
        return pd.DataFrame.from_dict(dict_perf)


@dataclass()
class AutomatedModel:
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """

    df: pd.Series
    # model_list: list = ["ARIMA","HOLT"]
    model_list: list
    season: str = "infer_from_data"
    forecast_len: int = 20
    GPU: bool = torch.cuda.is_available()

    def train_insample(self):
        dataframe, freq = parse_data(self.df)
        train, test = train_test_split(dataframe, train_proportion=0.75)
        forecast_len = len(test)
        models, seasonal = train_models(
            train,
            models=self.model_list,
            forecast_len=forecast_len,
            full_df=dataframe,
            seasonality=self.season,
            in_sample=True,
            freq=freq,
            GPU=self.GPU,
        )
        self.seasonality = seasonal

        return models, freq, test

    def train_outsample(self):
        dataframe, freq = parse_data(self.df)
        models, _ = train_models(
            dataframe["Target"],
            models=self.model_list,
            forecast_len=self.forecast_len,
            full_df=dataframe,
            seasonality=self.season,
            in_sample=False,
            freq=freq,
            GPU=self.GPU,
        )
        return models, freq, dataframe["Target"]

    def forecast_insample(self):
        models_dict, freq, test = self.train_insample()
        forecast_len = test.shape[0]
        forecast_dict = forecast_models(
            models_dict, forecast_len, freq, test, in_sample=True, GPU=self.GPU
        )
        forecast_frame = forecast_frame_insample(forecast_dict, test)
        self.models_dict_in = models_dict

        preformance = insample_performance(test, forecast_frame)
        _LOG.info("Successfully finished in sample forecast")

        return forecast_frame, preformance

    def forecast_outsample(self):
        models_dict, freq, dataframe = self.train_outsample()
        self.models_dict_out = models_dict
        self.freq = freq
        forecast_dict = forecast_models(
            models_dict,
            self.forecast_len,
            freq,
            dataframe,
            in_sample=False,
            GPU=self.GPU,
        )
        index = pd.date_range(
            dataframe.index[-1], periods=self.forecast_len + 1, freq=freq
        )[1:]
        forecast_frame = forecast_frame_outsample(
            forecast_dict, self.df, self.forecast_len, index
        )

        _LOG.info("Successfully finished out of sample forecast")
        return forecast_frame

    def ensemble(self, forecast_in, forecast_out):
        season = self.seasonality
        # if season==None:
        #   pass ValueError("Please first train a model using forecast_insample()")

        _LOG.info("Building LightGBM Ensemble from TS data (ensemble_lgb)")

        ensemble_lgb_in, ensemble_lgb_out = ensemble_lightgbm(
            forecast_in, forecast_out, self.freq
        )

        _LOG.info(
            "Building LightGBM Ensemble from PCA reduced TSFresh Features (ensemble_ts). \
                This can take a long time."
        )

        ensemble_ts_in, ensemble_ts_out = ensemble_tsfresh(
            forecast_in, forecast_out, season, self.freq
        )

        _LOG.info("Building Standard First Level Ensemble")
        df_ensemble_in, df_ensemble_out = ensemble_pure(forecast_in, forecast_out)
        middle_out = middle(ensemble_lgb_out, ensemble_ts_out, df_ensemble_out)
        middle_in = middle(ensemble_lgb_in, ensemble_ts_in, df_ensemble_in)

        _LOG.info("Building Final Multi-level Ensemble")
        middle_in, _ = ensemble_first(middle_in, forecast_in)
        all_ensemble_in, all_ensemble_out, all_performance = ensemble_doubled(
            middle_in, middle_out, forecast_in, forecast_out
        )

        return all_ensemble_in, all_ensemble_out, all_performance.T.sort_values("rmse")
