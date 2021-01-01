#! /usr/bin/env python
# coding: utf-8


"""ForecastGA: Main"""

from dataclasses import dataclass
import importlib
import warnings
import torch
import pandas as pd
from statsmodels.tools.eval_measures import rmse

from forecastga.helpers.logging import get_logger
from forecastga.helpers.data import parse_data, train_test_split
from forecastga.ensembles import (
    ensemble_lightgbm,
    ensemble_tsfresh,
    ensemble_pure,
    middle,
    ensemble_first,
    ensemble_doubled,
)
from forecastga.models import MODELS

pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")

_LOG = get_logger(__name__)


__version__ = "0.1.0"


def train_models(
    df, model_list, forecast_len, seasonality, train_proportion, GPU, in_sample, **kwargs
):

    models_dict = {}
    available_models = [k for k, v in MODELS.items() if v["status"] == "active"]

    for model_name in model_list:
        if model_name not in available_models:
            _LOG.warning("Model {} is not available.  Skipping...".format(model_name))
            continue

        _LOG.info(
            "Model {} is being loaded and \
                   trained for {} prediction".format(
                model_name, "in sample" if in_sample else "out of sample"
            )
        )
        model_data = MODELS[model_name]
        module = importlib.import_module(model_data["loc"])
        module_class = getattr(module, model_data["class"])
        model = module_class(
            df,
            seasonality=seasonality,
            forecast_len=forecast_len,
            train_proportion=train_proportion,
            GPU=GPU,
            in_sample=in_sample,
        )

        model.train(**kwargs)

        models_dict[model_name] = model

    return models_dict


def forecast_models(models_dict, in_sample):

    forecast_dict = {}
    for model_name, model in models_dict.items():

        _LOG.info(
            "Model {} is being used to forecast {}".format(
                model_name, "in sample" if in_sample else "out of sample"
            )
        )

        model.forecast()

        forecast_dict[model_name] = model.prediction

    return forecast_dict


def forecast_dataframe(df, forecast_dict):
    insample = df.to_frame()
    for name, forecast in forecast_dict.items():
        insample[name] = forecast
    return insample


def insample_performance(forecast_frame, as_dict=False):

    dict_perf = {}
    for col, _ in forecast_frame.iteritems():
        dict_perf[col] = {}
        dict_perf[col]["rmse"] = rmse(forecasts["Target"], forecasts[col])
        dict_perf[col]["mse"] = dict_perf[col]["rmse"] ** 2
        dict_perf[col]["mean"] = forecasts[col].mean()
    if as_dict:
        return dict_perf

    return pd.DataFrame.from_dict(dict_perf)


@dataclass()
class AutomatedModel:

    df: pd.Series
    model_list: list
    seasonality: str = "infer_from_data"
    train_proportion: float = 0.75
    forecast_len: int = 20
    GPU: bool = torch.cuda.is_available()

    def train_insample(self):
        return train_models(
            self.df,
            self.model_list,
            self.forecast_len,
            self.seasonality,
            self.train_proportion,
            self.GPU,
            True,
        )

    def train_outsample(self):

        return train_models(
            self.df,
            self.model_list,
            self.forecast_len,
            self.seasonality,
            self.train_proportion,
            self.GPU,
            False,
        )

    def forecast_insample(self):
        dataframe, _ = parse_data(self.df)
        _, test = train_test_split(
            dataframe, train_proportion=self.train_proportion
        )

        models_dict = self.train_insample()
        self.models_dict_insample = models_dict

        forecast_dict = forecast_models(models_dict, True)

        forecast_frame = forecast_dataframe(test, forecast_dict)

        preformance = insample_performance(forecast_frame)

        _LOG.info("Successfully finished in sample forecast")

        return forecast_frame, preformance

    def forecast_outsample(self):
        dataframe, freq = parse_data(self.df)

        models_dict = self.train_outsample()
        self.models_dict_outsample = models_dict

        forecast_dict = forecast_models(models_dict, False)

        dataframe = models_dict[list(models_dict.keys())[0]].dataframe

        future_index = pd.date_range(
            dataframe.index[-1], periods=self.forecast_len + 1, freq=freq
        )[1:]

        forecast_frame = forecast_dataframe(
            pd.DataFrame({"Target": 0}, index=future_index), forecast_dict
        )

        _LOG.info("Successfully finished out of sample forecast")
        return forecast_frame

    def ensemble(self, forecast_in, forecast_out):
        season = self.seasonality

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
