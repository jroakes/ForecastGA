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
from forecastga.helpers.data import parse_data, train_test_split, select_seasonality
from forecastga.ensembles import (
    ensemble_lightgbm,
    ensemble_tsfresh,
    ensemble_pure,
    middle,
    ensemble_first,
    ensemble_doubled,
)
from forecastga.models import MODELS
import forecastga.googleanalytics as ga

pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")

_LOG = get_logger(__name__)


__version__ = "0.1.0"


class ModelConfig:
    def __init__(
        self,
        df: pd.Series,
        seasonality: str = "infer_from_data",
        forecast_len: int = 20,
        train_proportion: float = 0.75,
        GPU: bool = torch.cuda.is_available(),
    ):

        self.df = df
        self.seasonality = seasonality
        self.forecast_len = forecast_len
        self.train_proportion = train_proportion
        self.GPU = GPU

        self.in_sample = None
        self.train_df = None
        self.forecast_df = None
        self.seasons = None
        self.periods = None

        self.dataframe, self.freq = parse_data(df)

    def set_in_sample(self):

        self.in_sample = True

        self.train_df, self.forecast_df = train_test_split(
            self.dataframe, train_proportion=self.train_proportion
        )
        self.forecast_len = len(self.forecast_df)

        self.seasons = select_seasonality(self.train_df, self.seasonality)
        self.periods = select_seasonality(self.train_df, "periodocity")

    def set_out_sample(self):

        self.in_sample = False

        self.train_df, self.forecast_df = self.dataframe["Target"], None

        self.seasons = select_seasonality(self.train_df, self.seasonality)
        self.periods = select_seasonality(self.train_df, "periodocity")



class AutomatedModel:

    def __init__(self,
                df: pd.Series,
                model_list: list,
                seasonality: str = "infer_from_data",
                train_proportion: float = 0.75,
                forecast_len: int = 20,
                GPU: bool = torch.cuda.is_available()
                ):

        self.df = df
        self.model_list = model_list
        self.seasonality = seasonality
        self.train_proportion = train_proportion
        self.forecast_len = forecast_len
        self.GPU = GPU
        self.models_dict = {}
        self.forecast_dict = {}

        self.config: ModelConfig = ModelConfig(
            df,
            seasonality=seasonality,
            forecast_len=forecast_len,
            train_proportion=train_proportion,
            GPU=GPU,
        )

    def forecast_insample(self, **kwargs):

        self.config.set_in_sample()

        self.models_dict = self.__train_models(**kwargs)
        self.forecast_dict = self.__forecast_models()
        forecast_frame = self.forecast_dataframe(self.config.test, self.forecast_dict)
        preformance = self.insample_performance(forecast_frame)

        _LOG.info("Successfully finished in sample forecast")

        return forecast_frame, preformance

    def forecast_outsample(self, **kwargs):

        self.config.set_out_sample()

        self.models_dict = self.__train_models(**kwargs)
        self.forecast_dict = self.__forecast_models()

        future_index = pd.date_range(
            self.config.dataframe.index[-1],
            periods=self.config.forecast_len + 1,
            freq=self.config.freq,
        )[1:]

        forecast_frame = self.forecast_dataframe(
            pd.DataFrame({"Target": 0}, index=future_index), self.forecast_dict
        )

        _LOG.info("Successfully finished out of sample forecast")

        return forecast_frame

    def print_model_info(self):
        _ = [print(v['name'], ":", v['description']) for k, v in MODELS.items() if v["status"] == "active"]

    def available_models(self):
        return [k for k, v in MODELS.items() if v["status"] == "active"]

    def __train_models(self, **kwargs):

        models_dict = {}

        for model_name in self.model_list:
            if model_name not in self.available_models():
                _LOG.warning(
                    "Model {} is not available.  Skipping...".format(model_name)
                )
                continue

            _LOG.info(
                "Model {} is being loaded and \
                       trained for {} prediction".format(
                    model_name,
                    "in sample" if self.config.in_sample else "out of sample",
                )
            )
            model_data = MODELS[model_name]
            module = importlib.import_module(model_data["loc"])
            module_class = getattr(module, model_data["class"])
            model = module_class(self.config)

            model.train(**kwargs)

            models_dict[model_name] = model

        return models_dict

    def __forecast_models(self, models_dict=None):

        models_dict = models_dict or self.models_dict

        forecast_dict = {}
        for model_name, model in models_dict.items():

            _LOG.info(
                "Model {} is being used to forecast {}".format(
                    model_name,
                    "in sample" if self.config.in_sample else "out of sample",
                )
            )

            model.forecast()

            forecast_dict[model_name] = model.prediction

        return forecast_dict

    @staticmethod
    def forecast_dataframe(df, forecast_dict):
        insample = df.to_frame()
        for name, forecast in forecast_dict.items():
            insample[name] = forecast
        return insample

    @staticmethod
    def insample_performance(forecast_frame, as_dict=False):

        dict_perf = {}
        for col, _ in forecast_frame.iteritems():
            dict_perf[col] = {}
            dict_perf[col]["rmse"] = rmse(forecast_frame["Target"], forecast_frame[col])
            dict_perf[col]["mse"] = dict_perf[col]["rmse"] ** 2
            dict_perf[col]["mean"] = forecast_frame[col].mean()
        if as_dict:
            return dict_perf

        return pd.DataFrame.from_dict(dict_perf)

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
