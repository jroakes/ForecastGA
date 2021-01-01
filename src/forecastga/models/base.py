#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Base Model"""

import pandas as pd
import torch

from forecastga.helpers.data import parse_data, train_test_split, select_seasonality


class BaseModel:
    """Base Model class of ForecastGA"""

    def __init__(
        self,
        df: pd.Series,
        seasonality: str = "infer_from_data",
        forecast_len: int = 20,
        train_proportion: float = 0.75,
        GPU: bool = torch.cuda.is_available(),
        in_sample: bool = True,
    ):

        self.seasonality = seasonality
        self.forecast_len = forecast_len
        self.train_proportion = train_proportion
        self.in_sample = in_sample
        self.GPU = GPU

        self.dataframe = None
        self.train_df = None
        self.forecast_df = None
        self.model = None
        self.prediction = None

        self.format_data(df)

    def format_data(self, df):
        self.dataframe, self.freq = parse_data(df)

        if self.in_sample:
            self.train_df, self.forecast_df = train_test_split(
                self.dataframe, train_proportion=self.train_proportion
            )
            self.forecast_len = len(self.forecast_df)

        else:
            self.train_df, self.forecast_df = self.dataframe["Target"], None

        self.seasons = select_seasonality(self.train_df, self.seasonality)
        self.periods = select_seasonality(self.train_df, "periodocity")

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
