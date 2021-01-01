#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Base Model"""


class BaseModel:
    """Base Model class of ForecastGA"""

    def __init__(self, config):

        if config.in_sample is None:
            raise ValueError(
                "The config class must be initialized with \
                              `set_in_sample()` or `set_out_sample()` prior to \
                              passing to a model."
            )

        self.seasonality = config.seasonality
        self.forecast_len = config.forecast_len
        self.train_proportion = config.train_proportion
        self.in_sample = config.in_sample
        self.GPU = config.GPU

        self.dataframe = config.dataframe
        self.train_df = config.train_df
        self.forecast_df = config.forecast_df
        self.model = None
        self.prediction = None

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
