#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Gluonts Model"""
import pandas as pd

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset

from forecastga.models.base  import BaseModel


class Gluonts_Model(BaseModel):
    """Gluonts Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self, **kwargs):

        # Adjust class freq.
        self.freq = pd.infer_freq(self.train_df.index)
        if self.freq == "MS":
            self.freq = "M"

        estimator = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.forecast_len,
            trainer=Trainer(epochs=6, ctx="gpu" if self.GPU else "cpu"),
        )

        self.model = estimator.train(
            training_data=self.format_input(self.train_df, self.freq)
        )

    def forecast(self):

        if self.in_sample:
            forecast = self.model.predict(
                self.format_input(self.forecast_df, self.freq)
            )
        else:
            forecast = self.model.predict(
                self.format_input(
                    self.train_df.tail(self.forecast_len),
                    self.freq,
                    self.train_df.index[-1]
                    + self.train_df.index.to_series().diff().min(),
                )
            )

        self.prediction = list(forecast)[0].samples.mean(axis=0)  # .quantile(0.5)

    @staticmethod
    def format_input(df, freq, start=None):
        if start:
            return ListDataset([{"start": start, "target": df.values}], freq=freq)

        return ListDataset([{"start": df.index[0], "target": df.values}], freq=freq)
