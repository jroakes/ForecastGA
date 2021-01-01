#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: PYAF Model"""

import pyaf.ForecastEngine as autof

from base import BaseModel


class PYAF_Model(BaseModel):
    """PYAF Model Class"""

    def train(self, **kwargs):
        self.model = autof()
        self.model.train(
            iInputDS=self.train_df.reset_index(),
            iTime="Date",
            iSignal="Target",
            iHorizon=len(self.train_df),
        )

    def forecast(self):
        self.model.forecast(
            iInputDS=self.train_df.reset_index(), iHorizon=self.forecast_len
        )
        self.prediction = self.model["Target_Forecast"][-self.forecast_len :].values
