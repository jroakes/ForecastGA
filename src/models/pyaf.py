#! /usr/bin/env python
# coding: utf-8
#

"""PYAF Model"""

import pyaf.ForecastEngine as autof

from base import BaseModel

class PYAF_Model(BaseModel):
    """PYAF Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self, df):
        df_pr = df.reset_index()
        df_pr.columns = ["ds", "y"]
        return df_pr

    def train(self):
        af = autof()
        af.train(
            iInputDS=train.reset_index(),
            iTime="Date",
            iSignal="Target",
            iHorizon=len(train),
        )  # bad coding to have horison here
        self.model = af.forecast(
            iInputDS=train.reset_index(), iHorizon=forecast_len
        )

    def forecast(self):
        self.prediction = model["Target_Forecast"][-forecast_len:].values
