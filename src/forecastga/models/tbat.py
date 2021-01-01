#! /usr/bin/env python
# coding: utf-8
#

"""TBAT Model"""

from tbats import TBATS

from base import BaseModel


class TBAT_Model(BaseModel):
    """TBAT Model Class"""

    def train(self, **kwargs):
        bat = TBATS(use_arma_errors=False, use_box_cox=True, use_trend=True)
        self.model = bat.fit(self.train_df)

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
