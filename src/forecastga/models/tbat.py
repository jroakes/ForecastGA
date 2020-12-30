#! /usr/bin/env python
# coding: utf-8
#

"""TBAT Model"""

from tbats import TBATS

from base import BaseModel


class TBAT_Model(BaseModel):
    """TBAT Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        bat = TBATS(use_arma_errors=False, use_box_cox=True, use_trend=True)
        self.model = bat.fit(train)

    def forecast(self):
        self.forecast = model.forecast(forecast_len)
