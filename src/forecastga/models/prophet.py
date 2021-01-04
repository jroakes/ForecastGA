#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Prophet Model"""

import pandas as pd
from fbprophet import Prophet

from forecastga.models.base import BaseModel


class Prophet_Model(BaseModel):
    """Prophet Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self, **kwargs):

        country_holidays = kwargs.get("country_holidays", "US")

        if self.freq == "D":
            ptm = Prophet(weekly_seasonality=True)
        else:
            ptm = Prophet()

        ptm.add_country_holidays(country_name=country_holidays)

        self.model = ptm.fit(self.format_input(self.train_df))

    def forecast(self):
        future = self.model.make_future_dataframe(
            periods=self.forecast_len, freq=self.freq
        )
        future_pred = self.model.predict(future)
        self.prediction = self.format_output(future_pred)[-self.forecast_len :]

    @staticmethod
    def format_input(df):
        df_pr = df.reset_index()
        df_pr.columns = ["ds", "y"]
        return df_pr

    @staticmethod
    def format_output(df):
        prophet_pred = pd.DataFrame({"Date": df["ds"], "Target": df["yhat"]})
        prophet_pred = prophet_pred.set_index("Date")
        return prophet_pred["Target"].values
