#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: Prophet Model with Box-Cox transform of the data"""

import pandas as pd
from fbprophet import Prophet
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from forecastga.models.base import BaseModel


class Prophet_BoxCox_Model(BaseModel):
    """Prophet Boxcox Model Class"""

    def __init__(self, config):
        self.boxcox_lambda = None
        super().__init__(config)

    def train(self, **kwargs):

        country_holidays = kwargs.get("country_holidays", "US")

        if self.freq == "D":
            ptm = Prophet(weekly_seasonality=True)
        else:
            ptm = Prophet()

        ptm.add_country_holidays(country_name=country_holidays)

        formatted_data = self.format_input(self.train_df)
        transformed_y, self.boxcox_lambda = boxcox(formatted_data["y"]+1)
        formatted_data["y"] = transformed_y

        self.model = ptm.fit(formatted_data)

    def forecast(self):
        future = self.model.make_future_dataframe(
            periods=self.forecast_len, freq=self.freq
        )
        future_pred = self.model.predict(future)
        future_pred = future_pred[-self.forecast_len :]
        if self.boxcox_lambda:
            future_pred["yhat"] = inv_boxcox(future_pred["yhat"], self.boxcox_lambda) - 1
        self.prediction = self.format_output(future_pred)

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
