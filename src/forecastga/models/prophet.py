#! /usr/bin/env python
# coding: utf-8
#

"""Prophet Model"""

from fbprophet import Prophet

from base import BaseModel

class Prophet_Model(BaseModel):
    """Prophet Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self, df):
        df_pr = df.reset_index()
        df_pr.columns = ["ds", "y"]
        return df_pr

    def train(self):
        if freq == "D":
            ptm = Prophet(daily_seasonality=True)
        else:
            ptm = Prophet()
        self.model = ptm.fit(self.dataframe(train))

    def forecast(self):
        future = self.model.make_future_dataframe(periods=forecast_len, freq=freq)
        future_pred = self.model.predict(future)
        self.prediction = self.original_dataframe(future_pred, freq)[-forecast_len:]

    @staticmethod
    def original_dataframe(df, freq):
        prophet_pred = pd.DataFrame({"Date": df["ds"], "Target": df["yhat"]})
        prophet_pred = prophet_pred.set_index("Date")
        return prophet_pred["Target"].values
