#! /usr/bin/env python
# coding: utf-8
#

"""ARIMA Model"""

from forecastga.models.base import BaseModel


class ARIMA_Model:
    """ARIMA Model Class"""

    def __init__(self, config):
        super().__init__(config)

        """
        Available model attributes:

        self.seasonality (str)
        self.forecast_len (int)
        self.freq (str)
        self.train_proportion (float)
        self.in_sample (bool)
        self.GPU (bool)

        self.dataframe (pd.Series)
        self.train_df (pd.Series)
        self.forecast_df (pd.Series) or None
        self.seasons (int)
        self.periods (int)
        """

    def train(self):
        self.model = pm.auto_arima(self.train_df, seasonal=True, m=self.seasons)

    def forecast(self):
        if self.insample:
            self.prediction = self.model.predict(self.forecast_len)
            # Prediction can be a list, np.Array, or pandas series.
        else:
            # Do something else if outsample.
            pass

    @staticmethod
    def format_input(df, forecast_length, constant=None):
        pass

    @staticmethod
    def format_output(df, forecast_length, constant=None):
        pass
