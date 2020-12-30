#! /usr/bin/env python
# coding: utf-8
#

"""Gluonts Model"""

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset

from base import BaseModel

class Gluonts_Model(BaseModel):
    """Gluonts Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self, df):
        freqed = pd.infer_freq(df.index)
        if freqed == "MS":
            freq = "M"
            # start = df.index[0] + relativedelta(months=1)
        else:
            freq = freqed
        df = ListDataset([{"start": df.index[0], "target": df.values}], freq=freq)
        return df

    def train(self):
        freqed = pd.infer_freq(train.index)
        if freqed == "MS":
            freq = "M"
        else:
            freq = freqed
        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=forecast_len,
            trainer=Trainer(epochs=6, ctx="gpu"),
        )  # use_feat_dynamic_real=True

        if GPU:
            self.model = estimator.train(training_data=self.dataframe(train))
        else:
            self.model = estimator.train(training_data=self.dataframe(train))


    def forecast(self):
        if freq == "MS":
            freq = "M"
        if in_sample:
            for df_entry, forecast in zip(
                self.dataframe(df), self.model.predict(self.dataframe(df))
            ):
                self.forecast = forecast.samples.mean(axis=0)
        else:
            future = ListDataset(
                [
                    {
                        "target": df[-forecast_len:],
                        "start": df.index[-1] + df.index.to_series().diff().min(),
                    }
                ],
                freq=freq,
            )
            # future = ListDataset([{"target": [df[-1]]*forecast_len, "start": df.index[-1] + relativedelta(months=1)}],freq=freq)

            for df_entry, forecast in zip(
                future, self.model.predict(future)
            ):  # next(predictor.predict(future))
                self.forecast = forecast.samples.mean(
                    axis=0
                )  # .quantile(0.5)
