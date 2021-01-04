#! /usr/bin/env python
# coding: utf-8
#
""" ForecastGA: Models """

MODELS = {
    "ARIMA": {
        "name": "ARIMA",
        "description": "Automated ARIMA Modelling",
        "loc": "forecastga.models.arima",
        "class": "ARIMA_Model",
        "status": "active",
    },
    "Prophet": {
        "name": "Prophet",
        "description": "Modeling Multiple Seasonality With Linear or Non-linear Growth",
        "loc": "forecastga.models.prophet",
        "class": "Prophet_Model",
        "status": "active",
    },
    "SARIMA": {
        "name": "SARIMA",
        "description": "A seasonal autoregressive integrated moving average (SARIMA) model",
        "loc": "forecastga.models.sarima",
        "class": "SARIMA_Model",
        "status": "not implemented",
    },
    "SARIMAX": {
        "name": "SARIMAX",
        "description": "Example: https://gist.github.com/natzir/befe1ff229fc2d0c01e0411d5fdd5209",
        "loc": "forecastga.models.sarimax",
        "class": "SARIMAX_Model",
        "status": "not implemented",
    },
    "HWAAS": {
        "name": "HWAAS",
        "description": "Exponential Smoothing With Additive Trend and Additive Seasonality",
        "loc": "forecastga.models.hwaas",
        "class": "HWAAS_Model",
        "status": "active",
    },
    "HWAMS": {
        "name": "HWAMS",
        "description": "Exponential Smoothing with Additive Trend and Multiplicative Seasonality",
        "loc": "forecastga.models.hwams",
        "class": "HWAMS_Model",
        "status": "active",
    },
    "NBEATS": {
        "name": "NBEATS",
        "description": "Neural basis expansion analysis (now fixed at 20 Epochs)",
        "loc": "forecastga.models.nbeats",
        "class": "NBEATS_Model",
        "status": "active",
    },
    "Gluonts": {
        "name": "Gluonts",
        "description": "RNN-based Model (now fixed at 20 Epochs)",
        "loc": "forecastga.models.gluonts",
        "class": "Gluonts_Model",
        "status": "active",
    },
    "TATS": {
        "name": "TATS",
        "description": "Seasonal and Trend no Box Cox",
        "loc": "forecastga.models.tats",
        "class": "TATS_Model",
        "status": "active",
    },
    "TBAT": {
        "name": "TBAT",
        "description": "Trend and Box Cox",
        "loc": "forecastga.models.tbat",
        "class": "TBAT_Model",
        "status": "active",
    },
    "TBATS1": {
        "name": "TBATS1",
        "description": "Trend, Seasonal (one), and Box Cox",
        "loc": "forecastga.models.tbats1",
        "class": "TBATS1_Model",
        "status": "active",
    },
    "TBATP1": {
        "name": "TBATP1",
        "description": "TBATS1 but Seasonal Inference is Hardcoded by Periodicity",
        "loc": "forecastga.models.tbatp1",
        "class": "TBATP1_Model",
        "status": "active",
    },
    "TBATS2": {
        "name": "TBATS2",
        "description": "TBATS1 With Two Seasonal Periods",
        "loc": "forecastga.models.tbats2",
        "class": "TBATS2_Model",
        "status": "active",
    },
    "PYAF": {
        "name": "PYAF",
        "description": "PYAF",
        "loc": "forecastga.models.pyaf",
        "class": "PYAF_Model",
        "status": "active but not tested",
    },
}
