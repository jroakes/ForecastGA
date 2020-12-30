#! /usr/bin/env python
# coding: utf-8
#
""" Atspy Models """

MODELS = {
    "ARIMA": {
        "name": "ARIMA",
        "description": "Automated ARIMA Modelling",
        "loc": "arima.py",
        "class": "ARIMA_Model",
    },
    "Prophet": {
        "name": "Prophet",
        "description": "Modeling Multiple Seasonality With Linear or Non-linear Growth",
        "loc": "prophet.py",
        "class": "Prophet_Model",
    },
    "SARIMA": {
        "name": "SARIMA",
        "description": "A seasonal autoregressive integrated moving average (SARIMA) model",
        "loc": "sarima.py",
        "class": "SARIMA_Model",
    },
    "SARIMAX": {
        "name": "SARIMAX",
        "description": "Example: https://gist.github.com/natzir/befe1ff229fc2d0c01e0411d5fdd5209",
        "loc": "sarimax.py",
        "class": "SARIMAX_Model",
    },
    "HWAAS": {
        "name": "HWAAS",
        "description": "Exponential Smoothing With Additive Trend and Additive Seasonality",
        "loc": "hwaas.py",
        "class": "HWAAS_Model",
    },
    "HWAMS": {
        "name": "HWAMS",
        "description": "Exponential Smoothing with Additive Trend and Multiplicative Seasonality",
        "loc": "hwams.py",
        "class": "HWAMS_Model",
    },
    "NBEATS": {
        "name": "NBEATS",
        "description": "Neural basis expansion analysis (now fixed at 20 Epochs)",
        "loc": "arima.py",
        "class": "NBEATS_Model",
    },
    "Gluonts": {
        "name": "Gluonts",
        "description": "RNN-based Model (now fixed at 20 Epochs)",
        "loc": "gluonts.py",
        "class": "Gluonts_Model",
    },
    "TATS": {
        "name": "TATS",
        "description": "Seasonal and Trend no Box Cox",
        "loc": "tats.py",
        "class": "TATS_Model",
    },
    "TBAT": {
        "name": "TBAT",
        "description": "Trend and Box Cox",
        "loc": "tbat.py",
        "class": "TBAT_Model",
    },
    "TBATS1": {
        "name": "TBATS1",
        "description": "Trend, Seasonal (one), and Box Cox",
        "loc": "tbats1.py",
        "class": "TBATS1_Model",
    },
    "TBATP1": {
        "name": "TBATP1",
        "description": "TBATS1 but Seasonal Inference is Hardcoded by Periodicity",
        "loc": "tbatp1.py",
        "class": "TBATP1_Model",
    },
    "TBATS2": {
        "name": "TBATS2",
        "description": "TBATS1 With Two Seasonal Periods",
        "loc": "tbats2.py",
        "class": "TBATS2_Model",
    },
    "PYAF": {
        "name": "PYAF",
        "description": "PYAF",
        "loc": "pyaf.py",
        "class": "PYAF_Model",
    },
}
