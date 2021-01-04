# ForecastGA
A Python tool to forecast GA data using several popular timeseries models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nmcu37MY02dfMdUbinrwwg7gA9ya3eud?usp=sharing)


## About

### Welcome to ForecastGA

ForecastGA is a tool that combines a couple of popular libraries, [Atspy](https://github.com/firmai/atspy) and [googleanalytics](https://github.com/debrouwere/google-analytics), with a few enhancements.

* The models are made more intuitive to upgrade and add by having the tool logic separate from the model training and prediction.
* When calling `am.forecast_insample()`, any kwargs included (e.g. `learning_rate`) are passed to the train method of the model.
* Google Analytics profiles are specified by simply passing the URL (e.g. https://analytics.google.com/analytics/web/?authuser=2#/report-home/aXXXXXwXXXXXpXXXXXX).
* You can provide a `data` dict with GA config options or a Pandas Series as the input data.
* Multiple log levels.
* Auto GPU detection (via Torch).
* List all available models, with descriptions, by calling `forecastga.print_model_info()`.
* Google API info can be passed in the `data` dict or uploaded as a JSON file named `identity.json`.
* Created a companion Google Colab notebook to easily run on GPU.
* A handy plot function for Colab, `forecastga.plot_colab(forecast_in, title="Insample Forecast", dark_mode=True)` that formats nicely and also handles Dark Mode!

### Models Available
* `ARIMA` : Automated ARIMA Modelling
* `Prophet` : Modeling Multiple Seasonality With Linear or Non-linear Growth
* `HWAAS` : Exponential Smoothing With Additive Trend and Additive Seasonality
* `HWAMS` : Exponential Smoothing with Additive Trend and Multiplicative Seasonality
* `NBEATS` : Neural basis expansion analysis (now fixed at 20 Epochs)
* `Gluonts` : RNN-based Model (now fixed at 20 Epochs)
* `TATS` : Seasonal and Trend no Box Cox
* `TBAT` : Trend and Box Cox
* `TBATS1` : Trend, Seasonal (one), and Box Cox
* `TBATP1` : TBATS1 but Seasonal Inference is Hardcoded by Periodicity
* `TBATS2` : TBATS1 With Two Seasonal Periods


### How To Use

#### Find Model Info:
`forecastga.print_model_info()`

#### Initialize Model:

##### Google Analytics:

```
data = { 'client_id': '<google api client_id>',
         'client_secret': '<google api client_secret>',
         'identity': '<google api identity>',
         'ga_start_date': '2018-01-01',
         'ga_end_date': '2019-12-31',
         'ga_metric': 'sessions',
         'ga_segment': 'organic traffic',
         'ga_url': 'https://analytics.google.com/analytics/web/?authuser=2#/report-home/aXXXXXwXXXXXpXXXXXX',
         'omit_values_over': 2000000
        }

model_list = ["TATS", "TBATS1", "TBATP1", "TBATS2", "ARIMA"]
am = forecastga.AutomatedModel(data , model_list=model_list, forecast_len=30 )
```

##### Pandas DataFrame:

```
# CSV with columns: Date and Sessions
df = pd.read_csv('ga_sessions.csv')
df.Date = pd.to_datetime(df.Date)
df = df.set_index("Date")
data = df.Sessions

model_list = ["TATS", "TBATS1", "TBATP1", "TBATS2", "ARIMA"]
am = forecastga.AutomatedModel(data , model_list=model_list, forecast_len=30 )
```

#### Forecast Insample:
`forecast_in, performance = am.forecast_insample()`

#### Forecast Outsample:
`forecast_out = am.forecast_outsample()`

#### Ensemble Performance:
`all_ensemble_in, all_ensemble_out, all_performance = am.ensemble(forecast_in, forecast_out)`

#### Pretty Plot in Google Colab
`forecastga.plot_colab(forecast_in, title="Insample Forecast", dark_mode=True)`


# Installation
Windows users may need to manually install the two items below via conda :
1. `conda install pystan`
1. `conda install pytorch -c pytorch`
1. `!pip install --upgrade git+https://github.com/jroakes/ForecastGA.git`

otherwise,
`pip install --upgrade forecastga`

This repo support GPU training. Below are a few libraries that may have to be manually installed to support.
```
pip install --upgrade mxnet-cu101
pip install --upgrade torch 1.7.0+cu101
```


## Acknowledgements

1. Majority of forecasting code taken from https://github.com/firmai/atspy and refactored heavily.
1. Google Analytics based off of: https://github.com/debrouwere/google-analytics

## Contribute
The goal of this repo is to grow the list of available models to test.  If you would like to contribute one please read on.  Feel free to have fun naming your models.

1. Fork the repo.
2. In the `/src/forecastga/models` folder there is a model called `template.py`.  You can use this as a template for creating your new model.  All available variables are there. Forecastga ensures each model has the right data and calls only the `train` and `forecast` methods for each model. Feel free to add additional methods that your model requires.
3. Edit the `/src/forecastga/models/__init__.py` file to add your model's information.  Follow the format of the other entries.  Forecastga relies on `loc` to find the model and `class` to find the class to use.
4. Edit `requirments.txt` with any additional libraries needed to run your model.  Keep in mind that this repo should support GPU training if available and some libraries have separate GPU-enabled versions.
5. Issue a pull request.

If you enjoyed this tool consider buying me some beer at: [Paypalme](https://www.paypal.com/paypalme/codeseo)
