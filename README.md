# ForecastGA: A Python tool to forecast GA data using several popular timeseries models.

Google Colab Notebook: https://colab.research.google.com/drive/1CVj6ObeR9BeqJXDj4J1TrFBpHuQS8MwO

## About

### Welcome to ForecastGA

To use:

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
data = pd.read_csv('ts.csv').sessions

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
`forecastga.plot_colab(forecast_in, title="Insample FOrecast", dark_mode=True)`


# Installation
Windows users may need to manually install the two items below via conda :
1. `conda install pystan`
1. `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
1. `!pip install --upgrade git+https://github.com/jroakes/ForecastGA.git`

otherwise,
`!pip install --upgrade git+https://github.com/jroakes/ForecastGA.git`

This repo support GPU training. Below are a few libraries that may have to be manually installed to support.
`pip install --upgrade mxnet-cu101==1.7.0`


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
