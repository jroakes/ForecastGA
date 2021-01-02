# ForecastGA: A Python tool to forecast GA data using several popular timeseries models.

## About

### Welcome to ForecastGA

To use:

#### Find Model Info:
`forecastga.print_model_info()``

#### Initialize Model:

##### Google Analytics:

```
data = { 'client_id': '<google api client_id>',
         'client_secret': '<google api client_secret>',
         'ga_end_date': '2019-12-31',
         'ga_metric': 'sessions',
         'ga_segment': 'organic traffic',
         'ga_start_date': '2018-01-01',
         'ga_url': 'https://analytics.google.com/analytics/web/?authuser=2#/report-home/a49839941w81675857p84563570',
         'identity': '<google api identity>',
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
`forecast_in, performance = am.forecast_insample()``

#### Forecast Outsample:
`forecast_out = am.forecast_outsample()``

#### Ensemble Performance:
`all_ensemble_in, all_ensemble_out, all_performance = am.ensemble(forecast_in, forecast_out)``

#### Pretty Plot in Google Colab
`forecastga.plot_colab(forecast_in, title="Insample FOrecast", dark_mode=True)``


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
