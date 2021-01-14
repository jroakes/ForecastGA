#! /usr/bin/env python
# coding: utf-8


"""ForecastGA: Main
"""


__version__ = "0.1.15"


from forecastga.helpers.ga_data import get_ga_data
from forecastga.auto import AutomatedModel
from forecastga.helpers.data import print_model_info
from forecastga.helpers.colab import plot_colab


def help():
    print("Welcome to ForecastGA")
    print()
    print("To use:")
    print()
    print("Find Model Info:")
    print("forecastga.print_model_info()")
    print()
    print("Initialize Model:")
    print()
    print(
        """
    Google Analytics:

        data = { 'client_id': '<google api client_id>',
                 'client_secret': '<google api client_secret>',
                 'ga_end_date': '2019-12-31',
                 'ga_metric': 'sessions',
                 'ga_segment': 'organic traffic',
                 'ga_start_date': '2018-01-01',
                 'ga_url': 'https://analytics.google.com/analytics/web/?authuser=2#/report-home/aXXXXXwXXXXXpXXXXXX',
                 'identity': '<google api identity>',
                 'omit_values_over': 2000000
                }

        model_list = ["TATS", "TBATS1", "TBATP1", "TBATS2", "ARIMA"]
        am = forecastga.AutomatedModel(data , model_list=model_list, forecast_len=30 )

         """
    )
    print()
    print(
        """
    Pandas DataFrame:

        # CSV with columns: Date and Sessions
        df = pd.read_csv('ga_sessions.csv')
        df.Date = pd.to_datetime(df.Date)
        df = df.set_index("Date")
        data = df.Sessions

        model_list = ["TATS", "TBATS1", "TBATP1", "TBATS2", "ARIMA"]
        am = forecastga.AutomatedModel(data , model_list=model_list, forecast_len=30 )

         """
    )
    print()
    print("Forecast Insample:")
    print("forecast_in, performance = am.forecast_insample()")
    print()
    print("Forecast Outsample:")
    print("forecast_out = am.forecast_outsample()")
    print()
    print("Ensemble Performance:")
    print(
        "all_ensemble_in, all_ensemble_out, all_performance = am.ensemble(forecast_in, forecast_out)"
    )
    print()
    print("Pretty Plot in Google Colab")
    print(
        'forecastga.plot_colab(forecast_in, title="Insample Forecast", dark_mode=True)'
    )
