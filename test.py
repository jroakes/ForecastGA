# Libraries
import pandas as pd
import numpy as np
import re
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY

import matplotlib.pyplot as plt

import json
import forecastga
import forecastga.googleanalytics as ga


# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)



from types import SimpleNamespace

with open('identity.json') as f:
    jf = json.load(f)
    identify_json = SimpleNamespace(**jf)


#@title Google Analytics
ga_url = "https://analytics.google.com/analytics/web/?authuser=2#/report-home/a49839941w81675857p84563570" #@param {type:"string"}
ga_segment = "organic traffic" #@param ["all users", "organic traffic", "direct traffic", "referral traffic", "mobile traffic", "tablet traffic"] {type:"string"}
ga_metric = "sessions" #@param ["sessions", "pageviews", "unique pageviews", "transactions"] {type:"string"}


# @title Historical Data

# @markdown #### Date Range:
ga_start_date = "2018-01-01" #@param {type:"date"}
ga_end_date = "2019-12-31" #@param {type:"date",name:"GA Date"}

# @markdown ***
# @markdown <div align="center">OR</div>

# @markdown #### Prior Months:
prior_months = 0 #@param {type:"integer"}

# @title Prediction Data

future_months = 2 # @param {type:"slider", min:1, max:24, step:1}
#@markdown ---
#@markdown `max_available_volume` is the total possible daily volume for a niche/geo. This helps keep the algorithm honest by putting a max possible upper bound on prediction.
max_available_volume = 12222 # @param {type:"integer", hint:"this is a description"}
#@markdown ---
#@markdown `omit_values_over` is a way to clean your existing data to remove one-time spikes, caused by a rare event that is unlikely to happen again.  This keeps the algorithm from using this data in its future predictions.
omit_values_over = 2000000 # @param {type:"integer"}
#@markdown ---
save_output = False # @param {type:"boolean"}



try:
    profile = ga.authenticate(
      client_id=identify_json.client_id,
      client_secret=identify_json.client_secret,
      identity=identify_json.identity,
      ga_url=ga_url,
      interactive=True
  )
    print('Authenticated')
except Exception as e:
    print('An error occured', str(e))


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def p_date(_dt):
    return datetime.strftime(_dt, '%Y-%m-%d')

def get_months(start_date,end_date ):
    strt_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    return rrule(MONTHLY, dtstart=strt_dt, until=end_dt).count()



def get_ga_data(profile, data):

    try:

        if data.prior_months and int(data.prior_months) > 0:
            print('Pulling {} prior months data.'.format(data.prior_months))
            sessions = \
                profile.core.query.metrics(data.ga_metric).segment(data.ga_segment).daily(months=0
                    - int(data.prior_months)).report
        else:
            print('Pulling data from {} to {}.'.format(data.ga_start_date, data.ga_end_date))
            sessions = \
                profile.core.query.metrics(data.ga_metric).segment(data.ga_segment).daily(data.ga_start_date,
                    data.ga_end_date).report

    except Exception as e:
        print ('Error. Error retreiving data from Google Analytics.', str(e))
        return None

    df = sessions.as_dataframe()

    df['date'] = pd.to_datetime(df['date'])


    # Clean data.
    if data.omit_values_over and int(data.omit_values_over) > 0:
        df.loc[df[data.ga_metric] > data.omit_values_over, data.ga_metric] = np.nan

    df.loc[df[data.ga_metric] < 1, data.ga_metric] = np.nan

    df.dropna(inplace=True, axis=0)


    print('Rows: {rows} Min Date: {min_date} Max Date: {max_date}'.format(rows=len(df),
                                                                      min_date=p_date(df.date.min()),
                                                                      max_date=p_date(df.date.max())
                                                                      ))
    # Backfilling missing values
    df = df.set_index('date').asfreq('d', method='bfill')

    return df


data = Struct(**{
    'ga_segment': ga_segment,
    'ga_metric': ga_metric,
    'ga_start_date': ga_start_date,
    'ga_end_date': ga_end_date,
    'prior_months': prior_months,
    'omit_values_over': omit_values_over,
    })


datafile = get_ga_data(profile, data)

print(datafile.head())


model_list = ["TATS", "TBATS1", "TBATP1", "TBATS2", "ARIMA","Gluonts"]

am = forecastga.AutomatedModel(df = datafile['sessions'] , model_list=model_list, forecast_len=30 )

forecast_frame, preformance = am.forecast_insample()

forecast_frame.head()
