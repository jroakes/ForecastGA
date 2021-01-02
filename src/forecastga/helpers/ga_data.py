#! /usr/bin/env python
# coding: utf-8

"""ForecastGA: Google Analytics Helper Functions"""

import os
from types import SimpleNamespace
from datetime import datetime
import pandas as pd

from forecastga.helpers.logging import get_logger

from forecastga import ga

_LOG = get_logger(__name__)

def load_identity(data=None):

    if data:
        jf = {k.lower():v for k,v in data.items() if k.lower() in ['client_id', 'client_secret', 'identity']}
        return SimpleNamespace(**jf)

    if not os.path.isfile('identity.json'):
        raise FileExistsError('A JSON file named `identity.json` must be accessible with your API credentials.')

    with open('identity.json') as f:
        jf = json.load(f)
        identify_json = SimpleNamespace(**jf)

    return identify_json


def load_profile(ga_url, identify_ns):

    try:
        profile = ga.authenticate(
          client_id=identify_ns.client_id,
          client_secret=identify_ns.client_secret,
          identity=identify_ns.identity,
          ga_url=ga_url,
          interactive=True
          )
        _LOG.info('Authenticated')
        return profile

    except Exception as e:
        _LOG.error('An error occured: ' + str(e))
        return None

def p_date(_dt):
  return datetime.strftime(_dt, '%Y-%m-%d')


def get_ga_data(data):

    if 'client_id' in data and 'client_secret' in data and 'identity' in data :
        identify_ns = load_identity(data)
    else:
        identify_ns = load_identity()

    if 'ga_url' not in data:
        raise AttributeError('You must provide the URL for your Google Analytics property.')

    profile = load_profile(data['ga_url'], identify_ns)

    if profile is None:
        return None

    try:
        print('Pulling data from {} to {}.'.format(data.ga_start_date, data.ga_end_date))
        sessions = \
            profile.core.query.metrics(data.ga_metric).segment(data.ga_segment).daily(data.ga_start_date,
                data.ga_end_date).report

    except Exception as e:
        _LOG.error('Error. Error retreiving data from Google Analytics.', str(e))
        return None


    df = sessions.as_dataframe()

    df['date'] = pd.to_datetime(df['date'])

    # Clean data.
    if data.omit_values_over and int(data.omit_values_over) > 0:
        df.loc[df[data.ga_metric] > data.omit_values_over, data.ga_metric] = np.nan

    df.loc[df[data.ga_metric] < 1, data.ga_metric] = np.nan

    df.dropna(inplace=True, axis=0)

    _LOG.info('Rows: {rows} Min Date: {min_date} Max Date: {max_date}'.format(rows=len(df),
                                                                      min_date=p_date(df.date.min()),
                                                                      max_date=p_date(df.date.max())
                                                                      ))
    # Backfilling missing values
    df = df.set_index('date').asfreq('d', method='bfill')

    return df[data.ga_metric]
