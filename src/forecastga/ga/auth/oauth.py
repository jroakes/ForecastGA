# encoding: utf-8

import json
import webbrowser

import addressable
from oauth2client import client
from apiclient import discovery

from forecastga.googleanalytics import utils, account
from .credentials import Credentials, normalize


class Flow(client.OAuth2WebServerFlow):
    def __init__(self, client_id, client_secret, redirect_uri):
        super(Flow, self).__init__(client_id, client_secret,
            scope='https://www.googleapis.com/auth/analytics.readonly',
            redirect_uri=redirect_uri)

    def step2_exchange(self, code):
        credentials = super(Flow, self).step2_exchange(code)
        return Credentials.find(complete=True, **credentials.__dict__)

# a simplified version of `oauth2client.tools.run_flow`
def authorize(client_id, client_secret):
    flow = Flow(client_id, client_secret,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob')

    authorize_url = flow.step1_get_authorize_url()
    print ('Go to the following link in your browser: ' + authorize_url)
    code = input('Enter verification code: ').strip()
    return flow.step2_exchange(code)

@normalize
def revoke(credentials):
    return credentials.revoke()

@normalize
def authenticate(credentials):
    client = credentials.authorize()
    service = discovery.build('analytics', 'v3', http=client, cache_discovery=False)
    raw_accounts = service.management().accounts().list().execute()['items']
    accounts = [account.Account(raw, service, credentials) for raw in raw_accounts]
    return addressable.List(accounts, indices=['id', 'name'], insensitive=True)
