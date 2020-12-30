from .ssa import mySSA
from .nbeats import plot_scatter, data_generator, train_100_grad_steps, load, save, eval_test
#from .pyaf import cForecastEngine as autof

## Traditional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import os
import seaborn as sns
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from matplotlib.pylab import rcParams
from scipy.signal import find_peaks
import lightgbm as lgb
from sklearn.model_selection import train_test_split as tts

warnings.filterwarnings("ignore")

## New
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
import mxnet as mx
from dataclasses import dataclass
import pmdarima as pm
from matplotlib.pylab import rcParams
from scipy.signal import find_peaks
pd.plotting.register_matplotlib_converters()
#import MySSA
from sklearn.model_selection import train_test_split as tts
#import pyaf.ForecastEngine as autof
from datetime import timedelta
warnings.filterwarnings("ignore")
from gluonts.model.prophet import ProphetPredictor
from dateutil.relativedelta import relativedelta
import torch
from torch import optim
from torch.nn import functional as F
from nbeats_pytorch.model import NBeatsNet # some import from the trainer script e.g. load/save functions.
from tbats import TBATS, BATS
from tsfresh.utilities.dataframe_functions import impute, roll_time_series
from tsfresh import extract_features
import pandas as pd
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import lightgbm as lgb
from seasonal.periodogram import periodogram
# import tensorflow as tf
# tf.test.is_gpu_available()






###### Utility Functions
#==================================================================

## Removing features that show the same value for the majority/all of the observations
def constant_feature_detect(data,threshold=0.98):
    """ detect features that show the same value for the
    majority/all of the observations (constant/quasi-constant features)

    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable as constant

    Returns
    -------
    list of variables names
    """

    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    print(len(quasi_constant_feature),' variables are found to be almost constant')
    return quasi_constant_feature

## More Diverse Selection For TBAT
def infer_seasonality_ssa(train,index=1): ##skip the first one, normally
  ssa = mySSA(train)
  ssa.embed(embedding_dimension=36, verbose=False)
  ssa.decompose(True)
  rec = ssa.view_reconstruction(ssa.Xs[index], names="Seasonality", return_df=True,plot=False)
  peaks, _ = find_peaks(rec.values.reshape(len(rec),), height=0)
  peak_diffs = [j-i for i, j in zip(peaks[:-1], peaks[1:])]
  seasonality = max(peak_diffs,key=peak_diffs.count)
  return seasonality

## Good First Selection
def infer_seasonality(train,index=0): ##skip the first one, normally
    interval, power = periodogram(train, min_period=4, max_period=None)
    try:
      season = int(pd.DataFrame([interval, power]).T.sort_values(1,ascending=False).iloc[0,index])
    except:
      print("Welch Season failed, defaulting to  SSA solution")
      season = int(infer_seasonality_ssa(train,index=1))
    return season

def infer_periodocity(train):
  perd = pd.infer_freq(train.index)
  if perd in ["MS","M","BM","BMS"]:
    periodocity = 12
  elif perd in ["BH","H"]:
    periodocity = 24
  elif perd=="B":
    periodocity = 5
  elif perd=="D":
    periodocity = 7
  elif perd in ["W","W-SUN","W-MON","W-TUE","W-WED","W-THU","W-FRI","W-SAT"]:
    periodocity = 52
  elif perd in ["Q","QS","BQ","BQS"]:
    periodocity = 4
  elif perd in ["A","BA","AS","BAS"]:
    periodocity = 10
  elif perd in ["T","min"]:
    periodocity = 60
  elif perd=="S":
    periodocity = 60
  elif perd in ["L","ms"]:
    periodocity = 1000
  elif perd in ["U","us"]:
    periodocity = 1000
  elif perd=="N":
    periodocity = 1000

  return periodocity

def select_seasonality(train, season):
  if season == "periodocity":
    seasonality = infer_periodocity(train)
  elif season== "infer_from_data":
    seasonality = infer_seasonality(train)
  return seasonality


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """
    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx

def parse_data(df):
  if type(df) ==pd.DataFrame:
    if df.shape[1]>1:
      raise ValueError("The dataframe should only contain one target column")
  elif type(df) == pd.Series:
    df = df.to_frame()
  else:
    raise TypeError("Please supply a pandas dataframe with one column or a pandas series")
  try:
    df.index.date
  except AttributeError:
    raise TypeError("The index should be a datetype")
  print(type(df))
  if df.isnull().any().values[0]:
    raise ValueError("The dataframe cannot have any null values, please interpolate")
  try:
    df.columns = ["Target"]
  except:
    raise ValueError("There should only be one column")

  df.index = df.index.rename("Date")
  df.index = add_freq(df.index)

  print("The data has been successfully parsed by infering a frequency, and establishing a 'Date' index and 'Target' column.")

  return df, pd.infer_freq(df.index)

def gluonts_dataframe(df):
  freqed = pd.infer_freq(df.index)
  if freqed=="MS":
    freq= "M"
    #start = df.index[0] + relativedelta(months=1)
  else:
    freq= freqed
  df = ListDataset(
    [{"start": df.index[0], "target": df.values}],
    freq =freq )
  return df

def nbeats_dataframe(df, forecast_length, in_sample,device, train_portion=0.75):

  backcast_length = 1 * forecast_length

  df = df.values  # just keep np array here for simplicity.
  norm_constant = np.max(df)
  df = df / norm_constant  # small leak to the test set here.

  x_train_batch, y = [], []

  for i in range(backcast_length+1, len(df) - forecast_length +1): #25% to 75% so 50% #### Watch out I had to plus one.
      x_train_batch.append(df[i - backcast_length:i])
      y.append(df[i:i + forecast_length])

  x_train_batch = np.array(x_train_batch)[..., 0]
  y = np.array(y)[..., 0]

  #x_train, y_train = x_train_batch, y

  net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
        forecast_length=forecast_length,
        thetas_dims=[7, 8],
        nb_blocks_per_stack=3,
        backcast_length=backcast_length,
        hidden_layer_units=128,
        share_weights_in_stack=False,
        device=device)

  if in_sample==True:
    c = int(len(x_train_batch) * train_portion)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]

    return x_train, y_train, x_test, y_test, net, norm_constant

  else:
    c = int(len(x_train_batch) * 1)
    x_train, y_train = x_train_batch[:c], y[:c]

    return x_train, y_train, net, norm_constant


def train_test_split(df, train_proportion=0.75):

  size = int(df['Target'].shape[0]*train_proportion); print(size)
  train, test = tts(df['Target'], train_size=size,shuffle=False, stratify=None)
  print("An insample split of training size {} and testing size {} has been constructed".format(len(train), len(test)))
  return train, test

## Dictionary Parameters - Makes sense topic area.
## In this case it would be very easy to make an app available online.
## This would allow you to play around with APIs and that sort of thing.


def prophet_dataframe(df):
  df_pr = df.reset_index()
  df_pr.columns = ['ds','y']
  return df_pr

# def original_dataframe(df, freq):
#   prophet_pred = pd.Series(df["yhat"].values, index =df['ds'])
#   prophet_pred = prophet_pred.rename("Target")
#   prophet_pred.index.name = 'Date'

#   print(prophet_pred)
#   print(prophet_pred.index)
#   print(prophet_pred.shape)
#   return prophet_pred

def original_dataframe(df, freq):
  prophet_pred = pd.DataFrame({"Date" : df['ds'], "Target" : df["yhat"]})
  prophet_pred = prophet_pred.set_index("Date")
  #prophet_pred.index.freq = pd.tseries.frequencies.to_offset(freq)
  return prophet_pred["Target"].values


def season_list(train):
  lista = []
  for i in range(15):
    i = 1+i
    lista.append(infer_seasonality_ssa(train,i))
  return lista

def get_unique_N(iterable, N):
    """Yields (in order) the first N unique elements of iterable.
    Might yield less if data too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        yield e
        if len(seen) == N:
            print("The following set of plausible SSA seasonalities have been identified: {}".format(seen))
            return


# device = torch.device('cpu')  # use the trainer.py to run on GPU.
# device = torch.device('cuda')

CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'

def train_models(train, models,forecast_len, full_df=None,seasonality="infer_from_data",in_sample=None, freq=None, GPU=None):

  seasons = select_seasonality(train, seasonality)

  periods = select_seasonality(train, 'periodocity')

  models_dict = {}
  for m in models:
    if in_sample:
      print("Model {} is being trained for in sample prediction".format(m))
    else:
      print("Model {} is being trained for out of sample prediction".format(m))
    if m=="ARIMA":
      models_dict[m] = pm.auto_arima(train, seasonal=True, m=seasons)
    if m=="Prophet":
      if freq=="D":
        model = Prophet(daily_seasonality=True)
      else:
        model = Prophet()
      models_dict[m] = model.fit(prophet_dataframe(train))
    if m=="HWAAS":
      try:
        models_dict[m] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
      except:
        models_dict[m] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='add', damped=True).fit(use_boxcox=False)
    if m=="HWAMS":
      try:
        models_dict[m] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
      except:
        try:
          models_dict[m] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='mul', damped=True).fit(use_boxcox=False)
        except:
          models_dict[m] = ExponentialSmoothing(train, seasonal_periods=seasons, trend=None, seasonal='add').fit(use_boxcox=False)

    # if m=="HOLT":
    #   models_dict["HOLT"] = Holt(train,exponential=True).fit()
    if m=="PYAF":
      model = autof()
      model.train(iInputDS = train.reset_index(), iTime = 'Date', iSignal = 'Target', iHorizon = len(train)) # bad coding to have horison here
      models_dict[m] = model.forecast(iInputDS = train.reset_index(), iHorizon = forecast_len)
    if m=="Gluonts":
      freqed = pd.infer_freq(train.index)
      if freqed=="MS":
        freq= "M"
      else:
        freq= freqed
      estimator = DeepAREstimator(freq=freq, prediction_length=forecast_len, trainer=Trainer(epochs=6,ctx='gpu')) #use_feat_dynamic_real=True
      if GPU:
        models_dict[m] = estimator.train(training_data=gluonts_dataframe(train))
      else:
        models_dict[m] = estimator.train(training_data=gluonts_dataframe(train))
    if m=="NBEATS":

      if GPU:
        device = torch.device('cuda')
      else:
        device = torch.device('cpu')

      if os.path.isfile(CHECKPOINT_NAME):
          os.remove(CHECKPOINT_NAME)
      stepped = 35
      batch_size = 10
      if in_sample:
        x_train, y_train, x_test, y_test, net, norm_constant = nbeats_dataframe(full_df, forecast_len, in_sample=True,device=device)
        optimiser = optim.Adam(net.parameters())
        data = data_generator(x_train, y_train, batch_size)
        #test_losses = []
        for r in range(stepped):

            train_100_grad_steps(data, device, net, optimiser) #test_losses
        models_dict[m] = {}
        models_dict[m]["model"] = net
        models_dict[m]["x_test"] = x_test
        models_dict[m]["y_test"] = y_test
        models_dict[m]["constant"] = norm_constant

      else: # if out_sample train is df

        x_train, y_train,net, norm_constant= nbeats_dataframe(full_df, forecast_len, in_sample=False,device=device)

        batch_size = 10  # greater than 4 for viz
        optimiser = optim.Adam(net.parameters())
        data = data_generator(x_train, y_train, batch_size)
        stepped = 5
        #test_losses = []
        for r in range(stepped):
            # _, forecast = net(torch.tensor(x_train, dtype=torch.float)) ### Not Used
            # if GPU:
            #   p = forecast.detach().numpy()                               ### Not Used
            # else:
            #   p = forecast.detach().numpy()                               ### Not Used
            train_100_grad_steps(data, device, net, optimiser) #test_losses
        models_dict[m] = {}
        models_dict[m]["model"] = net
        models_dict[m]["tuple"] = (x_train, y_train,net, norm_constant)

    # if m=="TBA":
    #   bat = TBATS(use_arma_errors=False,use_box_cox=True)
    #   models_dict[m] = bat.fit(train)
    if m=="TATS":
      bat = TBATS(seasonal_periods=list(get_unique_N(season_list(train), 1)),use_arma_errors=False,use_trend=True)
      models_dict[m] = bat.fit(train)
    if m=="TBAT":
      bat = TBATS(use_arma_errors=False,use_box_cox=True,
                  use_trend=True)
      models_dict[m] = bat.fit(train)
    if m=="TBATS1":
      bat = TBATS(seasonal_periods=[seasons],use_arma_errors=False,use_box_cox=True,
                  use_trend=True)
      models_dict[m] = bat.fit(train)
    if m=="TBATP1":
      bat = TBATS(seasonal_periods=[periods],use_arma_errors=False,use_box_cox=True,
                  use_trend=True)
      models_dict[m] = bat.fit(train)
    if m=="TBATS2":
      bat = TBATS(seasonal_periods=list(get_unique_N(season_list(train), 2)),use_arma_errors=False,use_box_cox=True,
                  use_trend=True)
      models_dict[m] = bat.fit(train)

    # if m=="ProphetGluonts":
    #   freqed = pd.infer_freq(train.index)
    #   if freqed=="MS":
    #     freq= "M"
    #   else:
    #     freq= freqed
    #   models_dict["ProphetGluonts"] = ProphetPredictor(freq=freq, prediction_length=forecast_len) #use_feat_dynamic_real=True
    #   models_dict["ProphetGluonts"] = list(models_dict["ProphetGluonts"])


  return models_dict, seasons

def forecast_models(models_dict, forecast_len, freq, df, in_sample=True, GPU=False): # test here means any df
  global fb
  forecast_dict = {}
  for name, model in models_dict.items():
    if in_sample:
      print("Model {} is being used to forcast in sample".format(name))
    else:
      print("Model {} is being used to forcast out of sample".format(name))
    if name=="ARIMA":
      forecast_dict[name] = model.predict(forecast_len)
    if name=="Prophet":
      future = model.make_future_dataframe(periods=forecast_len,freq=freq)
      future_pred = model.predict(future)
      forecast_dict[name] = original_dataframe(future_pred,freq)[-forecast_len:]
      fb = original_dataframe(future_pred,freq)[-forecast_len:]

    if name=="HWAAS":
      forecast_dict[name] = model.forecast(forecast_len)
      #hw = model.forecast(forecast_len)
    if name in ["HWAMS","HWAS"]:
      forecast_dict[name] = model.forecast(forecast_len)
    # if name=="HOLT":
    #   forecast_dict[name] = model.forecast(forecast_len)
    if name=="PYAF":
      forecast_dict[name] = model["Target_Forecast"][-forecast_len:].values

    if name=="Gluonts":
      if freq=="MS":
        freq= "M"
      if in_sample:
        for df_entry, forecast in zip(gluonts_dataframe(df), model.predict(gluonts_dataframe(df))):
          forecast_dict[name] = forecast.samples.mean(axis=0)
      else:
        future = ListDataset([{"target": df[-forecast_len:], "start": df.index[-1] + df.index.to_series().diff().min()}],freq=freq)
        #future = ListDataset([{"target": [df[-1]]*forecast_len, "start": df.index[-1] + relativedelta(months=1)}],freq=freq)

        for df_entry, forecast in zip(future, model.predict(future)): #next(predictor.predict(future))
          forecast_dict[name] = forecast.samples.mean(axis=0) # .quantile(0.5)

    if name=="NBEATS":
      if in_sample:
        net = model["model"]
        x_test = model["x_test"]
        y_test = model["y_test"]
        norm_constant =  model["constant"]
        net.eval()
        _, forecast = net(torch.tensor(x_test, dtype=torch.float))
        if GPU:
          p = forecast.cpu().detach().numpy()
        else:
          p = forecast.detach().numpy()
        forecast_dict[name] = p[-1]*norm_constant
      else:
        net = model["model"]
        net.eval()
        x_train, y_train, net, norm_constant= model["tuple"]
        _, forecast = net(torch.tensor(x_train, dtype=torch.float))
        if GPU:
          p = forecast.cpu().detach().numpy()
        else:
          p = forecast.detach().numpy()
        forecast_dict[name] = p[-1]*norm_constant
    if name in ["TBA","TATS","TBAT","TBATS1","TBATS2","TBATS3","TBATP1"]:
            # Forecast 14 steps ahead
       forecast_dict[name] = model.forecast(forecast_len)


  return forecast_dict

def forecast_frame(test, forecast_dict):
  insample = test.to_frame()
  for name, forecast in forecast_dict.items():
    insample[name] = forecast
  return insample

def forecast_frame_insample(forecast_dict,test):
    insample = test.to_frame()
    for name, forecast in forecast_dict.items():
      insample[name] = forecast
    return insample

def forecast_frame_outsample(forecast_dict,df,forecast_len,index):
    ra = -1
    for name, forecast in forecast_dict.items():
      ra += 1
      if ra==0:
        outsample = pd.DataFrame(data=forecast,columns=[name],index=index)
        outsample[name] = forecast
      else:
        outsample[name] = forecast
    return outsample


def insample_performance(test, forecast_dict,dict=False):
  forecasts = forecast_frame(test, forecast_dict)
  dict_perf = {}
  for col, _ in forecasts.iteritems():
    dict_perf[col] = {}
    dict_perf[col]["rmse"] = rmse(forecasts["Target"], forecasts[col])
    dict_perf[col]["mse"] = dict_perf[col]["rmse"]**2
    dict_perf[col]["mean"] = forecasts[col].mean()
  if dict:
    return dict_perf
  else:
    return pd.DataFrame.from_dict(dict_perf)


#### CLASS
#==================================================================================

###### CLASS CREATION
# frozen=True
@dataclass()
class AutomatedModel():
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """

    df: pd.Series
    #model_list: list = ["ARIMA","HOLT"]
    model_list: list
    season: str = "infer_from_data"
    forecast_len: int = 20
    GPU : bool = torch.cuda.is_available()

    def train_insample(self):
      dataframe, freq = parse_data(self.df)
      train, test = train_test_split(dataframe, train_proportion=0.75)
      forecast_len = len(test)
      models, seasonal = train_models(train, models= self.model_list,forecast_len= forecast_len,full_df=dataframe,seasonality=self.season,in_sample=True,freq=freq, GPU=self.GPU )
      self.seasonality = seasonal

      return models, freq, test

    def train_outsample(self):
      dataframe, freq = parse_data(self.df)
      models, _ = train_models(dataframe["Target"], models= self.model_list,forecast_len = self.forecast_len,full_df=dataframe, seasonality=self.season,in_sample=False,freq=freq, GPU=self.GPU)
      return models, freq, dataframe["Target"]

    def forecast_insample(self):
      models_dict, freq, test = self.train_insample()
      forecast_len = test.shape[0]
      forecast_dict = forecast_models(models_dict, forecast_len, freq,test, in_sample=True, GPU=self.GPU)
      forecast_frame = forecast_frame_insample(forecast_dict,test)
      self.models_dict_in = models_dict

      preformance = insample_performance(test, forecast_frame)
      print("Successfully finished in sample forecast")

      return forecast_frame, preformance

    def forecast_outsample(self):
      models_dict, freq, dataframe  = self.train_outsample()
      self.models_dict_out = models_dict
      self.freq = freq
      forecast_dict = forecast_models(models_dict, self.forecast_len, freq,dataframe,in_sample=False, GPU=self.GPU)
      index = pd.date_range(dataframe.index[-1], periods=self.forecast_len+1, freq=freq)[1:]
      forecast_frame = forecast_frame_outsample(forecast_dict,self.df,self.forecast_len, index)

      print("Successfully finished out of sample forecast")
      return forecast_frame

    def ensemble(self, forecast_in, forecast_out):
      season = self.seasonality
      # if season==None:
      #   pass ValueError("Please first train a model using forecast_insample()")

      print("Building LightGBM Ensemble from TS data (ensemble_lgb)")

      ensemble_lgb_in, ensemble_lgb_out = ensemble_lightgbm(forecast_in,forecast_out, self.freq)

      print("Building LightGBM Ensemble from PCA reduced TSFresh Features (ensemble_ts). This can take a long time.")

      ensemble_ts_in, ensemble_ts_out = ensemble_tsfresh(forecast_in, forecast_out, season, self.freq)

      print("Building Standard First Level Ensemble")
      df_ensemble_in, df_ensemble_out = ensemble_pure(forecast_in, forecast_out)
      middle_out = middle(ensemble_lgb_out,ensemble_ts_out, df_ensemble_out )
      middle_in = middle(ensemble_lgb_in, ensemble_ts_in, df_ensemble_in )

      print("Building Final Multi-level Ensemble")
      middle_in, _ = ensemble_first(middle_in,forecast_in)
      all_ensemble_in, all_ensemble_out, all_performance = ensemble_doubled(middle_in,middle_out,forecast_in, forecast_out)

      return all_ensemble_in, all_ensemble_out, all_performance.T.sort_values("rmse")
