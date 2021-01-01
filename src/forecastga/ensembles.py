#! /usr/bin/env python
# coding: utf-8
#
"""ForecastGA: Ensembles"""

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
from statsmodels.tools.eval_measures import rmse
from tsfresh.utilities.dataframe_functions import impute, roll_time_series
from tsfresh import extract_features
from tsfresh import select_features
import lightgbm as lgb

from forecastga.helpers.logging import get_logger
from forecastga.helpers.data import constant_feature_detect

_LOG = get_logger(__name__)


def ensemble_performance(forecasts):
    dict_perf = {}
    for col, _ in forecasts.iteritems():
        dict_perf[col] = {}
        dict_perf[col]["rmse"] = rmse(forecasts["Target"], forecasts[col])
        dict_perf[col]["mse"] = dict_perf[col]["rmse"] ** 2
        dict_perf[col]["mean"] = forecasts[col].mean()
    return pd.DataFrame.from_dict(dict_perf)


def time_feature(df, perd):
    if perd in ["MS", "M", "BM", "BMS"]:
        df["month"] = df.index.month
    elif perd in ["BH", "H"]:
        df["hour"] = df.index.hour
    elif perd == "B":
        df["dayofweek"] = df.index.dayofweek
    elif perd == "D":
        df["dayofweek"] = df.index.dayofweek
    elif perd in ["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
        df["week"] = df.index.week
    elif perd in ["Q", "QS", "BQ", "BQS"]:
        df["quarter"] = df.index.quarter
    elif perd in ["T", "min"]:
        df["minute"] = df.index.minute
    elif perd == "S":
        df["second"] = df.index.second
    # elif perd in ["L","ms"]:
    #   periodocity = 1000
    # elif perd in ["U","us"]:
    #   periodocity = 1000
    # elif perd=="N":
    #   periodocity = 1000
    return df


def ensemble_lightgbm(forecast_in, forecast_out, pred):

    forecast_in_copy = forecast_in.copy()

    forecast_in_copy = time_feature(forecast_in_copy, pred)
    forecast_in_copy["mean"] = forecast_in_copy.drop(["Target"], axis=1).mean(axis=1)
    forecast_train, forecast_test = tts(
        forecast_in_copy, train_size=0.5, shuffle=False, stratify=None
    )

    target = "Target"
    d_train = lgb.Dataset(
        forecast_train.drop(columns=[target]), label=forecast_train[target]
    )

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmsle",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbose": 0,
        "num_threads": 16,
    }

    model = lgb.train(params, d_train, 100, verbose_eval=1)

    ensemble_lgb = pd.DataFrame(index=forecast_test.index)

    ensemble_lgb["ensemble_lgb"] = model.predict(forecast_test.drop(columns=[target]))

    ensemble_lgb_out = pd.DataFrame(index=forecast_out.index)

    ensemble_lgb_out["ensemble_lgb"] = model.predict(forecast_out)

    return ensemble_lgb, ensemble_lgb_out


def ensemble_tsfresh(forecast_in, forecast_out, season, perd):
    """
    Create rolled time series for ts feature extraction
    """

    def tsfresh_run(forecast, season, insample=True, forecast_out=None):
        df_roll_prep = forecast.reset_index()
        if insample:
            df_roll_prep = df_roll_prep.drop(["Target", "Date"], axis=1)
            df_roll_prep["id"] = 1
            target = forecast["Target"]
        else:
            df_roll_prep = df_roll_prep.drop(["index"], axis=1)
            df_roll_prep["id"] = 1

        df_roll = roll_time_series(
            df_roll_prep,
            column_id="id",
            column_sort=None,
            column_kind=None,
            rolling_direction=1,
            max_timeshift=season - 1,
        )
        counts = df_roll["id"].value_counts()
        df_roll_cut = df_roll[df_roll["id"].isin(counts[counts >= season].index)]

        # TS feature extraction
        concat_df = pd.DataFrame()

        concat_df = extract_features(
            df_roll_cut.ffill(),
            column_id="id",
            column_sort="sort",
            n_jobs=season,
            show_warnings=False,
            disable_progressbar=True,
        )

        if insample:

            concat_df = concat_df.dropna(axis=1, how="all")
            concat_df.index = (
                target[df_roll_cut["id"].value_counts().index]
                .sort_index()
                .to_frame()
                .index
            )
            concat_df = pd.merge(
                target[df_roll_cut["id"].value_counts().index].sort_index().to_frame(),
                concat_df,
                left_index=True,
                right_index=True,
                how="left",
            )
            concat_df_list = constant_feature_detect(data=concat_df, threshold=0.95)
            concat_df = concat_df.drop(concat_df_list, axis=1)
        else:
            forecast_out.index.name = "Date"
            concat_df.index = forecast_out.index

        concat_df = impute(concat_df)

        return concat_df

    _LOG.info("LightGBM ensemble have been successfully built")

    concat_df_drop_in = tsfresh_run(forecast_in, season, insample=True)

    extracted_n_selected = select_features(
        concat_df_drop_in.drop("Target", axis=1),
        concat_df_drop_in["Target"],
        fdr_level=0.01,
        n_jobs=12,
    )  # fdr is the significance level.

    forecast_out_add = pd.concat(
        (forecast_in.iloc[-season + 1 :, :].drop(["Target"], axis=1), forecast_out),
        axis=0,
    )
    concat_df_drop_out = tsfresh_run(
        forecast_out_add, season, insample=False, forecast_out=forecast_out
    )
    extracted_n_selected_out = concat_df_drop_out[extracted_n_selected.columns]

    # Reduce the dimensions of generated time series features
    pca2 = PCA(n_components=8)
    pca2.fit(extracted_n_selected)
    pca2_results_in = pca2.transform(extracted_n_selected)
    pca2_results_out = pca2.transform(extracted_n_selected_out)

    cols = 0
    for i in range(pca2_results_in.shape[1]):
        cols = cols + 1
        extracted_n_selected["pca_" + str(i)] = pca2_results_in[:, i]
        extracted_n_selected_out["pca_" + str(i)] = pca2_results_out[:, i]

    df = forecast_in.iloc[season - 1 :, :].copy()
    df = time_feature(df, perd)
    df["mean"] = df.drop(["Target"], axis=1).mean(axis=1)

    df_new = pd.concat(
        (df.reset_index(), extracted_n_selected.iloc[:, -cols:].reset_index(drop=True)),
        axis=1,
    )
    df_new = df_new.set_index("Date")
    forecast_train, forecast_test = tts(
        df_new, train_size=0.5, shuffle=False, stratify=None
    )
    target = "Target"
    d_train = lgb.Dataset(
        forecast_train.drop(columns=[target]), label=forecast_train[target]
    )

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmsle",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbose": 0,
        "num_threads": 16,
    }

    model = lgb.train(params, d_train, 100, verbose_eval=1)

    ensemble_ts = pd.DataFrame(index=forecast_test.index)

    ensemble_ts["ensemble_ts"] = model.predict(forecast_test.drop(columns=[target]))

    df_out = forecast_out.copy()
    df_out = time_feature(df_out, perd)
    df_out["mean"] = df_out.mean(axis=1)

    ensemble_ts_out = pd.DataFrame(index=df_out.index)
    ensemble_ts_out["ensemble_ts"] = model.predict(df_out)

    _LOG.info("LightGBM ensemble have been successfully built")

    return ensemble_ts, ensemble_ts_out


def ensemble_pure(forecast_in, forecast_out):
    """
    Pure Emsemble
    """

    df_perf = ensemble_performance(forecast_in).drop("Target", axis=1)

    def run_ensemble(df_perf, forecast):

        many = len(df_perf.iloc[0, :].sort_values())

        # Note these can fail, should see if that many indices actually exists.
        df_ensemble = pd.DataFrame(index=forecast.index)
        if many == 1:
            ValueError("You need more than one model to ensemble.")
        if many >= 2:
            df_ensemble[
                "_".join(list(df_perf.iloc[0, :].sort_values()[:2].index.values))
            ] = forecast[list(df_perf.iloc[0, :].sort_values()[:2].index)].mean(axis=1)
        if many >= 3:
            df_ensemble[
                "_".join(list(df_perf.iloc[0, :].sort_values()[:3].index.values))
            ] = forecast[list(df_perf.iloc[0, :].sort_values()[:3].index)].mean(axis=1)
        if many >= 4:
            df_ensemble[
                "_".join(list(df_perf.iloc[0, :].sort_values()[:4].index.values))
            ] = forecast[list(df_perf.iloc[0, :].sort_values()[:4].index)].mean(axis=1)

        return df_ensemble

    df_ensemble_in = run_ensemble(df_perf, forecast_in)
    df_ensemble_out = run_ensemble(df_perf, forecast_out)

    return df_ensemble_in, df_ensemble_out


def middle(ensemble_lgb, ensemble_ts, pure_ensemble):
    first_merge = pd.merge(
        ensemble_ts, ensemble_lgb, left_index=True, right_index=True, how="left"
    )
    second_merge = pd.merge(
        first_merge, pure_ensemble, left_index=True, right_index=True, how="left"
    )
    return second_merge


def ensemble_first(middle_in, forecast_in):
    third_merge = pd.merge(
        middle_in,
        forecast_in[["Target"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    return third_merge, ensemble_performance(third_merge).drop("Target", axis=1)


def ensemble_doubled(middle_in, middle_out, forecast_in, forecast_out):

    third_merge_in = pd.merge(
        middle_in.drop(["Target"], axis=1),
        forecast_in,
        left_index=True,
        right_index=True,
        how="left",
    )
    third_merge_out = pd.merge(
        middle_out, forecast_out, left_index=True, right_index=True, how="left"
    )

    # Double Ensemble
    df_perf = ensemble_performance(third_merge_in).drop("Target", axis=1)

    def inner_ensemble(df_perf, third_merge):
        df_ensemble = pd.DataFrame(index=third_merge.index)
        # Note these can fail, should see if that many indices actually exists.

        many = len(df_perf.iloc[0, :].sort_values())

        if many == 1:
            ValueError("You need more than one model to ensemble.")
        if many >= 2:
            df_ensemble[
                "__X__".join(list(df_perf.iloc[0, :].sort_values()[:2].index.values))
            ] = third_merge[list(df_perf.iloc[0, :].sort_values()[:2].index)].mean(
                axis=1
            )
        if many >= 3:
            df_ensemble[
                "__X__".join(list(df_perf.iloc[0, :].sort_values()[:3].index.values))
            ] = third_merge[list(df_perf.iloc[0, :].sort_values()[:3].index)].mean(
                axis=1
            )
        if many >= 5:
            df_ensemble[
                "__X__".join(list(df_perf.iloc[0, :].sort_values()[:5].index.values))
            ] = third_merge[list(df_perf.iloc[0, :].sort_values()[:5].index)].mean(
                axis=1
            )
        if many >= 7:
            df_ensemble[
                "__X__".join(list(df_perf.iloc[0, :].sort_values()[:7].index.values))
            ] = third_merge[list(df_perf.iloc[0, :].sort_values()[:7].index)].mean(
                axis=1
            )
        return df_ensemble

    df_ensembled_in = inner_ensemble(df_perf, third_merge_in)
    df_ensembled_out = inner_ensemble(df_perf, third_merge_out)

    last_merge_in = pd.merge(
        third_merge_in, df_ensembled_in, left_index=True, right_index=True, how="left"
    )  # .drop(["month","mean"],axis=1)
    last_merge_out = pd.merge(
        third_merge_out, df_ensembled_out, left_index=True, right_index=True, how="left"
    )

    df_perf_last = ensemble_performance(last_merge_in).drop("Target", axis=1)

    return last_merge_in, last_merge_out, df_perf_last
