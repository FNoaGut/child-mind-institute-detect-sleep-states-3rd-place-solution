'''
This code allow us to replicate the same inference we are doing in kaggle notebook but locally
'''

# ============================
# LIBRARIES
# ============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple
import polars as pl
import yaml
from types import SimpleNamespace
import collections
import os
import importlib
import pickle
import sys
import glob
import random
# ============================
# BASIC_DIRECTORIES
# ============================
if os.getcwd().find('fnoa') != -1:
    MAIN_DIR = '.'
    MAIN_DIR_DATA = f'{MAIN_DIR}/data/raw_data'
    DIR_MODELS = f'{MAIN_DIR}/output'
    DIR_YAMLS = f'{MAIN_DIR}/yaml'
    DIR_STATIC_FEATS = f'{MAIN_DIR}/static_feats'
    sys.path.append("models")
else:
    MAIN_DIR_DATA = '/kaggle/input/child-mind-institute-detect-sleep-states'
    SLEEP_DATASET =  '/kaggle/input/sleep-ensemble'
    DIR_MODELS = SLEEP_DATASET
    DIR_YAMLS = f'{SLEEP_DATASET}/yaml'
    DIR_STATIC_FEATS = f'{SLEEP_DATASET}/static_feats'
    sys.path.append("/kaggle/input/sleep-ensemble/models")
    sys.path.append("/kaggle/input/sleep-ensemble")

# Import basic functions fnoa
mdl = importlib.import_module('util_functions_normal')
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
globals().update({k: getattr(mdl, k) for k in names})

################## PREPROCESSING SPECIFICATIONS
# Put here features that you want to apply normalization
cols_f = []

# Put here features that you want to save
num_cols_f = ['step_first','hour','minute','noise_removal',
              'enmo_std_norm','anglez_std_norm','anglez_std_norm_orig',
              'anglez','enmo',
              'anglez_equal1','anglez_equal2',
              'std_q50_left','std_q50_right','fe__check_c','porc_step','group_len_c_norm',
              'hmin_onset','hmin_wakeup']
def specific_process(df):
    '''
    Agregations here will be automatically normalize with robustscaler by series_id
    '''
    old_columns = df.columns
    df = df.with_columns([
        pl.col('anglez').diff().abs().rolling_sum(window_size=60).rolling_max(60).alias('anglez_diff_last_halfhour60'),
        pl.col('anglez').diff().abs().rolling_sum(window_size=30).rolling_max(30).alias('anglez_diff_last_halfhour30'),
        pl.col('anglez').diff().abs().rolling_sum(window_size=15).rolling_max(15).alias('anglez_diff_last_halfhour15'),
        pl.col('anglez').diff().abs().rolling_sum(window_size=6).rolling_max(6).alias('anglez_diff_last_halfhour6'),
    ])
    new_columns = sorted(list(np.setdiff1d(df.columns, old_columns)))
    return df, new_columns

def make_prediction_fnoa(SER_ID, MODELS, MODEL_FOLD, EVAL, models_read, names_read):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INIT_HOUR = 17
    seq_len = int(2880 * 1)
    shift = int(2880 * 1)
    offset = int(0)
    target_cols = ['onset', 'wakeup']
    DOWNSAMPLING = 6
    bs = 16

    # Filter the id
    if EVAL:
        series = pl.scan_parquet(f'{MAIN_DIR_DATA}/train_series.parquet').filter(pl.col('series_id') == SER_ID)
    else:
        series = pl.scan_parquet(f'{MAIN_DIR_DATA}/test_series.parquet').filter(pl.col('series_id') == SER_ID)

    series = series.with_columns([pl.col('step').cast(pl.Int64),
                                  pl.col('anglez').alias('anglez_orig'),
                                  pl.col('anglez').abs().alias('anglez')])
    # print('Read series DONE')

    #### CREATE GOD FEATURE
    series = series.with_columns([
        pl.col('anglez').rolling_std(window_size=21, min_periods=21, center=True).
        rolling_quantile(window_size=180, min_periods=180, quantile=0.1, center=True).shift(-180).over(
            'series_id').alias('fe__check_c')
    ])

    series = (series.with_columns([pl.col('fe__check_c') / pl.col('fe__check_c').max().over('series_id')]).
              with_columns([pl.col('fe__check_c').clip(0.1, 1).alias('fe__check_c')]))

    series = series.with_columns([
        (((pl.col('fe__check_c') - pl.col('fe__check_c').shift(1)).abs() > 0).cumsum()).over('series_id').fill_null(
            0).cast(pl.Int32).alias('group_c')
    ])

    series = series.with_columns([
        (pl.col('step').count().over(['group_c', 'series_id'])).cast(pl.Int32).alias('group_len_c')
    ])

    series = series.with_columns([
        pl.when((pl.col('fe__check_c') == 0.1) & (pl.col('group_len_c') > 360)).
        then(1).when(pl.col('fe__check_c') > 0.1).then(0).otherwise(0).cast(pl.Int64).alias('fe__check_c')
    ]).with_columns([
        pl.col('fe__check_c').shift(120).over('series_id').alias('fe__check_c')
    ])

    series = series.with_columns([
        pl.col('fe__check_c').rolling_max(window_size=241, min_periods=241, center=True)
    ]).with_columns([
        (((pl.col('fe__check_c') - pl.col('fe__check_c').shift(1)).abs() > 0.1).cumsum()).over('series_id').fill_null(
            0).cast(pl.Int32).alias('group_c')
    ]).with_columns([
        pl.col('fe__check_c').cast(pl.Float32).fill_null(0).fill_nan(0)
    ])

    series = series.with_columns([
        (pl.count('fe__check_c').over(['group_c', 'series_id'])).cast(pl.Int32).alias('group_len_c')
    ])
    series = series.with_columns([
        pl.col('fe__check_c').cumcount().over(['group_c', 'series_id']).cast(pl.Int32).alias('group_step')
    ])
    series = series.with_columns([
        pl.when(pl.col('fe__check_c') == 1).then(pl.col('group_step') / 8640).when(pl.col('fe__check_c') == 0).then(
            -pl.col('group_step') / 8640).otherwise(0).alias('porc_step'),
        (pl.col('group_len_c') / 17280).alias('group_len_c_norm')])

    ### Normalize enmo and anglez globally
    ##########################################

    for col in ['anglez', 'enmo', 'anglez_orig']:
        series = series.with_columns([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        ])

    ### Calculating first instance at 17:00 locally (step_first)
    ##########################################
    series = (
        series.with_columns([
            pl.col('timestamp').str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias('timestamp_local')]).
        with_columns([
            pl.col('timestamp_local').dt.hour().cast(pl.Int32).alias('hour'),
            pl.col('timestamp_local').dt.minute().cast(pl.Int32).alias('minute')
        ]))
    series = series.sort(by=['series_id', 'step'], descending=[True, False])

    # Initialazing all days at 19 hours
    df_sf = (
        series.select(['series_id', 'step', 'hour', 'minute']).filter(
            (pl.col('hour') == INIT_HOUR) & (pl.col('minute') == 0)).
        sort(by=['series_id', 'step'], descending=[True, False]).group_by(['series_id']).agg(
            pl.col('step').first()).rename({'step': 'step_first'}))
    series = series.join(df_sf, on=['series_id'], how='left')

    # print('Calculating first instance at 17:00 locally (step_first) DONE')
    ### Calculating features related with the first step
    ##############################################################
    # rel_step: relative step respect to first step
    # night: first complete day is night 0
    # st_hour: calculated hour from steps
    # st_step: relative step inside an st_hour
    series = series.with_columns([
        (pl.col('step') - pl.col('step_first')).alias('rel_step')
    ]).with_columns([
        (pl.col('rel_step') // 17280).alias('night')
    ]).with_columns([
        pl.when(pl.col('rel_step') >= 0).then((pl.col('rel_step') - pl.col('night') * 17280) // (60 * 12)).
        otherwise((pl.col('rel_step') + 17280) // (60 * 12)).alias('st_hour')
    ]).with_columns([
        pl.when(pl.col('rel_step') >= 0).then(
            (pl.col('rel_step') - pl.col('st_hour') * 60 * 12 - pl.col('night') * 17280)).
        otherwise((pl.col('rel_step') + 17280 - pl.col('st_hour') * 60 * 12)).alias('st_step')
    ])

    # print('Calculating features related with the first step DONE')
    ### Detecting noise through same values in same st_hour and st_step for same series
    ##############################################################
    # repeat: number of times that the value appears in the series
    # noise: binary feature is 1 when repeat>1
    # group_len: number of consecutive values with noise
    # noise_removal: binary feat is 1 when is noise==1 and group_len>360

    temp_df = (series.group_by(['series_id', 'st_hour', 'st_step', 'anglez']).
               agg(pl.col('step').count().alias('repeat')))

    series = series.join(temp_df, on=['series_id', 'st_hour', 'st_step', 'anglez'], how='left')

    series = series.with_columns([
        pl.when(pl.col('repeat') > 1).then(1).otherwise(0).cast(pl.Int32).alias('noise'),
    ])

    series = series.with_columns([
        (((pl.col('noise') - pl.col('noise').shift(1)).abs() > 0).cumsum()).over('series_id').fill_null(0).cast(
            pl.Int32).alias('group')
    ])
    series = series.with_columns([
        pl.col('group').max().over('series_id').cast(pl.Int32).alias('group_max')
    ])

    series = series.with_columns([
        (pl.col('step').count().over(['group', 'series_id'])).cast(pl.Int32).alias('group_len')
    ])
    series = series.with_columns([
        pl.when((pl.col('noise') == 1) & (pl.col('group_len') > 360)).then(1).otherwise(0).cast(pl.Int32).alias(
            'noise_removal')
    ])

    # print('Detecting noise DONE')
    ### Downsampling series
    ##########################
    if DOWNSAMPLING > 1:
        series = series.with_columns([
            (((pl.col('step')) // DOWNSAMPLING)).alias('step')
        ])

        # Calculations
        series = series.sort(by=['series_id', 'step'], descending=[True, False])
        series = (series.group_by(['series_id', 'step']).agg([
            pl.col('timestamp').first(),
            pl.col('night').first(),
            pl.col('anglez').mean().alias('anglez'),
            pl.col('enmo').mean().alias('enmo'),
            pl.col('anglez').std().alias('anglez_std_norm'),
            pl.col('anglez_orig').std().alias('anglez_std_norm_orig'),
            pl.col('enmo').std().alias('enmo_std_norm'),
            pl.col('fe__check_c').max().alias('fe__check_c'),
            pl.col('porc_step').mean().alias('porc_step'),
            pl.col('group_len_c_norm').mean().alias('group_len_c_norm'),
            (pl.col('noise').sum() / DOWNSAMPLING).alias('noise'),
            (pl.col('noise_removal').sum() / DOWNSAMPLING).alias('noise_removal'),
        ]))

    else:
        series = series.with_columns([
            pl.col('anglez').alias('anglez_std_norm'),
            pl.col('enmo').alias('enmo_std_norm'),
        ])

    series = series.sort(by=['series_id', 'step'], descending=[True, False])

    # print('Downsampling series DONE')

    ### Get basic time features
    #################################
    series = (series.with_columns([
        pl.col('timestamp').str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias('timestamp_local')
    ]).with_columns([
        pl.col('timestamp_local').dt.minute().cast(pl.Int32).alias('minute'),
        pl.col('timestamp_local').dt.second().cast(pl.Int32).alias('second'),
        pl.col('timestamp_local').dt.hour().cast(pl.Int32).alias('hour'),
        pl.col('timestamp_local').dt.day().cast(pl.Int32).alias('day'),
        pl.col('timestamp_local').dt.year().cast(pl.Int32).alias('year'),
        pl.col('timestamp_local').dt.weekday().cast(pl.Int32).alias('weekday'),
    ]))
    series = series.sort(by=['series_id', 'step'], descending=[True, False])
    # print('Get basic time features DONE')

    ### Get basic feature enginering
    #################################
    # anglez_equal1: +1 day lag features for detecting repetitive patterns
    # anglez_equal2: -1 day lag features for detecting repetitive patterns
    series = series.with_columns([
        pl.when((pl.col('anglez') == pl.col('anglez').shift(int(17280 / DOWNSAMPLING)))).then(1).otherwise(0).alias(
            'anglez_equal1'),
        pl.when((pl.col('anglez') == pl.col('anglez').shift(int(-17280 / DOWNSAMPLING)))).then(1).otherwise(0).alias(
            'anglez_equal2'),
        pl.col('anglez_std_norm').rolling_quantile(window_size=15, quantile=0.5, center=False).over(
            ['series_id']).alias(
            'std_q50_left'),
        pl.col('anglez_std_norm').reverse().rolling_quantile(window_size=15, quantile=0.5, center=False).reverse().over(
            ['series_id']).alias(
            'std_q50_right'),
        pl.lit(0).alias('invert')
    ])
    series = series.sort(by=['series_id', 'step'], descending=[True, False])

    # print('Get basic feature enginering DONE')
    ### Calculating first instance at 17:00 locally (step_first)
    ##########################################
    if series.select(pl.count()).collect()['count'][0] > 40:
        df_sf = (
            series.select(['series_id', 'step', 'hour', 'minute']).filter(
                (pl.col('hour') == INIT_HOUR) & (pl.col('minute') == 0)).
            sort(by=['series_id', 'step'], descending=[True, False]).group_by(['series_id']).agg(
                pl.col('step').first()).rename({'step': 'step_first'}))
        series = series.join(df_sf, on=['series_id'], how='left')
        series = series.sort(by='step', descending=False)
    else:
        series = series.with_columns([pl.lit(0).alias('step_first')])

    # print('Get first step DONE')
    ### ADDING TARGET_ENCODING
    ##########################################
    # hminute: hour*120 + minute * 2 (hour_halfminute)
    series = series.with_columns([
        (pl.col('minute') * 2).alias('hminute'),
    ]).with_columns([
        pl.when(pl.col('second') < 30).then(0).when(pl.col('second') >= 30).then(1).otherwise(0).alias('half')
    ]).with_columns([
        pl.when(pl.col('half') == 1).then(pl.col('hminute') + 1).otherwise(pl.col('hminute')).alias('hminute')
    ])
    series = series.with_columns([
        (pl.col('hour') * 120 + pl.col('hminute')).alias('hminute')
    ])

    # Frequency of onsets and wakeups at each hour-halfminute
    minute_info = pl.scan_csv('./static_feats/minute_info.csv').with_columns([
        pl.col('hminute').cast(pl.Int32),
        pl.col('fold').cast(pl.Int32),
        pl.col('hmin_onset').cast(pl.Float32),
        pl.col('hmin_wakeup').cast(pl.Float32)
    ])
    minute_info = minute_info.group_by(['hminute']).agg([
        pl.col('hmin_onset').mean().alias('hmin_onset'),
        pl.col('hmin_wakeup').mean().alias('hmin_wakeup')
    ])
    series = series.join(minute_info, on=['hminute'], how='left')

    # print('Add target encoding DONE')

    series = series.with_columns(pl.lit(0).alias('onset'))
    series = series.with_columns(pl.lit(0).alias('wakeup'))

    all_features, id_info = preprocess_id(SER_ID, series, cols_f, num_cols_f, seq_len=seq_len, shift=shift,
                                          offset=offset, target_cols=target_cols, specific_process=specific_process)

    if models_read is None:
        ALL_MODELS = []
        ALL_NAMES = []
        for name in MODELS:
            # print(all_features)
            _models, _names = load_models(DIR_YAMLS, DIR_MODELS, name, fold=MODEL_FOLD, cfg=None,
                                          all_features=all_features)
            ALL_MODELS.append(_models)
            ALL_NAMES.append(_names)
    else:
        ALL_MODELS = models_read
        ALL_NAMES = names_read

    num_array = np.concatenate(id_info[0], axis=0)
    target_array = np.concatenate(id_info[1], axis=0)
    mask_array = np.concatenate(id_info[2], axis=0)
    pred_use_array = np.concatenate(id_info[3], axis=0)
    time_array = np.concatenate(id_info[4], axis=0)
    id_list = np.concatenate(id_info[5], axis=0)
    df_id = pd.DataFrame()
    df_id["series_id"] = id_list

    val_ = CustomDataset(num_array,
                         mask_array,
                         train=True,
                         y=target_array)

    val_loader = DataLoader(dataset=val_,
                            batch_size=bs,
                            shuffle=False, num_workers=1)

    val_preds = np.ndarray((0, seq_len, 2))
    # tk0 = tqdm(val_loader, total=len(val_loader))

    save_preds = {}
    for idx1, mod in enumerate(ALL_MODELS):
        for idx2, mod_fold in enumerate(mod):
            save_preds.update({f'{idx1}_{idx2}': np.ndarray((0, seq_len, 2))})

    with torch.no_grad():
        # Predicting on validation set
        for d in val_loader:  # tk0:
            input_data_numerical_array = d['input_data_numerical_array'].to(device)
            input_data_mask_array = d['input_data_mask_array'].to(device)
            attention_mask = d['attention_mask'].to(device)

            for idx1, mod in enumerate(ALL_MODELS):
                for idx2, mod_fold in enumerate(mod):
                    tmp_preds1 = mod_fold(input_data_numerical_array,
                                          input_data_mask_array,
                                          attention_mask).sigmoid().detach().cpu().numpy()[:, :, :2]
                    save_preds[f'{idx1}_{idx2}'] = np.concatenate([save_preds[f'{idx1}_{idx2}'], tmp_preds1], axis=0)

        save_preds_q = save_preds
        num = 0
        for i, j in save_preds.items():
            if num == 0:
                preds_mean = save_preds[i] / len(save_preds)
            else:
                preds_mean += save_preds[i] / len(save_preds)
            num += 1

            save_preds_q[i][:, :, 0] = np.where(save_preds[i][:, :, 0] > 0.1, save_preds[i][:, :, 0], 0)
            save_preds_q[i][:, :, 1] = np.where(save_preds[i][:, :, 1] > 0.1, save_preds[i][:, :, 1], 0)

            preds_mean += save_preds_q[i]

        gc.collect()
        val_preds = np.concatenate([val_preds, preds_mean], axis=0)


    pred_valid_index = (mask_array == 1) & (pred_use_array == 1)

    # Steps
    sol_time = time_array[pred_valid_index].ravel()

    # Ids
    val_id_array_rep = np.repeat(id_list[:, np.newaxis], num_array.shape[1], 1)
    sol_id = val_id_array_rep[pred_valid_index].ravel()

    # Predictions
    sol_onset = val_preds[pred_valid_index][:, 0].ravel()
    sol_wakeup = val_preds[pred_valid_index][:, 1].ravel()

    # Features
    sol_step_first = num_array[pred_valid_index][:, 0].ravel()
    sol_hour = num_array[pred_valid_index][:, 1].ravel()
    sol_minute = num_array[pred_valid_index][:, 2].ravel()

    onset = {'series_id': sol_id, 'step': sol_time, 'event': 'onset', 'score': sol_onset,
             'sf': sol_step_first, 'hour': sol_hour, 'minute': sol_minute}
    onset = pd.DataFrame(onset)

    wakeup = {'series_id': sol_id, 'step': sol_time, 'event': 'wakeup', 'score': sol_wakeup,
              'sf': sol_step_first, 'hour': sol_hour, 'minute': sol_minute}
    wakeup = pd.DataFrame(wakeup)

    submission = pd.concat([onset, wakeup], axis=0)

    # Create submission from raw predictions
    submission = submission.sort_values(by='score', ascending=False)
    submission['step'] = (submission['step'] * DOWNSAMPLING).astype(int)
    submission['sf'] = (submission['sf'] * DOWNSAMPLING).astype(int)
    submission['num_night'] = (1 + ((submission['step'] - submission['sf']) // 17280)).astype(int)
    submission['hour'] = (submission['hour']).astype(int)
    submission['minute'] = (submission['minute']).astype(int)

    return submission, ALL_MODELS, ALL_NAMES


def postprocess_from_god(df):
    df = (df.pivot(index=['series_id', 'step', 'sf', 'hour', 'minute', 'num_night'], columns="event",
                   values="score").sort_index(level=[1, 0]))
    df = df.reset_index()
    # df = pd.DataFrame(observations, columns=["video_id", "frame_id", "pred_challenge", "pred_throwin", "pred_play"])

    val_df = df.sort_values(["series_id", "step"]).copy().reset_index(drop=True)

    window_first = [7, 7]
    window_size = [7, 7]
    ignore_width = [11, 11]

    val_df["onset"] = val_df["onset"] ** 1
    val_df["wakeup"] = val_df["wakeup"] ** 1

    train_df = []
    # gdf = df.loc[df['series_id']==video_id]
    for video_id, gdf in val_df.groupby('series_id'):
        for k, event in enumerate(['onset', 'wakeup']):
            # Moving averages are used to smooth out the data.
            prob_arr = gdf[f"{event}"].rolling(window=window_first[k], center=True).mean().fillna(0).rolling(
                window=window_size[k], center=True).mean().fillna(0).values
            gdf['rolling_prob'] = prob_arr
            sort_arr = np.argsort(-prob_arr)
            rank_arr = np.empty_like(sort_arr)
            rank_arr[sort_arr] = np.arange(len(sort_arr))
            # index list for detected action
            idx_list = []
            for i in range(len(prob_arr)):
                this_idx = sort_arr[i]
                if this_idx >= 0:
                    # Add maximam index to index_list
                    idx_list.append(this_idx)
                    for parity in (-1, 1):
                        for j in range(1, ignore_width[k] + 1):
                            ex_idx = this_idx + j * parity
                            if ex_idx >= 0 and ex_idx < len(prob_arr):
                                # Exclude frames near this_idx where the action occurred.
                                sort_arr[rank_arr[ex_idx]] = -1
            this_df = gdf.iloc[idx_list].reset_index(drop=True).reset_index().rename(columns={'index': 'rank'})[
                ['rank', 'series_id', 'step', 'rolling_prob']]
            this_df['event'] = event

            this_df = this_df.loc[this_df['rolling_prob'] > np.quantile(this_df['rolling_prob'], 0.87)].reset_index(
                drop=True)

            train_df.append(this_df)

    train_df = pd.concat(train_df)
    train_df1 = train_df.copy()

    train_df1['step'] = train_df1['step'] - 0.5
    train_df1['time'] = train_df1['step'] / 12
    # train_df['score'] = 0.5 * train_df['rolling_prob'] + 0.5 * (1 / (train_df['rank'] + 1))
    train_df1['score'] = train_df1['rolling_prob']

    return train_df1

if __name__ == '__main__':
    MODELS = ['gru_abs']
    MODEL_FOLD = 0  # Put None for taking models of all folds
    LOCAL_VAL = True  # Put tu False when making submission

    with open('folds_new.pickle', 'rb') as handle:
        folds_div = pickle.load(handle)

    # Read all de Ids
    series = pl.read_parquet(f'{MAIN_DIR_DATA}/test_series.parquet',
                             columns=['series_id'])
    UNIQUE_IDS = folds_div[MODEL_FOLD]#list(series['series_id'].unique())  # ['038441c925bb', '03d92c9f6f8a', '0402a003dae9']#f
    print(UNIQUE_IDS)

    # Start loop making predictions
    ALL_PREDS = []
    models_read = None
    names_read = None
    RAW_PREDS = []
    for ser_id in tqdm(UNIQUE_IDS):
        # Make raw pred
        tmp, models_read, names_read = make_prediction_fnoa(ser_id, MODELS, MODEL_FOLD, EVAL=LOCAL_VAL, models_read=models_read,names_read=names_read)

        RAW_PREDS.append(tmp)
        # Make postprocess
        preds_fnoa = postprocess_from_god(tmp)
        ALL_PREDS.append(preds_fnoa[['series_id', 'step', 'event', 'score']])

    # Concat
    ALL_PREDS = pd.concat(ALL_PREDS)
    RAW_PREDS = pd.concat(RAW_PREDS)

    # Prepare sub format and save
    submission = ALL_PREDS.sort_values(by=['series_id', 'step'], ascending=True).reset_index(drop=True)
    submission['row_id'] = submission.index

    from metric import *
    if True:
        events = pl.read_csv(f'{MAIN_DIR_DATA}/train_events.csv')
        sol = events.filter(pl.col('step').is_not_null()).filter(pl.col('step').is_not_nan())

        actual_map_score = event_detection_ap(sol.filter(pl.col('series_id').is_in(folds_div[MODEL_FOLD])).to_pandas(),
                                              submission,
                                              {"onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
                                               "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]})
        print(actual_map_score)












