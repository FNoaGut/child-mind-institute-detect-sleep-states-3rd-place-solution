'''
This script reads the raw data and saves data prepared for insertion into a model
'''
#################
## LIBRARIES
#################
import gc
import pandas as pd
import polars as pl
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,RobustScaler
from tqdm import tqdm
import pickle
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("-m", "--mode", default='test')
parser.add_argument("-n", "--name", default='basic', type=str)
parser.add_argument("-calcs", "--calculations", default=0, type=int)
parser.add_argument("-dw", "--downsample", default=0, type=int)
parser.add_argument("-cl", "--clean", default=0, type=int)
parser.add_argument("-inv", "--invert_series", default=0, type=int)
parser_args, _ = parser.parse_known_args(sys.argv)

#################
## PARAMETERS
#################
NAME_DATASET = parser_args.name
MODE = parser_args.mode# 'train' or 'test' # When train we make different augmentations, when test we dont make augs because we use test for local validation
DOWNSAMPLING = parser_args.downsample
ADD_INVERT_SERIES = bool(parser_args.invert_series!=0) # Create invert series on training
ADD_TARGETS_ONLY = bool(parser_args.clean!=0)
FOLD_NAME ='folds_new'

PURE_CALCULATIONS = bool(parser_args.calculations!=0) # This parameter allows to skip som calculations that are always the same (just put to True first time)
DEBUG = False # When debug True we just preprocess 1 series_id, to check everything works

INIT_HOUR = 17 # All days will be initialized from this hour
SAVE_PREPROCESS = False # save a csv with basic preprocess features (for debugging)
REMOVE_NIGHTS_WITH_NOISE = False # Put to True if you want to remove all noisy parts
TH = 0.5 # Percentage of the night that has to be noise for being removed
REMOVE_LAST_REPETITIVE_DAYS = False # Remove last days if they are noise
REMOVE_LAST_NIGHTS_WITHOUT_LABELS = True # Remove non labeled data at the end of series

fe = f"features_{NAME_DATASET}_{MODE}" # Name for saving data (we have to put this name on the yaml)

print(f'{MODE} -> {NAME_DATASET} -> {DOWNSAMPLING} -> {PURE_CALCULATIONS} -> {ADD_INVERT_SERIES} -> {ADD_TARGETS_ONLY}')

# Create directories
if not os.path.exists(f"../output/fe/fe{fe}"):
    os.makedirs(f"../output/fe/fe{fe}")
    os.makedirs(f"../output/fe/fe{fe}/save")

# BASIC PARAMETERS
target_cols = ['onset','wakeup']
seq_len = int(17280/DOWNSAMPLING)
shift = int(17280/(DOWNSAMPLING))
offset = int(0)


# Put here features that you want to apply normalization
cols_f = []

# Put here features that you want to save
num_cols_f = ['step_first','hour','minute','noise_removal',
              'enmo_std_norm','anglez_std_norm','anglez_std_norm_orig',
              'anglez_equal1','anglez_equal2','fe__check_c','porc_step','group_len_c_norm',
              'hmin_onset','hmin_wakeup']

def specific_process(df):
    '''
    Agregations here will be automatically normalize with robustscaler by series_id
    '''
    old_columns = df.columns

    df = df.with_columns([
        pl.col('anglez_std_norm').mean().over(['st_hour','st_step']).alias('mean_anglez_std_norm')
    ])
    new_columns = sorted(list(np.setdiff1d(df.columns, old_columns)))
    return df, new_columns


####################
## READ STATIC INFO
####################


### Read series
####################
if PURE_CALCULATIONS:

    if DEBUG:
        series = (pl.read_parquet('../data/raw_data/train_series.parquet').
                  filter(pl.col('series_id').is_in(['a3e59c2ce3f6'])))
    else:
        series = pl.read_parquet('../data/raw_data/train_series.parquet')

    series = (series.with_columns([
        pl.col('step').cast(pl.Int64),
        pl.col('anglez').alias('anglez_orig'),
        pl.col('anglez').abs().alias('anglez')
    ]))


    print('Read series DONE')

    #### CREATE NOISE FEATURE

    series = series.with_columns([
        pl.col('anglez').rolling_std(window_size=21,min_periods=21,center=True).
        rolling_quantile(window_size=180,min_periods=180,quantile=0.1,center=True).shift(-180).over('series_id').alias('fe__check_c')
    ])

    series = series.with_columns([
        pl.col('fe__check_c')/pl.col('fe__check_c').max().over('series_id')
    ]).with_columns([
        pl.col('fe__check_c').clip(0.1,1).alias('fe__check_c')
    ])


    series = series.with_columns([
        (((pl.col('fe__check_c')-pl.col('fe__check_c').shift(1)).abs()>0).cumsum()).over('series_id').fill_null(0).cast(pl.Int32).alias('group_c')
    ])

    series = series.with_columns([
        (pl.col('step').count().over(['group_c','series_id'])).cast(pl.Int32).alias('group_len_c')
    ])


    series = series.with_columns([
        pl.when((pl.col('fe__check_c')==0.1) & (pl.col('group_len_c')>360)).
        then(1).when(pl.col('fe__check_c')>0.1).then(0).otherwise(0).cast(pl.Int64).alias('fe__check_c')
    ]).with_columns([
        pl.col('fe__check_c').shift(120).over('series_id').alias('fe__check_c')
    ])


    series = series.with_columns([
        pl.col('fe__check_c').rolling_max(window_size=241,min_periods=241,center=True)
    ]).with_columns([
        (((pl.col('fe__check_c')-pl.col('fe__check_c').shift(1)).abs()>0.1).cumsum()).over('series_id').fill_null(0).cast(pl.Int32).alias('group_c')
    ]).with_columns([
        pl.col('fe__check_c').cast(pl.Float32).fill_null(0).fill_nan(0)
    ])

    series = series.with_columns([
        (pl.count('fe__check_c').over(['group_c','series_id'])).cast(pl.Int32).alias('group_len_c')
    ])

    series = series.with_columns([
        pl.col('fe__check_c').cumcount().over(['group_c','series_id']).cast(pl.Int32).alias('group_step')
    ])


    series = series.with_columns([
        pl.when(pl.col('fe__check_c')==1).then(pl.col('group_step')/8640).when(pl.col('fe__check_c')==0).then(-pl.col('group_step')/8640).otherwise(0).alias('porc_step'),
        (pl.col('group_len_c')/17280).alias('group_len_c_norm')])


    ### Normalize enmo and anglez globally
    ##########################################
    ALL = []
    for ser_id in series['series_id'].unique():
        tmp = series.filter(pl.col('series_id')==ser_id).with_columns([
            ((pl.col('anglez') - pl.col('anglez').mean())/pl.col('anglez').std()).alias('anglez'),
            ((pl.col('enmo') - pl.col('enmo').mean()) / pl.col('enmo').std()).alias('enmo'),
            ((pl.col('anglez_orig') - pl.col('anglez_orig').mean()) / pl.col('anglez_orig').std()).alias('anglez_orig')
        ])
        ALL.append(tmp)
    series = pl.concat(ALL)

    print('Normalize enmo and anglez globally DONE')

    ### Calculating first instance at 17:00 locally (step_first)
    ##########################################

    series = (
        series.with_columns([
            pl.col('timestamp').str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias('timestamp_local')]).
        with_columns([
            pl.col('timestamp_local').dt.hour().cast(pl.Int32).alias('hour'),
            pl.col('timestamp_local').dt.minute().cast(pl.Int32).alias('minute')
        ]))
    series = series.sort(by=['series_id','step'],descending=[True,False])

    # Initialazing all days at 19 hours
    df_sf = (series[['series_id', 'step', 'hour','minute']].filter((pl.col('hour') == INIT_HOUR) & (pl.col('minute') == 0)).
          sort(by=['series_id', 'step'], descending=[True, False]).group_by(['series_id']).agg(pl.col('step').first()))
    df_sf.columns = ['series_id','step_first']
    series = series.join(df_sf,on=['series_id'],how='left')


    print('Calculating first instance at 17:00 locally (step_first) DONE')

    ### Calculating features related with the first step
    ##############################################################
    # rel_step: relative step respect to first step
    # night: first complete day is night 0
    # st_hour: calculated hour from steps
    # st_step: relative step inside an st_hour
    series = series.with_columns([
        (pl.col('step')-pl.col('step_first')).alias('rel_step')
    ]).with_columns([
        (pl.col('rel_step')//17280).alias('night')
    ]).with_columns([
        pl.when(pl.col('rel_step')>=0).then((pl.col('rel_step') - pl.col('night')*17280)//(60*12)).
        otherwise((pl.col('rel_step')+17280)//(60*12)).alias('st_hour')
    ]).with_columns([
        pl.when(pl.col('rel_step') >= 0).then((pl.col('rel_step') - pl.col('st_hour') * 60 * 12 - pl.col('night') * 17280)).
        otherwise((pl.col('rel_step') + 17280 - pl.col('st_hour') * 60 * 12)).alias('st_step')
    ])

    print('Calculating features related with the first step DONE')

    ### Detecting noise through same values in same st_hour and st_step for same series
    ##############################################################
    # repeat: number of times that the value appears in the series
    # noise: binary feature is 1 when repeat>1
    # group_len: number of consecutive values with noise
    # noise_removal: binary feat is 1 when is noise==1 and group_len>360

    temp_df = (series.group_by(['series_id','st_hour','st_step','anglez']).
               agg(pl.col('step').count().alias('repeat')))

    series = series.join(temp_df, on=['series_id','st_hour','st_step','anglez'], how='left')

    series = series.with_columns([
        pl.when(pl.col('repeat')>1).then(1).otherwise(0).cast(pl.Int32).alias('noise'),
    ])

    series = series.with_columns([
        (((pl.col('noise')-pl.col('noise').shift(1)).abs()>0).cumsum()).over('series_id').fill_null(0).cast(pl.Int32).alias('group')
    ])
    series = series.with_columns([
        pl.col('group').max().over('series_id').cast(pl.Int32).alias('group_max')
    ])
    if REMOVE_LAST_REPETITIVE_DAYS and MODE=='train':
        series = series.filter(~((pl.col('noise')==1) & (pl.col('group')==pl.col('group_max'))))
        print('Remove last repetitive days DONE')

    series = series.with_columns([
        (pl.col('step').count().over(['group','series_id'])).cast(pl.Int32).alias('group_len')
    ])
    series = series.with_columns([
        pl.when((pl.col('noise')==1) & (pl.col('group_len')>360)).then(1).otherwise(0).cast(pl.Int32).alias('noise_removal')
    ])

    print('Detecting noise DONE')
    series.write_parquet('series_tr.parquet')
else:
    if DEBUG:
        series = pl.read_parquet('series_tr.parquet').filter(pl.col('series_id').is_in(['78569a801a38']))
    else:
        series = pl.read_parquet('series_tr.parquet')

### Downsampling series
##########################
series = series.with_columns([
    (((pl.col('step'))//DOWNSAMPLING)).alias('step')
])

# Calculations
series = series.sort(by=['series_id','step'],descending=[True,False])
series = (series.group_by(['series_id','step']).agg([
    pl.col('timestamp').first(),
    pl.col('night').first(),
    pl.col('st_hour').max(),
    pl.col('st_step').max(),
    pl.col('anglez').mean().alias('anglez'),
    pl.col('enmo').mean().alias('enmo'),
    pl.col('anglez').std().alias('anglez_std_norm'),
    pl.col('anglez_orig').std().alias('anglez_std_norm_orig'),
    pl.col('enmo').std().alias('enmo_std_norm'),
    pl.col('fe__check_c').max().alias('fe__check_c'),
    pl.col('porc_step').mean().alias('porc_step'),
    pl.col('group_len_c_norm').mean().alias('group_len_c_norm'),
    (pl.col('noise').sum()/DOWNSAMPLING).alias('noise'),
    (pl.col('noise_removal').sum()/DOWNSAMPLING).alias('noise_removal'),
]))


series = series.sort(by=['series_id','step'],descending=[True,False])

print('Downsampling series DONE')

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
series = series.sort(by=['series_id','step'],descending=[True,False])
print('Get basic time features DONE')

### Get basic feature enginering
#################################
# anglez_equal1: +1 day lag features for detecting repetitive patterns
# anglez_equal2: -1 day lag features for detecting repetitive patterns
series = series.with_columns([
    pl.when((pl.col('anglez') == pl.col('anglez').shift(int(17280/DOWNSAMPLING)))).then(1).otherwise(0).alias('anglez_equal1'),
    pl.when((pl.col('anglez') == pl.col('anglez').shift(int(-17280/DOWNSAMPLING)))).then(1).otherwise(0).alias('anglez_equal2'),
    pl.col('anglez_std_norm').rolling_quantile(window_size=15, quantile=0.5, center=False).over(['series_id']).alias(
        'std_q50_left'),
    pl.col('anglez_std_norm').reverse().rolling_quantile(window_size=15, quantile=0.5, center=False).reverse().over(
        ['series_id']).alias(
        'std_q50_right'),
    pl.lit(0).alias('invert')
])
series = series.sort(by=['series_id','step'],descending=[True,False])

print('Get basic feature enginering DONE')
### Calculating first instance at 17:00 locally (step_first)
##########################################
df_sf = (series[['series_id', 'step', 'hour','minute']].filter((pl.col('hour') == INIT_HOUR) & (pl.col('minute') == 0)).
      sort(by=['series_id', 'step'], descending=[True, False]).group_by(['series_id']).agg(pl.col('step').first()))
df_sf.columns = ['series_id','step_first']
series = series.join(df_sf,on=['series_id'],how='left')
series = series.sort(by='step',descending=False)

print('Get first step DONE')
### ADDING TARGET_ENCODING
##########################################
# hminute: hour*120 + minute * 2 (hour_halfminute)
series = series.with_columns([
    (pl.col('minute')*2).alias('hminute'),
]).with_columns([
    pl.when(pl.col('second')<30).then(0).when(pl.col('second')>=30).then(1).otherwise(0).alias('half')
]).with_columns([
    pl.when(pl.col('half')==1).then(pl.col('hminute')+1).otherwise(pl.col('hminute')).alias('hminute')
])
series = series.with_columns([
    (pl.col('hour')*120 + pl.col('hminute')).alias('hminute')
])

### Read fold division
##########################################
with open(f'../{FOLD_NAME}.pickle', 'rb') as handle:
    folds_div = pickle.load(handle)
series = series.with_columns([pl.lit(0).alias('fold')])
for i in range(5):
    series = series.with_columns([
        pl.when(pl.col('series_id').is_in(folds_div[i])).then(i).otherwise(pl.col('fold')).alias('fold')
    ])


# Frequency of onsets and wakeups at each hour-halfminute
minute_info = pl.read_csv(f'../static_feats/minute_info.csv').with_columns([
pl.col('hminute').cast(pl.Int32),
pl.col('fold').cast(pl.Int32),
pl.col('hmin_onset').cast(pl.Float32),
pl.col('hmin_wakeup').cast(pl.Float32)
])
series = series.join(minute_info,on=['hminute','fold'],how='left')

print('Add target encoding DONE')

### ADD TARGETS
##########################################
gc.collect()
val=True
if val:
    # Read labels and downsample
    events = pl.read_csv('/home/fnoa/Escritorio/sleep/data/raw_data/train_events.csv').with_columns([
    ((pl.col('step'))//DOWNSAMPLING).alias('step')
        ])

    # Calculate last night on labels
    max_night = events.group_by(['series_id']).agg(pl.col('night').max().alias('max_night'))


    onsets = events.filter(pl.col('event')=='onset').with_columns(pl.lit(1).alias('onset')).with_columns([
        pl.col('step').cast(pl.Int64)
    ]).select(['step','series_id','onset'])
    wakeups = events.filter(pl.col('event') == 'wakeup').with_columns(pl.lit(1).alias('wakeup')).with_columns([
        pl.col('step').cast(pl.Int64)
    ]).select(['step','series_id','wakeup'])

    series = series.join(onsets,on=['step','series_id'],how='left').with_columns(pl.col('onset').fill_nan(0).fill_null(0))
    series = series.join(wakeups,on=['step','series_id'],how='left').with_columns(pl.col('wakeup').fill_nan(0).fill_null(0))
    series = series.join(max_night, on=['series_id'], how='left')

    if REMOVE_LAST_NIGHTS_WITHOUT_LABELS and MODE=='train':
        series = series.filter(pl.col('step')<=((pl.col('max_night'))*(17280/DOWNSAMPLING)))



    if REMOVE_NIGHTS_WITH_NOISE and MODE=='train':
        series = series.with_columns([
            pl.col('noise_removal').sum().over(['series_id', 'night']).alias('noise_num')
        ])
        series = series.filter(~((pl.col('noise_num')>(17280/DOWNSAMPLING)*TH) & (pl.col('night')>-1)))

    series = series.sort(by=['series_id', 'step'], descending=[True, False])

    if SAVE_PREPROCESS:
        series.write_csv('series_eng.csv')

print('Add targets DONE')

num_array = []
target_array = []
id_list = []
mask_array = []
pred_use_array = []
time_array = []

UNIQUE_IDS = series['series_id'].unique()


def preprocess_id(id, series,cols_f,num_cols_f):
    gc.collect()

    # Filter for the individual Id
    df = series.filter(pl.col('series_id') == id)


    # Get the first step at 17:00
    step_first = df['step_first'][0]

    # Get how much we have to padding aty the left
    offset_first = seq_len - step_first


    if step_first>0: # If first step is not at 17:00

        if step_first > seq_len or seq_len<(17280//DOWNSAMPLING):
            offset_first = 0
            df = df.with_columns([
                pl.col('step').cast(pl.Int64)
            ])
        elif len(df)>seq_len: # When longitude of the entire serie is higher than 1 day
            df_back = df[:(offset_first)][['series_id','step']].with_columns([
                pl.lit(pl.Series([i for i in range((step_first-seq_len),0)])).alias('step')
            ])
            df = df.with_columns([
                pl.col('step').cast(pl.Int64)
            ])
            df = pl.concat([df_back, df], how="diagonal")
        else:
            mult = int(np.ceil(seq_len/len(df)))
            ver = [df for i in range(mult)]
            df_back = pl.concat(ver)
            df_back = df_back[:(offset_first)][['series_id','step']].with_columns([
                pl.lit(pl.Series([i for i in range((step_first-seq_len),0)])).alias('step')
            ])
            df = pl.concat([df_back,df], how="diagonal")
    elif step_first==0:
        offset_first = 0

    # Number of fragments we will create
    batch = 1 + (len(df)-1) // shift



    # Make specific preprocess
    df, new_cols = specific_process(df)

    # Update columns
    cols = cols_f + new_cols
    num_cols = num_cols_f + new_cols

    # Save order of features
    with open(f"../output/fe/fe{fe}/feats_{fe}.pkl", 'wb') as f:
        pickle.dump(num_cols, f)

    # Normalize if we have specific features
    sc = RobustScaler()
    if len(cols)>0:
        df[cols] = sc.fit_transform(df[cols].to_numpy())
    df = df.with_columns(pl.col(num_cols).fill_nan(0).fill_null(0))

    # Feats, targets and time to numpy
    num = df[num_cols].to_numpy()
    target = df[target_cols].fill_nan(0).fill_null(0).with_columns([
        pl.col('onset').cast(pl.Float32),
        pl.col('wakeup').cast(pl.Float32)
    ]).to_numpy()
    time = df["step"].to_numpy()

    # Create initial zero numpys
    id_array_ = np.full([batch],id, dtype='U{}'.format(len(id)))
    num_array_ = np.zeros([batch, seq_len, len(num_cols)], dtype=np.float16)
    target_array_ = np.zeros([batch, seq_len, len(target_cols)], dtype=np.float16)
    time_array_ = np.zeros([batch, seq_len], dtype=int)
    mask_array_ = np.zeros([batch, seq_len], dtype=int)
    pred_use_array_ = np.zeros([batch, seq_len], dtype=int)

    for n, b in enumerate(range(batch)):
        if b == (batch-1):
            num_ = num[b * shift:]
            num_array_[b, :len(num_), :] = num_
            target_ = target[b * shift:]
            target_array_[b, :len(target_), :] = target_
            mask_array_[b, offset: len(target_)] = 1
            pred_use_array_[b, offset:len(target_)] = 1
            time_ = time[b * shift:]
            time_array_[b, :len(time_)] = time_
        elif b == 0:
            num_ = num[offset_first : seq_len]
            num_array_[b, offset_first:, :] = num_
            target_ = target[offset_first:seq_len ]
            target_array_[b, offset_first:, :] = target_
            mask_array_[b, offset_first:seq_len] = 1
            pred_use_array_[b, offset_first:seq_len] = 1
            time_ = time[b:seq_len]
            time_array_[b, :] = time_
        else:
            num_ = num[b * shift:b * shift + seq_len]
            num_array_[b, :, :] = num_
            target_ = target[b * shift:b * shift + seq_len]
            target_array_[b, :, :] = target_
            mask_array_[b, offset:offset + shift] = 1
            pred_use_array_[b, offset:offset + shift] = 1
            time_ = time[b * shift:b * shift + seq_len]
            time_array_[b, :] = time_


    if MODE=='train' and ADD_TARGETS_ONLY:
        cleaned_idx = np.sum(np.sum(target_array_,1),1)>0
        num_array_ = num_array_[cleaned_idx,:,:]
        target_array_ = target_array_[cleaned_idx, :, :]
        mask_array_ = mask_array_[cleaned_idx, :]
        pred_use_array_ = pred_use_array_[cleaned_idx, :]
        time_array_ = time_array_[cleaned_idx, :]
        id_array_ = id_array_[cleaned_idx]

    num_array.append(num_array_)
    target_array.append(target_array_)
    mask_array.append(mask_array_)
    pred_use_array.append(pred_use_array_)
    time_array.append(time_array_)
    id_list.append(id_array_)

    if ADD_INVERT_SERIES and MODE=='train':
        # ADD INVERT SERIES
        num_array_flip = np.flip(num_array_, axis=1).copy()
        target_array_flip = np.flip(np.flip(target_array_, axis=1), axis=2)
        mask_array_flip = np.flip(mask_array_, axis=1)
        pred_use_array_flip = np.flip(pred_use_array_, axis=1)

        num_array.append(num_array_flip)
        target_array.append(target_array_flip)
        mask_array.append(mask_array_flip)
        pred_use_array.append(pred_use_array_flip)
        time_array.append(time_array_)
        id_list.append(id_array_)


for id in tqdm(UNIQUE_IDS):

    preprocess_id(id, series,cols_f,num_cols_f)



num_array = np.concatenate(num_array, axis=0)
target_array = np.concatenate(target_array, axis=0)
id_list = np.concatenate(id_list, axis=0)
mask_array = np.concatenate(mask_array, axis=0)
pred_use_array = np.concatenate(pred_use_array, axis=0)
time_array = np.concatenate(time_array, axis=0)



np.save(f"../output/fe/fe{fe}/fe{fe}_num_array.npy", num_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_target_array.npy", target_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_id_array.npy", id_list)
np.save(f"../output/fe/fe{fe}/fe{fe}_mask_array.npy", mask_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_time_array.npy", time_array)
np.save(f"../output/fe/fe{fe}/fe{fe}_pred_use_array.npy", pred_use_array)



