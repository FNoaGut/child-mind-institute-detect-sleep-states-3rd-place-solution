'''
This script will create target_encoding features for onsets and wakeups
'''

import numpy as np
import pandas as pd
import pickle
import polars as pl
import matplotlib.pyplot as plt
from metric import *


# Read solution
events = pl.read_csv(f'./data/raw_data/train_events.csv')
events = (events.with_columns([
        pl.col('timestamp').str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias('timestamp_local'),
    ]))
dict_shape = {0:1,-30:1}
SHAPES = []
for add,sc in dict_shape.items():
    tmp = events.with_columns([
        pl.lit(add).alias('add')
    ]).with_columns([
        (pl.col("timestamp_local") + pl.duration(seconds="add")).alias("timestamp_local"),
    ]).with_columns([
        pl.lit(sc).cast(pl.Float32).alias('score')
    ])
    SHAPES.append(tmp)
SHAPES = pl.concat(SHAPES)
# Get time features

events = SHAPES.with_columns([
pl.col('timestamp_local').dt.weekday().alias('weekday'),
    pl.col('timestamp_local').dt.hour().alias('hour'),
pl.col('timestamp_local').dt.minute().alias('minute'),
    pl.col('timestamp_local').dt.second().alias('second'),
])

# Create fold column
with open('folds_new.pickle', 'rb') as handle:
    folds_div = pickle.load(handle)
events = events.with_columns([
    pl.lit(0).alias('fold')
])
for i in range(5):
    events = events.with_columns([
        pl.when(pl.col('series_id').is_in(folds_div[i])).then(i).otherwise(pl.col('fold')).alias('fold')
    ])

########### CREATE HOUR-HALFMINUTE INFO
if True:
    ALL_ = []

    events = events.with_columns([
        pl.col('minute').cast(pl.Int32)
    ]).with_columns([
        pl.when(pl.col('second')>=30).then(pl.col('hour')*120 + pl.col('minute')*2 + 1).
        when(pl.col('second')<30).then(pl.col('hour')*120 +pl.col('minute')*2).otherwise(np.nan).alias('hminute'),
    ])

    ALL = []
    for FOLD in range(5):
        hour_info = (
        events.filter(pl.col('fold') != FOLD).filter(pl.col('step').is_not_null()).groupby(
            ['event', 'hminute']).agg(
            pl.sum('score').alias('eventos')
        )).select(['event', 'hminute', 'eventos'])

        hour_info = hour_info.with_columns([
            pl.lit(FOLD).alias('fold')
        ])
        ALL.append(hour_info)


    ALL = pl.concat(ALL)



    ALL = ALL.pivot(values="eventos", index=["hminute","fold"], columns="event", aggregate_function="sum").with_columns([
            pl.col('onset').fill_null(0),
        pl.col('wakeup').fill_null(0)
        ])
    ALL = ALL.rename({'onset':'hmin_onset','wakeup':'hmin_wakeup'})

    max_onset = ALL['hmin_onset'].max()
    max_wakeup = ALL['hmin_wakeup'].max()
    ALL = ALL.with_columns([
            (pl.col('hmin_onset')/max_onset).alias('hmin_onset'),
            (pl.col('hmin_wakeup')/max_wakeup).alias('hmin_wakeup')
        ]).with_columns([
            pl.col('hmin_onset').cast(pl.Float32),
            pl.col('hmin_wakeup').cast(pl.Float32)
        ])

    # Create directories
    import os
    if not os.path.exists(f"./static_feats"):
        os.makedirs(f"./static_feats")

    ALL.write_csv(f'static_feats/minute_info.csv')
