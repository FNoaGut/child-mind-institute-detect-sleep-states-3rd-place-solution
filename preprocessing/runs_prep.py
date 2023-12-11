import os
import pandas as pd


codigo = 'preprocess'

# # CREATE TEST
# mode = 'test'
# pure_calculations = 1
# name = 'basic'
# downsample = 6
# clean = 0
# invert_series = 0
# os.system(f'python {codigo}.py --mode {mode} --name {name} --calculations {pure_calculations} --downsample {downsample} --clean {clean} --invert_series {invert_series}')

# CREATE TRAIN
mode = 'train'
pure_calculations = 0
name = 'basic'
downsample = 6
clean = 0
invert_series = 1
os.system(f'python {codigo}.py --mode {mode} --name {name} --calculations {pure_calculations} --downsample {downsample} --clean {clean} --invert_series {invert_series}')

# CREATE TRAIN
mode = 'train'
pure_calculations = 0
name = 'no_inv'
downsample = 6
clean = 0
invert_series = 0
os.system(f'python {codigo}.py --mode {mode} --name {name} --calculations {pure_calculations} --downsample {downsample} --clean {clean} --invert_series {invert_series}')

# CREATE TRAIN
mode = 'train'
pure_calculations = 0
name = 'clean'
downsample = 6
clean = 1
invert_series = 1
os.system(f'python {codigo}.py --mode {mode} --name {name} --calculations {pure_calculations} --downsample {downsample} --clean {clean} --invert_series {invert_series}')