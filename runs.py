import os
import pandas as pd


codigo = 'train'
folds_split = 'new'

# TRAIN gru_abs
name = 'gru_abs'
expand = 9
pot = 1
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')

# TRAIN gru_abs_short
name = 'gru_abs_short'
expand = 9
pot = 1
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')

# TRAIN gru_abs_clean
name = 'gru_abs_clean'
expand = 9
pot = 1
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')

# TRAIN gru_abs_new
name = 'gru_abs_new'
expand = 9
pot = 1
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')


# TRAIN gru_abs_noinv
name = 'gru_abs_noinv'
expand = 9
pot = 1
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')

# TRAIN gru_no_abs
name = 'gru_no_abs'
expand = 9
pot = 2
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')


# TRAIN gru_abs_comb
name = 'gru_abs_comb'
expand = 9
pot = 2
fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Linear --fold_split {folds_split}')


# TRAIN unet_abs
name = 'unet_abs'
expand = 9
pot = 1

fold_num = 0
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Cosine --fold_split {folds_split}')
fold_num = 1
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Cosine --fold_split {folds_split}')
fold_num = 2
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Cosine --fold_split {folds_split}')
fold_num = 3
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Cosine --fold_split {folds_split}')
fold_num = 4
os.system(f'python {codigo}.py --config yaml/{name}.yaml --pot {pot} --expand {expand} --fold {fold_num} --schedule Cosine --fold_split {folds_split}')
