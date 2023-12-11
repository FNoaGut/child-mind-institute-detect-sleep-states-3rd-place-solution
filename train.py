'''
Script for training
'''
import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
import yaml
from types import SimpleNamespace
import os
from metric import *
import pickle
import neptune
import polars as pl

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("models")
sys.path.append("datasets")



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(dataset=train_ds,
                              batch_size=cfg.training.batch_size,
                              shuffle=True,
                              num_workers=cfg.environment.number_of_workers)
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader
def get_val_dataloader(val_ds, cfg):
    val_dataloader = DataLoader(dataset=val_ds,
                            batch_size=cfg.validation.batch_size,
                            shuffle=False,
                            num_workers=cfg.environment.number_of_workers)

    print(f"val: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader
def get_scheduler(cfg, optimizer, total_steps):
    if cfg.training.schedule == "Linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
            num_cycles=2.5
        )
    return scheduler
def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def read_processed_data(fe):
    id_path = f"output/fe/fe{fe}/fe{fe}_id_array.npy"
    numerical_path = f"output/fe/fe{fe}/fe{fe}_num_array.npy"
    target_path = f"output/fe/fe{fe}/fe{fe}_target_array.npy"
    mask_path = f"output/fe/fe{fe}/fe{fe}_mask_array.npy"
    pred_use_path = f"output/fe/fe{fe}/fe{fe}_pred_use_array.npy"
    time_path = f"output/fe/fe{fe}/fe{fe}_time_array.npy"

    id_array = np.load(id_path)
    numerical_array = np.load(numerical_path)
    target_array = np.load(target_path)
    mask_array = np.load(mask_path)
    pred_use_array = np.load(pred_use_path)
    time_array = np.load(time_path)

    with open(f"output/fe/fe{fe}/feats_{fe}.pkl", 'rb') as f:
        all_features = pickle.load(f)

    return all_features, id_array,numerical_array,target_array ,mask_array,pred_use_array,time_array
def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg.architecture)
def load_models(MAIN_DIR_YAMLS, MAIN_DIR_MODELS, name,fold=None, cfg=None, all_features=None):
    # LOAD MODEL
    if cfg is None:
        cfg = yaml.safe_load(open(f"{MAIN_DIR_YAMLS}/{name}.yaml").read())
        for k, v in cfg.items():
            if type(v) == dict:
                cfg[k] = SimpleNamespace(**v)
        cfg = SimpleNamespace(**cfg)

    if all_features is not None:
        cfg.architecture.all_features = {k: v for v, k in enumerate(all_features)}

    model = get_model(cfg)

    d = torch.load(f"{MAIN_DIR_MODELS}/{name}/checkpoint_expand{cfg.expand}_pot{cfg.pot}_{cfg.fold_div}_{fold}_epoch_24.pth", map_location="cpu")

    model_weights = d["model"]
    model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}

    for k in list(model_weights.keys()):
        if "aux" in k or "loss_fn" in k:
            del model_weights[k]

    import collections
    model.load_state_dict(collections.OrderedDict(model_weights), strict=True)

    del d
    del model_weights
    gc.collect()

    return model


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", default='yaml/gru_abs2.yaml', help="config filename")
parser.add_argument("-EXP", "--expand", default=8, type=int)
parser.add_argument("-POT", "--pot", default=1, type=int)
parser.add_argument("-F", "--fold", default=0, type=int)
parser.add_argument("-FS", "--fold_split", default='new', type=str)
parser.add_argument("-chpt", "--checkpoint", default=False, type=bool)
parser.add_argument("-sch", "--schedule", default=False, type=str)
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)

print(f'train_name {cfg.dataset.train_name}')
print(f'valid_name {cfg.dataset.valid_name}')

cfg.dataset.fold = [parser_args.fold]
cfg.fold_div = 'new'
cfg.training.checkpoint = bool(parser_args.checkpoint)
cfg.training.schedule = str(parser_args.schedule)
print(parser_args.expand)
cfg.expand = int(parser_args.expand)
cfg.pot = int(parser_args.pot)
print(cfg)
print(f'Checkpoint {cfg.training.checkpoint}')
print(f'Schedule {cfg.training.schedule}')
print(f'FOLD {cfg.dataset.fold}')
print(f'EXPAND {cfg.expand}')
print(f'POT {cfg.pot}')

DOWNSAMPLING = 6

os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)
os.makedirs(f"predictions/{cfg.experiment_name}", exist_ok=True)
cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset

if __name__ == "__main__":
    device = "cuda:0"
    cfg.device = device

    # SET SEED
    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    set_seed(cfg.environment.seed)

    with open(f'folds_{cfg.fold_div}.pickle', 'rb') as handle:
        folds_div = pickle.load(handle)

    for FOLD in cfg.dataset.fold:

        def expand_target(target_,n,pot):

            if pot==0:
                left = [0]
            elif pot==1:
                left = [0.5, 1.7]
            elif pot == 2:
                left = [2.5]


            right = list(left[::-1])
            pot = 1
            left = list(left) + [np.power(2.5, pot)]
            print(left)

            kernel1 = np.array(np.concatenate((left, [1], right)), dtype=float)
            kernel2 = kernel1
            for i in range(target_.shape[0]):
                target_[i,:, 0] = np.convolve(target_[i,:, 0], kernel1, mode='same')
                target_[i,:, 1] = np.convolve(target_[i,:, 1], kernel2, mode='same')
                target_[i,:, 0][target_[i,:, 0] == 1] = 2.5**(pot)
                target_[i,:, 1][target_[i,:, 1] == 1] = 2.5**(pot)

            target_[:,:,:2] = target_[:,:,:2]/(2.5**(pot))
            return target_

        if FOLD==-1:
            events = pl.read_csv('./data/raw_data/train_events.csv')
            solution = events.filter(pl.col('series_id').is_in(folds_div[0])).filter(
                pl.col('step').is_not_nan()).filter(pl.col('step').is_not_null()).to_pandas()
        else:
            events = pl.read_csv('./data/raw_data/train_events.csv')
            solution = events.filter(pl.col('series_id').is_in(folds_div[FOLD])).filter(
                pl.col('step').is_not_nan()).filter(pl.col('step').is_not_null()).to_pandas()

        # PREPARE TRAINING
        all_features, id_array,numerical_array,target_array,mask_array,pred_use_array,time_array = read_processed_data(cfg.dataset.train_name)
        target_array = expand_target(target_array,n=cfg.expand,pot=cfg.pot)


        if FOLD==-1:
            train_numerical_array = numerical_array
            train_target_array = target_array
            train_mask_array = mask_array
        else:
            train_idx = np.where(~np.isin(id_array, folds_div[FOLD]))

            train_numerical_array = numerical_array[train_idx]
            train_target_array = target_array[train_idx]
            train_mask_array = mask_array[train_idx]

        all_features, id_array, numerical_array, target_array, mask_array, pred_use_array, time_array = read_processed_data(cfg.dataset.valid_name)
        target_array = expand_target(target_array,n=cfg.expand,pot=cfg.pot)
        if FOLD==-1:
            valid_idx = np.where(np.isin(id_array, folds_div[0]))
        else:
            valid_idx = np.where(np.isin(id_array, folds_div[FOLD]))


        val_numerical_array = numerical_array[valid_idx]
        val_target_array = target_array[valid_idx]
        val_mask_array = mask_array[valid_idx]
        val_pred_array = pred_use_array[valid_idx]
        val_time_array = time_array[valid_idx]
        val_id_array = id_array[valid_idx]


        train_dataset = cfg.CustomDataset(train_numerical_array,
                            train_mask_array,
                            train=True,
                            y=train_target_array)

        val_dataset = cfg.CustomDataset(val_numerical_array,
                          val_mask_array,
                          train=True,
                          y=val_target_array)


        cfg.train_dataset = train_dataset
        cfg.architecture.all_features = {k: v for v, k in enumerate(all_features)}

        train_dataloader = get_train_dataloader(train_dataset, cfg)
        val_dataloader = get_val_dataloader(val_dataset, cfg)

        if cfg.training.checkpoint:
            MAIN_DIR_MODELS = './output'
            MAIN_DIR_YAMLS = './yaml'
            model = load_models(MAIN_DIR_YAMLS, MAIN_DIR_MODELS, cfg.experiment_name,FOLD,cfg)
            print(model)
            print('Model read')
        else:
            model = get_model(cfg)
            print(model)

        model.to(device)
        total_steps = len(train_dataset)
        params = model.parameters()

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']#
        differential_layers = cfg.training.differential_learning_rate_layers
        optimizer = torch.optim.AdamW(#.AdamW
            [
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (not any(nd in name for nd in no_decay))
                    ],
                    "lr": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (any(nd in name for nd in no_decay))
                    ],
                    "lr": cfg.training.learning_rate,
                    "weight_decay": 0,
                }
            ],
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )


        scheduler = get_scheduler(cfg, optimizer, total_steps)

        if cfg.environment.mixed_precision:
            scaler = GradScaler()


        from cbloss.loss import FocalLoss
        criterion = FocalLoss(num_classes=cfg.architecture.out_size, gamma=2, alpha=0.25)


        cfg.curr_step = 0
        i = 0
        best_val_loss = np.inf
        optimizer.zero_grad()
        for epoch in range(cfg.training.epochs):
            set_seed(cfg.environment.seed + epoch)

            cfg.curr_epoch = epoch
            print("EPOCH:", epoch)

            progress_bar = tqdm(range(len(train_dataloader)))#
            tr_it = iter(train_dataloader)
            losses = []
            gc.collect()
            model.train()

            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1
                cfg.curr_step += cfg.training.batch_size
                data = next(tr_it)
                input_data_numerical_array, input_data_mask_array, attention_mask, target  = cfg.CustomDataset.batch_to_device(data, device)
                h = None

                if cfg.environment.mixed_precision:
                    with autocast():
                        output = model(input_data_numerical_array,
                                           input_data_mask_array,
                                           attention_mask)
                else:
                    output = model(input_data_numerical_array,
                                       input_data_mask_array,
                                       attention_mask)

                loss = criterion(output[input_data_mask_array == 1], target[input_data_mask_array == 1][:,:cfg.architecture.out_size])

                losses.append(loss.item())

                if cfg.training.grad_accumulation != 1:
                    loss /= cfg.training.grad_accumulation

                if cfg.environment.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.training.grad_accumulation == 0:
                        if cfg.training.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.training.gradient_clip
                            )

                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.training.grad_accumulation == 0:
                        if cfg.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.training.gradient_clip
                            )
                    optimizer.step()
                    optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.curr_step % cfg.training.batch_size == 0:
                    progress_bar.set_description(
                        f"lr: {np.round(optimizer.param_groups[0]['lr'],7)}, loss: {np.mean(losses[-12:]):.6f}"
                    )

            if True:

                val_preds = np.ndarray((0, val_numerical_array.shape[1], cfg.architecture.out_size))
                progress_bar = tqdm(range(len(val_dataloader)))#
                val_it = iter(val_dataloader)

                model.eval()
                preds = []
                probabilities = []
                all_targets = []
                losses_eval = []
                for itr in progress_bar:
                    data = next(val_it)
                    input_data_numerical_array, input_data_mask_array, attention_mask, target = cfg.CustomDataset.batch_to_device(data, device)
                    h = None

                    if cfg.environment.mixed_precision:
                        with autocast():
                            output = model(input_data_numerical_array,
                                               input_data_mask_array,
                                               attention_mask,train=False)
                    else:
                        output = model(input_data_numerical_array,
                                           input_data_mask_array,
                                           attention_mask,train=False)
                        gc.collect()
                        torch.cuda.empty_cache()

                    loss = criterion(output[input_data_mask_array == 1], target[input_data_mask_array == 1][:,:cfg.architecture.out_size])
                    losses_eval.append(loss.item())

                    val_preds = np.concatenate([val_preds, output.sigmoid().detach().cpu().numpy()], axis=0)

                print(f"EVAL loss: {np.mean(losses_eval):.6f}")

                if epoch>(cfg.training.epochs-3):
                    checkpoint = {"model": model.state_dict()}
                    torch.save(checkpoint,
                               f"output/{cfg.experiment_name}/checkpoint_expand{cfg.expand}_pot{cfg.pot}_{cfg.fold_div}_{FOLD}_epoch_{epoch}.pth")

                # Getting validation metric
                if cfg.dataset_class == "gru_dataset" and (epoch==(cfg.training.epochs-1) or (epoch%4)==0) :#
                    pred_valid_index = (val_mask_array == 1) & (val_pred_array == 1)

                    # Steps
                    sol_time = val_time_array[pred_valid_index].ravel()

                    # Ids
                    val_id_array_rep = np.repeat(val_id_array[:, np.newaxis],  val_numerical_array.shape[1], 1)
                    sol_id = val_id_array_rep[pred_valid_index].ravel()

                    # Predictions
                    sol_onset = val_preds[pred_valid_index][:,0].ravel()
                    sol_wakeup = val_preds[pred_valid_index][:,1].ravel()

                    # Features
                    sol_step_first = val_numerical_array[pred_valid_index][:, 0].ravel()
                    sol_hour = val_numerical_array[pred_valid_index][:, 1].ravel()
                    sol_minute = val_numerical_array[pred_valid_index][:, 2].ravel()
                    sol_noise = val_numerical_array[pred_valid_index][:, 3].ravel()

                    onset = {'series_id': sol_id, 'step': sol_time, 'event': 'onset', 'score': sol_onset,
                             'sf': sol_step_first, 'hour': sol_hour, 'minute': sol_minute,'noise':sol_noise}
                    onset = pd.DataFrame(onset)

                    wakeup = {'series_id': sol_id, 'step': sol_time, 'event': 'wakeup', 'score': sol_wakeup,
                              'sf': sol_step_first, 'hour': sol_hour, 'minute': sol_minute,'noise':sol_noise}
                    wakeup = pd.DataFrame(wakeup)

                    submission = pd.concat([onset,wakeup],axis=0)

                    # Create submission from raw predictions
                    submission = submission.sort_values(by='score',ascending=False)
                    submission['step'] = submission['step']*DOWNSAMPLING
                    submission['sf'] = submission['sf'] * DOWNSAMPLING
                    submission['num_night'] = 1 + ((submission['step'] - submission['sf']) // 17280)


                    raw_predictions = submission.copy()

                    def postprocess(df):
                        df = (
                            df.pivot(index=['series_id', 'step', 'sf', 'hour', 'minute', 'num_night'], columns="event",
                                     values="score").sort_index(level=[1, 0]))
                        df = df.reset_index()

                        val_df = df.sort_values(["series_id", "step"]).copy().reset_index(drop=True)

                        window_first = 5
                        window_size = [5, 5]
                        ignore_width = [11, 11]

                        val_df["onset"] = val_df["onset"] ** 1
                        val_df["wakeup"] = val_df["wakeup"] ** 1

                        train_df = []
                        for _, gdf in val_df.groupby('series_id'):
                            for k, event in enumerate(['onset', 'wakeup']):

                                prob_arr = gdf[f"{event}"].rolling(window=window_first, center=True).mean().fillna(
                                    0).rolling(
                                    window=window_size[k], center=True).mean().fillna(0).values
                                gdf['rolling_prob'] = prob_arr
                                sort_arr = np.argsort(-prob_arr)
                                rank_arr = np.empty_like(sort_arr)
                                rank_arr[sort_arr] = np.arange(len(sort_arr))

                                idx_list = []
                                for i in range(len(prob_arr)):
                                    this_idx = sort_arr[i]
                                    if this_idx >= 0:
                                        idx_list.append(this_idx)
                                        for parity in (-1, 1):
                                            for j in range(1, ignore_width[k] + 1):
                                                ex_idx = this_idx + j * parity
                                                if ex_idx >= 0 and ex_idx < len(prob_arr):
                                                    sort_arr[rank_arr[ex_idx]] = -1
                                this_df = gdf.iloc[idx_list].reset_index(drop=True).reset_index().rename(
                                    columns={'index': 'rank'})[
                                    ['rank', 'series_id', 'step', 'rolling_prob']]
                                this_df['event'] = event

                                this_df = this_df.loc[
                                    this_df['rolling_prob'] > np.quantile(this_df['rolling_prob'], 0.85)].reset_index(
                                    drop=True)

                                train_df.append(this_df)

                        train_df = pd.concat(train_df)
                        train_df1 = train_df.copy()

                        train_df1['step'] = train_df1['step'] - 0.5
                        train_df1['score'] = train_df1['rolling_prob']

                        return train_df1

                    submission = postprocess(submission)

                    map_score, AP = event_detection_ap(solution.loc[solution['series_id'].isin(list(submission['series_id'].unique()))],submission,{"onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
                                                            "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]})

                    print("Validation metric", map_score)

                    if epoch == (cfg.training.epochs - 1):
                        raw_predictions.to_csv(f'predictions/{cfg.experiment_name}/sub_all_expand{cfg.expand}_pot{cfg.pot}_{FOLD}.csv', index=False)
