'''
Util functions for inference
'''
# ============================
# library
# ============================
import torch
from torch.utils.data import DataLoader, Dataset
import gc
import yaml
from types import SimpleNamespace
import collections
import importlib
import glob
import polars as pl
from sklearn.preprocessing import RobustScaler

def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg.architecture)


def load_models(MAIN_DIR_YAMLS, MAIN_DIR_MODELS, name, fold=None, cfg=None, all_features=None,keep_last_epoch=None):
    # LOAD MODEL
    if cfg is None:
        cfg = yaml.safe_load(open(f"{MAIN_DIR_YAMLS}/{name}.yaml").read())
        for k, v in cfg.items():
            if type(v) == dict:
                cfg[k] = SimpleNamespace(**v)
        cfg = SimpleNamespace(**cfg)

    if all_features is not None:
        cfg.architecture.all_features = {k: v for v, k in enumerate(all_features)}


    models = []
    if fold is not None:
        PAT = f"{MAIN_DIR_MODELS}/{name}/checkpoint_expand9*.pth"
    else:
        PAT = f"{MAIN_DIR_MODELS}/{name}/checkpoint_expand9*.pth"

    NAMES = []
    for i in glob.glob(PAT):
        if fold is not None:
            if i.find(f'_{fold}_')!=-1:
                print(i)
                NAMES.append(i)
                model = get_model(cfg).to("cuda").eval()

                d = torch.load(i, map_location="cpu")

                model_weights = d["model"]
                model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}

                for k in list(model_weights.keys()):
                    if "aux" in k or "loss_fn" in k:
                        del model_weights[k]

                model.load_state_dict(collections.OrderedDict(model_weights), strict=True)

                del d
                del model_weights
                gc.collect()

                models.append(model)
        else:
            print(i)
            NAMES.append(i)
            model = get_model(cfg).to("cuda").eval()

            d = torch.load(i, map_location="cpu")

            model_weights = d["model"]
            model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}

            for k in list(model_weights.keys()):
                if "aux" in k or "loss_fn" in k:
                    del model_weights[k]

            model.load_state_dict(collections.OrderedDict(model_weights), strict=True)

            del d
            del model_weights
            gc.collect()

            models.append(model)

    return models, NAMES


def preprocess(numerical_array,
               mask_array,
               ):
    attention_mask = mask_array == 0

    return {
        'input_data_numerical_array': numerical_array,
        'input_data_mask_array': mask_array,
        'attention_mask': attention_mask,
    }
class CustomDataset(Dataset):
    def __init__(self, numerical_array,
                 mask_array,
                 train=True, y=None):
        self.numerical_array = numerical_array
        self.mask_array = mask_array
        self.train = train
        self.y = y

    def __len__(self):
        return len(self.numerical_array)

    @staticmethod
    def batch_to_device(batch, device):
        input_data_numerical_array = batch['input_data_numerical_array'].to(device)
        input_data_mask_array = batch['input_data_mask_array'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch["y"].to(device)
        return input_data_numerical_array, input_data_mask_array, attention_mask, target

    def __getitem__(self, item):
        data = preprocess(
            self.numerical_array[item],
            self.mask_array[item],

        )

        # Return the processed data where the lists are converted to `torch.tensor`s
        if self.train:
            return {
                'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'], dtype=torch.float32),
                'input_data_mask_array': torch.tensor(data['input_data_mask_array'], dtype=torch.long),
                'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
                "y": torch.tensor(self.y[item], dtype=torch.float32)
            }
        else:
            return {
                'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'], dtype=torch.float32),
                'input_data_mask_array': torch.tensor(data['input_data_mask_array'], dtype=torch.long),
                'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
            }



def preprocess_id(id, series,cols_f,num_cols_f,
                  seq_len=2880,
                  shift=2880,
                  offset=0,
                  target_cols=['onset','wakeup'],
                  specific_process = None,
                  MODE='test'):

    num_array = []
    target_array = []
    id_list = []
    mask_array = []
    pred_use_array = []
    time_array = []

    gc.collect()

    # Filter for the individual Id
    df = series.filter(pl.col('series_id') == id).collect()

    # Get the first step at 17:00
    step_first = df['step_first'][0]

    # Get how much we have to padding aty the left
    offset_first = seq_len - step_first


    if step_first>0: # If first step is not at 17:00

        if len(df)>seq_len: # When longitude of the entire serie is higher than 1 day
            df_back = df[:(offset_first)][['series_id','step']].with_columns([
                pl.lit(pl.Series([i for i in range((step_first-seq_len),0)])).alias('step')
            ])
            df = df.with_columns([
                pl.col('step').cast(pl.Int64)
            ])
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

    # Normalize if we have specific features
    sc = RobustScaler()
    if len(cols)>0:
        df[cols] = sc.fit_transform(df[cols].to_numpy())
    df = df.with_columns(pl.col(num_cols).fill_nan(0).fill_null(0))

    # Feats, targets and time to numpy
    num = df[num_cols].to_numpy()
    target = df[target_cols].with_columns([
        pl.col('onset').cast(pl.Float32),
        pl.col('wakeup').cast(pl.Float32),
        # pl.col('target_1').cast(pl.Float32),
        # pl.col('target_2').cast(pl.Float32),
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
            target_ = target[offset_first:seq_len]
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



    # cleaned_idx = np.sum(np.sum(target_array_,1),1)>0
    # num_array_ = num_array_[cleaned_idx,:,:]
    # target_array_ = target_array_[cleaned_idx, :, :]
    # mask_array_ = mask_array_[cleaned_idx, :]
    # pred_use_array_ = pred_use_array_[cleaned_idx, :]
    # time_array_ = time_array_[cleaned_idx, :]
    # id_array_ = id_array_[cleaned_idx]

    num_array.append(num_array_)
    target_array.append(target_array_)
    mask_array.append(mask_array_)
    pred_use_array.append(pred_use_array_)
    time_array.append(time_array_)
    id_list.append(id_array_)

    return num_cols, [num_array,target_array,mask_array,pred_use_array,time_array,id_list]

"""Event Detection Average Precision

An average precision metric for event detection in time series and
video.

"""

from bisect import bisect_left
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


# Set some placeholders for global parameters
series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
use_scoring_intervals = False
tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    use_scoring_intervals: bool = False,
) -> float:
    # Validate metric parameters
    assert len(tolerances) > 0, "Events must have defined tolerances."
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), (
        f"Solution column {event_column_name} must contain the same events "
        "as defined in tolerances."
    )
    assert pd.api.types.is_numeric_dtype(
        solution[time_column_name]
    ), f"Solution column {time_column_name} must be of numeric type."

    # Validate submission format
    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(f"Submission must have column '{column_name}'.")

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    # Set these globally to avoid passing around a bunch of arguments
    globals()["series_id_column_name"] = series_id_column_name
    globals()["time_column_name"] = time_column_name
    globals()["event_column_name"] = event_column_name
    globals()["score_column_name"] = score_column_name
    globals()["use_scoring_intervals"] = use_scoring_intervals

    return event_detection_ap(solution, submission, tolerances)


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]] = tolerances,  # type: ignore
) -> float:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        # intervals = (
        #     solution.query("event in ['start', 'end']")
        #     .assign(
        #       interval=lambda x: x.groupby([series_id_column_name, event_column_name]).cumcount()
        #     )
        #     .pivot(
        #         index="interval",
        #         columns=[series_id_column_name, event_column_name],
        #         values=time_column_name,
        #     )
        #     .stack(series_id_column_name)
        #     .swaplevel()
        #     .sort_index()
        #     .loc[:, ["start", "end"]]
        #     .apply(lambda x: pd.Interval(*x, closed="both"), axis=1)
        # )
        pass

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(drop=True)

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        # detections_filtered = []
        # for (det_group, dets), (int_group, ints) in zip(
        #     detections.groupby(series_id_column_name), intervals.groupby(series_id_column_name)
        # ):
        #     assert det_group == int_group
        #     detections_filtered.append(filter_detections(dets, ints))  # noqa: F821
        # detections_filtered = pd.concat(detections_filtered, ignore_index=True)
        pass
    else:
        detections_filtered = detections

    # Create table of event-class x tolerance x series_id values
    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])

    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(match_detections(dets["tolerance"].iloc[0], gts, dets))
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths[event_column_name].unique()
    ap_table = (
        detections_matched.query("event in @event_classes")  # type: ignore
        .groupby([event_column_name, "tolerance"])
        .apply(
            lambda group: average_precision_score(
                group["matched"].to_numpy(),
                group[score_column_name].to_numpy(),
                class_counts[group[event_column_name].iat[0]],
            )
        )
    )
    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby(event_column_name).mean().sum() / len(event_classes)

    return mean_ap, ap_table


def find_nearest_time_idx(times, target_time, excluded_indices, tolerance):
    """Find the index of the nearest time to the target_time
    that is not in excluded_indices."""
    idx = bisect_left(times, target_time)

    best_idx = None
    best_error = float("inf")

    offset_range = min(len(times), tolerance)
    for offset in range(-offset_range, offset_range):  # Check the exact, one before, and one after
        check_idx = idx + offset
        if 0 <= check_idx < len(times) and check_idx not in excluded_indices:
            error = abs(times[check_idx] - target_time)
            if error < best_error:
                best_error = error
                best_idx = check_idx

    return best_idx, best_error


def match_detections(
    tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    detections_sorted = detections.sort_values(score_column_name, ascending=False).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    ground_truths_times = ground_truths.sort_values(time_column_name)[time_column_name].tolist()
    matched_gt_indices: set[int] = set()

    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        det_time = getattr(det, time_column_name)

        best_idx, best_error = find_nearest_time_idx(
            ground_truths_times, det_time, matched_gt_indices, tolerance
        )

        if best_idx is not None and best_error < tolerance:
            is_matched[i] = True
            matched_gt_indices.add(best_idx)

    detections_sorted["matched"] = is_matched
    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []  # type: ignore

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = (
        tps / p
    )  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])