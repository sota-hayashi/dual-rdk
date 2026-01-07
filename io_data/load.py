from pathlib import Path
from typing import List, Tuple
import json

import pandas as pd
import numpy as np

from common.config import PRACTICE_ROWS, ROWS_PER_SESSION, ROWS_FOR_AWARENESS


def load_data(path: Path) -> pd.DataFrame:
    """
    Load CSV or JSON (jsPsych export) into a DataFrame.
    - CSV: same挙動 as before.
    - JSON: expects an array of trial dictionaries (jsPsych.data.get().json()).
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with open(path, "r") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported file type: {suffix}")


def filter_task_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with target_group white/black and drop initial practice block."""
    task_mask = df["target_group"].isin(["white", "black"])
    task_df = df.loc[task_mask].copy()
    trimmed_df = task_df.iloc[PRACTICE_ROWS:].reset_index(drop=True)
    trimmed_df_learning = trimmed_df.iloc[:-ROWS_FOR_AWARENESS].copy()
    trimmed_df_awareness = trimmed_df.iloc[-ROWS_FOR_AWARENESS:].copy()
    return trimmed_df_learning, trimmed_df_awareness


def annotate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add num_session and num_trial columns as described."""
    df = df.copy()
    df["num_session"] = df.index // ROWS_PER_SESSION
    df["num_trial"] = (df.index % ROWS_PER_SESSION) // 2
    return df


def concatenate_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the two rows (RDK + response) that share the same num_session/num_trial.
    - The first row acts as the base.
    - reward_points / rotation_angle from the second row are copied into new columns.
    """
    combined_rows = []
    group_cols = ["num_session", "num_trial"]

    for key, group in df.groupby(group_cols, sort=False):
        group_sorted = group.sort_index()
        if len(group_sorted) != 2:
            raise ValueError(f"Expected 2 rows per trial for {key}, found {len(group_sorted)}")
        base = group_sorted.iloc[0].copy()
        follow = group_sorted.iloc[1]
        base["reward_points"] = follow.get("reward_points")
        base["angular_error_target"] = follow.get("angular_error_target")
        base["angular_error_distractor"] = follow.get("angular_error_distractor")
        base["rt"] = follow.get("rt")
        base["response_angle_rdk"] = follow.get("response_angle_rdk")
        combined_rows.append(base)

    combined_df = pd.DataFrame(combined_rows).reset_index(drop=True)
    return combined_df

def annotate_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate each trial with whether the target or distractor was chosen.
       0 if target chosen, 1 if distractor chosen, -1 if undecided.
    """
    df = df.copy()

    def determine_choice(row):
        if abs(row["angular_error_target"]) < 45.0:
            return 1
        if abs(row["angular_error_distractor"]) < 45.0:
            return 0
        else:
            return -1

    df["chosen_item"] = df.apply(determine_choice, axis=1)
    return df

def exclude_slow_trials(
    df: pd.DataFrame,
    rt_threshold: float = 10000.0
) -> pd.DataFrame:
    """
    データフレームから、RTが rt_threshold ms を超える試行をNaNに置き換える。
    """
    df = df.copy()
    if "rt" not in df.columns:
        raise ValueError("DataFrame lacks 'rt' column.")
    df.loc[df["rt"] > rt_threshold, "rt"] = np.nan
    return df

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Full pipeline: load -> filter -> annotate -> concatenate."""
    df = load_data(path)
    trimmed_learning, trimmed_awareness = filter_task_rows(df)
    annotated_learning = annotate_sessions(trimmed_learning)
    concatenated_learning = concatenate_trials(annotated_learning)
    chosen_learning = annotate_choices(concatenated_learning)
    filtered_learning = exclude_slow_trials(chosen_learning)
    annotated_awareness = annotate_sessions(trimmed_awareness)
    concatenated_awareness = concatenate_trials(annotated_awareness)
    chosen_awareness = annotate_choices(concatenated_awareness)
    filtered_awareness = exclude_slow_trials(chosen_awareness)
    return filtered_learning, filtered_awareness

def load_all_concatenated(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load all csv/json in data_dir and return list of (subject_id, concatenated_df)."""
    datasets_learning = []
    datasets_awareness = []
    for file_path in sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))):
        subj_id = file_path.stem
        if subj_id in [
            "666306b0bf2de127943c419f",
            "667aca76f4fb2f1d50d80c2e",
            "673757f92aa69c13b7841d90",
            "673f0e83fbba6c167eebd6f7",
            "677e4656af6e5525f72fc926",
            "678f3b13379c83cf1027d2ed",
            "596634e005f2df00017281ae", # 極端にターゲットを選んでいる回数が多い 49/1
            # "6743c8da977b0d274dad1fc2", # 極端にターゲットを選んでいる回数が多いその２ 41/3
            "66534b438dbae7a1d0a36a08", # 28試行においてターゲット/ディストラクターを回答していない

            "6755b42b20cf26a928acaa05", # ANOVAとロジスティック回帰で有意な結果（ターゲット選択割合の向上傾向）が確認されている被験者 in df_learning
            "67e03ba35f26a1779f406b6a", # ANOVAとロジスティック回帰で有意な結果（ターゲット選択割合の向上傾向）が確認されている被験者 in df_learning and df_awareness
        ]:
            continue
        try:
            concat_df_learning, concat_df_awareness = load_and_prepare(file_path)
            datasets_learning.append((subj_id, concat_df_learning))
            datasets_awareness.append((subj_id, concat_df_awareness))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
    return datasets_learning, datasets_awareness


def extract_rts_from_online_data(data_dir: Path) -> List[float]:
    """
    指定されたディレクトリ内のすべてのJSONファイルから'rt'を抽出します。
    'rt'が存在し、nullでない試行のみを対象とします。
    """
    all_rts = []
    if not data_dir.is_dir():
        print(f"Error: Directory not found at {data_dir}")
        return all_rts

    for file_path in sorted(data_dir.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                trials = json.load(f)
                for trial in trials:
                    if isinstance(trial, dict) and "rt" in trial:
                        rt_value = trial["rt"]
                        if rt_value is not None:
                            all_rts.append(float(rt_value))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path.name}")
        except Exception as e:
            print(f"Warning: An error occurred while processing {file_path.name}: {e}")

    return all_rts

def combine_subjects(concat_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-subject concatenated data with a subject column."""
    rows = []
    for subj_id, df in concat_list:
        tmp = df.copy()
        tmp["subject"] = subj_id
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)