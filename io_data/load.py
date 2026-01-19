from pathlib import Path
from typing import List, Tuple
import json
import ast

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
    # index.html側でpractice/learning/awarenessを分けていないため、ここで分割する
    # 改善案: index.html側でphase情報を付与する
    trimmed_df_practice = task_df.iloc[:PRACTICE_ROWS].reset_index(drop=True)
    trimmed_df_practice["num_session"] = 0
    trimmed_df_practice["num_trial"] = trimmed_df_practice.index // 2
    trimmed_df_learning = task_df.iloc[PRACTICE_ROWS : -ROWS_FOR_AWARENESS].reset_index(drop=True)
    trimmed_df_learning["num_session"] = 1
    trimmed_df_learning["num_trial"] = trimmed_df_learning.index // 2
    trimmed_df_awareness = task_df.iloc[-ROWS_FOR_AWARENESS:].reset_index(drop=True)
    trimmed_df_awareness["num_session"] = 2
    trimmed_df_awareness["num_trial"] = trimmed_df_awareness.index // 2
    return  trimmed_df_practice, trimmed_df_learning, trimmed_df_awareness


# def annotate_sessions(df: pd.DataFrame) -> pd.DataFrame:
#     """Add num_session and num_trial columns as described."""
#     df = df.copy()
#     df["num_session"] = df.index // ROWS_PER_SESSION
#     df["num_trial"] = (df.index % ROWS_PER_SESSION) // 2
#     return df


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
        base["response_angle_css"] = follow.get("response_angle_css")
        base["random_initial_angle"] = follow.get("random_initial_angle")
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
    
    df["bad_response"] = df.apply(lambda row: row["random_initial_angle"] == row["response_angle_css"], axis=1)
    df["chosen_item"] = df.apply(determine_choice, axis=1)
    return df

def exclude_trials(
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
    if "bad_response" not in df.columns:
        raise ValueError("DataFrame lacks 'bad_response' column.")
    df.loc[df["bad_response"] == True, "rt"] = np.nan
    return df

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Full pipeline: load -> filter -> annotate -> concatenate."""
    df = load_data(path)
    trimmed_practice, trimmed_learning, trimmed_awareness = filter_task_rows(df)
    concatenated_practice = concatenate_trials(trimmed_practice)
    chosen_practice = annotate_choices(concatenated_practice)
    filtered_practice = exclude_trials(chosen_practice)
    concatenated_learning = concatenate_trials(trimmed_learning)
    chosen_learning = annotate_choices(concatenated_learning)
    filtered_learning = exclude_trials(chosen_learning)
    concatenated_awareness = concatenate_trials(trimmed_awareness)
    chosen_awareness = annotate_choices(concatenated_awareness)
    filtered_awareness = exclude_trials(chosen_awareness)
    return filtered_practice, filtered_learning, filtered_awareness

def load_all_concatenated(
    data_dir: Path,
    subjects_include: List[str] = None
    ) -> List[Tuple[str, pd.DataFrame]]:
    """Load all csv/json in data_dir and return list of (subject_id, concatenated_df)."""
    datasets_practice = []
    datasets_learning = []
    datasets_awareness = []
    for file_path in sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))):
        subj_id = file_path.stem
        if subjects_include is not None and subj_id not in subjects_include:
            continue
        if subj_id in [
            ## Excluded subjects:
            ## 2025/12/15に集めたデータのうち、以下の被験者は除外する##
            # "666306b0bf2de127943c419f",
            # "667aca76f4fb2f1d50d80c2e",
            # "673757f92aa69c13b7841d90",
            # "673f0e83fbba6c167eebd6f7",
            # "677e4656af6e5525f72fc926",
            # "678f3b13379c83cf1027d2ed",
            # "596634e005f2df00017281ae", # 極端にターゲットを選んでいる回数が多い 49/1
            # # "6743c8da977b0d274dad1fc2", # 極端にターゲットを選んでいる回数が多いその２ 41/3
            # "66534b438dbae7a1d0a36a08", # 28試行においてターゲット/ディストラクターを回答していない
            # "6755b42b20cf26a928acaa05", # ANOVAとロジスティック回帰で有意な結果（ターゲット選択割合の向上傾向）が確認されている被験者 in df_learning
            # "67e03ba35f26a1779f406b6a", # ANOVAとロジスティック回帰で有意な結果（ターゲット選択割合の向上傾向）が確認されている被験者 in df_learning and df_awareness

            # # HMMによりスイッチの回数が多かった被験者群
            # "596634e005f2df00017281ae",
            # "5cfd24ccf8ff8a00017319d0",
            # "667aca76f4fb2f1d50d80c2e",
            # "673a1dcffde7de9c08f6d2e6",
            # "673f0e83fbba6c167eebd6f7",
            # "673f4f8fa5b4a47492e30aea",
            # "6743c8da977b0d274dad1fc2",
            # "678f3b13379c83cf1027d2ed",
            # "67f789ca382e36a759a011af",
            # "66c9b31cbfa4d79905b6414d",
            # "6133a0d1026a4b5c9c5aaa43",
            # "67e03ba35f26a1779f406b6a",



            ## 2026/1/11に集めたデータのうち、以下の被験者は除外する##
            # "5ee7fbc114d0a60f9b076fb6",
            # # "650f65aac58fe4dc08bbe23f",
            # "660d675bbdf59327d9deb4ad", # else(-1)がn=12と多い
            # "67d1d172e049a486152a5ce9", # else(-1)がn=13と多い

            # # "5e92178e8ee4fe54b65b7c39",
            # # "6932b19c5260dda743fca4af",
            # # "602e48dbf732e9962e27fdbd",
            # # "66723b1f7c3cf6961f0868a3",

            # "5671131573f58b0005664333",
            # "602e48dbf732e9962e27fdbd",
            # "616033a44ba802b7e18daaa9",
            # "650f65aac58fe4dc08bbe23f",
            # "65794b62e4bbf95a4f2c9f03",
            # "660d675bbdf59327d9deb4ad",
            # "6614fb6af3c5aa23b962ea2d",
            # "66723b1f7c3cf6961f0868a3",
            # "669533c82c03a4d6320159d3",
            # "67d1d172e049a486152a5ce9",
            # "692e41b6e14a945652e39997",
            # "6932b19c5260dda743fca4af",
            "5e92178e8ee4fe54b65b7c39",
            "6614fb6af3c5aa23b962ea2d",
            "6932b19c5260dda743fca4af",
        ]:
            print(f"Excluding subject {subj_id}")
            continue
        try:
            concat_df_practice, concat_df_learning, concat_df_awareness = load_and_prepare(file_path)
            datasets_practice.append((subj_id, concat_df_practice))
            datasets_learning.append((subj_id, concat_df_learning))
            datasets_awareness.append((subj_id, concat_df_awareness))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
    return datasets_practice, datasets_learning, datasets_awareness


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



def load_hmm_summary(
    summary_path: Path,
    needed_columns: List[str] = ["subject", "frac_exploit", "switch_count", "mean_run_explore", "mean_run_exploit", "state_labels", "A", "B", "pi", "loglik", "states","observations", "mapped_observations"]
    ) -> pd.DataFrame:
    """Load HMM summary CSV and ensure needed columns are present."""
    df = pd.read_csv(summary_path)
    missing = [col for col in needed_columns if col not in df.columns]
    if missing:
        raise ValueError(f"HMM summary missing columns: {missing}")

    matrix_columns = ["A", "B", "pi", "states","observations", "mapped_observations"]
    for col in matrix_columns:
        # 各列の各要素（文字列）にjson.loadsを適用し、結果をNumPy配列に変換
        df[col] = df[col].apply(lambda s: np.array(json.loads(s)))

    df['state_labels'] = df['state_labels'].apply(ast.literal_eval)

    return df

def load_categorized_subjects(
    summary_path: Path,
    needed_columns: List[str] = ["subject", "category"],
    needed_categories: List[str] = ["explore-exploit-cycling"]
    ) -> List[str]:
    """Extract subjects belonging to needed categories from categorized DataFrame."""
    df = load_hmm_summary(summary_path)
    missing = [col for col in needed_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Categorized DataFrame missing columns: {missing}")
    filtered = df[df["category"].isin(needed_categories)]
    return filtered["subject"].tolist()