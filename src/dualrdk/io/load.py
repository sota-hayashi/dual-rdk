from pathlib import Path
from typing import Dict, List, Tuple
import json
import ast

import pandas as pd
import numpy as np

from dualrdk.config import PRACTICE_ROWS, ROWS_PER_SESSION, ROWS_FOR_AWARENESS, EXCLUDED_SUBJECTS
from dualrdk.features.behavior import label_if_ooz


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
        base["session_rotation"] = follow.get("rotation_angle")
        base["reward_points"] = follow.get("reward_points")
        base["target_direction"] = follow.get("target_direction")
        # distractor_direction is only present in the base (stimulus) row, not the follow (response) row
        base["angular_error_target"] = follow.get("angular_error_target")
        base["angular_error_distractor"] = follow.get("angular_error_distractor")
        base["rt"] = follow.get("rt")
        base["response_angle_rdk"] = follow.get("response_angle_rdk")
        base["response_angle_css"] = follow.get("response_angle_css")
        base["random_initial_angle"] = follow.get("random_initial_angle")
        base["target_group"] = follow.get("target_group")
        combined_rows.append(base)

    combined_df = pd.DataFrame(combined_rows).reset_index(drop=True)
    return combined_df

def annotate_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate each trial with whether the target or distractor was chosen.
       0 if target chosen, 1 if distractor chosen, -1 if undecided.
    """
    df = df.sort_values("num_trial").copy() if "num_trial" in df.columns else df.copy()

    def determine_choice(row):
        if abs(row["angular_error_target"]) < 45.0:
            return 1
        if abs(row["angular_error_distractor"]) < 45.0:
            return 0
        else:
            return -1
    
    def determine_target_color(row):
        if row["target_group"] == "white":
            return 0
        elif row["target_group"] == "black":
            return 1
        else:
            return np.nan
        
    def infer_choice(row):
        if row["chosen_item"] == 1:
            return row["target_group"]  # ターゲット色
        if row["chosen_item"] == 0:
            return "white" if row["target_group"] == "black" else "black"  # 反対色=ディストラクター色
        return np.nan  # -1は除外
    
    df["target_item"] = df.apply(determine_target_color, axis=1)
    df["chosen_item"] = df.apply(determine_choice, axis=1)
    df["chosen_color"] = df.apply(infer_choice, axis=1)
    df["random_initial_angle_reverted"] = df.apply(lambda row: (450 - row["random_initial_angle"] + 360) % 360, axis=1)
    df["bad_response"] = df.apply(lambda row: row["random_initial_angle"] == row["response_angle_css"] and row["chosen_item"] == -1, axis=1)

    df["prev_chosen_color"] = df["chosen_color"].shift(1)
    df["prev_reward_points"] = df["reward_points"].shift(1)

    # 前試行の報酬で win/lose を判定
    df["prev_win"] = (df["prev_reward_points"] > 2).astype('Int64')
    df["prev_lose"] = (df["prev_reward_points"] <= 2).astype('Int64')

    # stay/switch（前試行の chosen_color と比較）
    df["stay"] = (df["chosen_color"] == df["prev_chosen_color"]).astype('Int64')
    df["switch"] = (df["chosen_color"] != df["prev_chosen_color"]).astype('Int64')

    # 前or今が NaN のときは stay/switch も未定義に
    invalid_choice = df["chosen_color"].isna() | df["prev_chosen_color"].isna()
    df.loc[invalid_choice, ["stay", "switch"]] = pd.NA
    
    invalid_reward = df["prev_reward_points"].isna()
    df.loc[invalid_reward, ["prev_win", "prev_lose"]] = pd.NA

    # win-stay / lose-switch
    # boolean型で初期化し、条件を満たす行のみTrueを設定。他はNAのまま。
    df["win_stay"] = pd.Series(dtype='boolean')
    df["lose_switch"] = pd.Series(dtype='boolean')

    valid_win_stay = (df["prev_win"] == 1) & (df["stay"] == 1)
    df.loc[valid_win_stay, "win_stay"] = True

    valid_lose_switch = (df["prev_lose"] == 1) & (df["switch"] == 1)
    df.loc[valid_lose_switch, "lose_switch"] = True

    # 不要になった中間列を最後に削除
    df.drop(columns=["prev_chosen_color", "prev_reward_points"], inplace=True)
    
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
    if "bad_response" not in df.columns:
        raise ValueError("DataFrame lacks 'bad_response' column.")
    df.loc[df["bad_response"] == True, "rt"] = np.nan
    # df.loc[df["rt"] > rt_threshold, "rt"] = np.nan
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
    subjects_include: List[str] = None,
    apply_exclusion_list: bool = True,
    ) -> List[Tuple[str, pd.DataFrame]]:
    """Load all csv/json in data_dir and return list of (subject_id, concatenated_df).

    Parameters
    ----------
    apply_exclusion_list : bool
        False にすると config.EXCLUDED_SUBJECTS を適用せず全参加者を読む。
        除外基準そのものを検証・再計算するとき（tests/test_exclusions.py）に
        使う。通常の解析では True のままにすること。
    """
    datasets_practice = []
    datasets_learning = []
    datasets_awareness = []
    for file_path in sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))):
        subj_id = file_path.stem
        if subjects_include is not None and subj_id not in subjects_include:
            continue
        if apply_exclusion_list and subj_id in EXCLUDED_SUBJECTS:
            # print(f"Excluding subject {subj_id}")
            continue
        try:
            concat_df_practice, concat_df_learning, concat_df_awareness = load_and_prepare(file_path)
            datasets_practice.append((subj_id, concat_df_practice))
            datasets_learning.append((subj_id, concat_df_learning))
            datasets_awareness.append((subj_id, concat_df_awareness))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    datasets_learning, _ = label_if_ooz(datasets_learning)
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

def load_gaussian_hmm_summary(summary_path: Path) -> List[Dict]:
    """
    Gaussian HMM の結果CSVを読み込み、run_gaussian_hmm() と同じ形式の
    List[Dict] として返す。

    Parameters
    ----------
    summary_path : Path
        save_gaussian_hmm_results() で保存したCSVのパス

    Returns
    -------
    List[Dict]
        各要素は以下のキーを持つ辞書:
            participant_id  : str
            n_trials        : int
            viterbi_states  : np.ndarray (n_trials,)
            means           : np.ndarray (2, 1)
            covars          : np.ndarray (2, 1, 1)
            transmat        : np.ndarray (2, 2)
            state_labels    : dict  {0: "engaged", 1: "disengaged"}
            log_likelihood  : float
    """
    df = pd.read_csv(summary_path)

    required = ["participant_id", "n_trials", "viterbi_states", "means", "covars",
                "transmat", "state_labels", "log_likelihood"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gaussian HMM summary missing columns: {missing}")

    results = []
    for _, row in df.iterrows():
        results.append({
            "participant_id": row["participant_id"],
            "n_trials": int(row["n_trials"]),
            "viterbi_states": np.array(json.loads(row["viterbi_states"]), dtype=int),
            "means": np.array(json.loads(row["means"])),
            "covars": np.array(json.loads(row["covars"])),
            "transmat": np.array(json.loads(row["transmat"])),
            "state_labels": json.loads(row["state_labels"]),
            "log_likelihood": float(row["log_likelihood"]),
        })
    return results


def load_categorized_subjects(
    summary_path: Path,
    needed_columns: List[str] = ["subject", "category", "states", "state_labels", "observations"],
    needed_categories: List[str] = ["on-to-on","off-to-on","on-to-off","off-to-off", "on-off-cycling"]
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract subjects and their states belonging to needed categories.
    Returns a list of (subject_id, states_array) tuples.
    """
    df = load_hmm_summary(summary_path)
    missing = [col for col in needed_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Categorized DataFrame missing columns: {missing}")
        
    filtered = df[df["category"].isin(needed_categories)]
    subject_states_list = list(filtered[['subject', 'category', 'states', 'state_labels', 'observations']].to_records(index=False))
    
    return subject_states_list
    