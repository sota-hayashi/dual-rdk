from typing import List, Tuple

import numpy as np
import pandas as pd

from features.behavior import calculate_rt_moving_mean, calculate_rt_deviance_mean
from io_data.load import exclude_trials

from common.config import TRIALS_PER_SESSION
def compute_out_of_zone_ratio_by_AE(df: pd.DataFrame, n_trial: int = TRIALS_PER_SESSION) -> float:
    """
    各被験者のマインドワンダリング指標（out of the zone）の全試行に対する割合を計算する。

    out of the zone の定義:
      被験者が回答した角度が，target/distractorの45度以内に入っていない試行を注意が途切れた試行とし，
      その総試行数に対する割合．
      つまり，chosen_item == -1を満たす試行の割合を計算する．
    """
    needed = ["rt", "chosen_item"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")

    valid = df.dropna(subset=needed).copy()
    if valid.empty:
        return np.nan

    #前半と後半で分けてMW vs rewardを見ている
    # n_out_of_zone = ((valid["chosen_item"] == -1) & (valid["num_trial"] <= n_trial - 1)).sum()
    n_out_of_zone = ((valid["chosen_item"] == -1)).sum()
    n_total = len(valid)

    return n_out_of_zone / n_total if n_total > 0 else np.nan


def compute_out_of_zone_ratio_of_mean_AE(df: pd.DataFrame, max_abs_angle: float = 180.0, n_trial: int = TRIALS_PER_SESSION) -> float:
    """
    各被験者の平均角度誤差を計算し、0〜1にスケールする。
    """
    needed = ["angular_error_target","angular_error_distractor","chosen_item"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")
    if max_abs_angle <= 0:
        raise ValueError("max_abs_angle must be positive.")

    df = df[df["chosen_item"].isin([0, 1])].copy()
    valid = df.dropna(subset=needed).copy()
    if valid.empty:
        return np.nan

    abs_errors = valid.apply(
        lambda row: abs(row["angular_error_target"]) if row["chosen_item"] == 1 else abs(row["angular_error_distractor"]),
        axis=1
    )
    mean_abs_error = abs_errors.mean()
    scaled = mean_abs_error / max_abs_angle
    return float(np.clip(scaled, 0.0, 1.0))

def compute_out_of_zone_ratio_by_rt(df: pd.DataFrame, n_trial: int = TRIALS_PER_SESSION) -> float:
    """
    各被験者のマインドワンダリング指標（out of the zone）の全試行に対する割合を計算する。

    out of the zone の定義:
      rt < M_mean かつ rt_deviance_mean > T
    を満たす試行を 1 (out of the zone) とラベル付けする。
    """
    needed = ["rt", "ooz", "num_trial"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")

    valid = df.dropna(subset=needed).copy()
    if valid.empty:
        return np.nan
    
    valid = valid[valid["chosen_item"].isin([0, 1])]
    # 今は前後半で分けてMW vs rewardを見ている
    # n_out_of_zone = ((valid["ooz"] == 1) & (valid["num_trial"] <= n_trial - 1)).sum()
    n_out_of_zone = (valid["ooz"] == 1).sum()
    n_total = len(valid)

    return n_out_of_zone / n_total if n_total > 0 else np.nan


def label_if_ooz(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Step1-3 に従ってOOZをラベル付けする。
    OOZ(t) = (M_t < M_mean) and (D_t_smooth > T)
    """
    with_moving = calculate_rt_moving_mean(concat_list, window=3)
    with_deviance = calculate_rt_deviance_mean(with_moving)

    medians = []
    for _, df in with_deviance:
        median_val = df["rt_deviance_mean"].dropna().median()
        if np.isfinite(median_val):
            medians.append(median_val)
    threshold = float(np.mean(medians)) if medians else np.nan

    labeled = []
    for subj_id, df in with_deviance:
        work = df.copy()
        m_mean = work["rt_moving_mean"].dropna().mean()
        cond_fast = work["rt_moving_mean"] < m_mean
        cond_deviant = work["rt_deviance_mean"] > threshold
        work["ooz"] = (cond_fast & cond_deviant).astype(int)
        labeled.append((subj_id, work))
    return labeled
