from typing import List, Tuple

import numpy as np
import pandas as pd

from features.behavior import calculate_rt_moving_mean, calculate_rt_deviance_mean
from io_data.load import exclude_slow_trials
def compute_out_of_zone_ratio_by_AE(df: pd.DataFrame, lower: float = 45.0, upper: float = 60.0) -> float:
    """
    各被験者のマインドワンダリング指標（out of the zone）の全試行に対する割合を計算する。

    out of the zone の定義:
      lower < |angular_error_target| < upper かつ lower < |angular_error_distractor| < upper
    を満たす試行を 1 (out of the zone) とラベル付けする。
    """
    needed = ["angular_error_target", "angular_error_distractor"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")

    valid = df.dropna(subset=needed).copy()
    if valid.empty:
        return np.nan

    abs_target = valid["angular_error_target"].abs()
    abs_distractor = valid["angular_error_distractor"].abs()

    out_of_zone_mask = (
        (abs_target > lower) & (abs_target < upper) &
        (abs_distractor > lower) & (abs_distractor < upper)
    )

    n_out_of_zone = out_of_zone_mask.sum()
    n_total = len(valid)

    return n_out_of_zone / n_total if n_total > 0 else np.nan


def compute_mean_distractor_AngularError_ratio(df: pd.DataFrame, max_abs_angle: float = 180.0) -> float:
    """
    各被験者のdistractorに対する平均角度誤差を計算し、0〜1にスケールする。
    """
    if "angular_error_distractor" not in df.columns:
        raise ValueError("DataFrame lacks 'angular_error_distractor' column.")
    if max_abs_angle <= 0:
        raise ValueError("max_abs_angle must be positive.")

    valid = df["angular_error_distractor"].dropna()
    if valid.empty:
        return np.nan

    mean_abs_error = valid.abs().mean()
    scaled = mean_abs_error / max_abs_angle
    return float(np.clip(scaled, 0.0, 1.0))

def compute_out_of_zone_ratio_by_rt(df: pd.DataFrame) -> float:
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
    n_out_of_zone = ((valid["ooz"] == 1) & (valid["num_trial"] < 20)).sum()
    n_total = len(valid)

    return n_out_of_zone / n_total if n_total > 0 else np.nan


def label_if_ooz(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Step1-3 に従ってOOZをラベル付けする。
    OOZ(t) = (M_t < M_mean) and (D_t_smooth > T)
    """
    with_moving = calculate_rt_moving_mean(concat_list)
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
