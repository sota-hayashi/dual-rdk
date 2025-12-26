import numpy as np
import pandas as pd


def compute_out_of_zone_ratio(df: pd.DataFrame, lower: float = 45.0, upper: float = 60.0) -> float:
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
