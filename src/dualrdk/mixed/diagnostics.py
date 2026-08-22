"""モデル診断。

特に等分散性の検定を重視する。MixedLM は残差分散 sigma^2 に参加者の
添字を持たないため、「全参加者の試行間ばらつきは等しい」と仮定している。
この仮定が破れているなら、破れの中身こそが rt_cv（MW 指標）であり、
location-scale モデルへ進む根拠になる。
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from dualrdk.mixed.results import MixedResult


def residual_diagnostics(res: MixedResult, df: pd.DataFrame, groups: str = "subject") -> dict:
    """残差の正規性と等分散性を調べる。"""
    resid = np.asarray(res.raw.resid)
    rs = pd.DataFrame({"g": df[groups].values, "r": resid})
    per = rs.groupby("g")["r"].std()

    grouped = [v["r"].values for _, v in rs.groupby("g") if len(v) > 1]
    lev_W, lev_p = stats.levene(*grouped, center="median")

    n = min(len(resid), 500)
    sub = np.random.default_rng(0).choice(resid, n, replace=False)
    sh_W, sh_p = stats.shapiro(sub)

    return {
        "skew": float(stats.skew(resid)),
        "kurtosis": float(stats.kurtosis(resid)),
        "shapiro_W": float(sh_W),
        "shapiro_p": float(sh_p),
        "shapiro_n": int(n),
        "resid_sd_assumed": float(np.sqrt(res.scale)) if res.scale else np.nan,
        "resid_sd_min": float(per.min()),
        "resid_sd_max": float(per.max()),
        "resid_sd_ratio": float(per.max() / per.min()),
        "resid_sd_between_sd": float(per.std(ddof=1)),
        "levene_W": float(lev_W),
        "levene_p": float(lev_p),
        "homoscedastic": bool(lev_p >= 0.05),
    }


def per_subject_residual_sd(res: MixedResult, df: pd.DataFrame, groups: str = "subject") -> pd.DataFrame:
    """参加者ごとの残差 SD。これが location-scale モデルでいう sigma_i にあたる。"""
    rs = pd.DataFrame({"subject": df[groups].values, "r": np.asarray(res.raw.resid)})
    out = rs.groupby("subject")["r"].agg(["size", "mean", "std"])
    out.columns = ["n", "resid_mean", "resid_sd"]
    out["log_resid_sd"] = np.log(out["resid_sd"])
    # 参加者ごとの SD 推定の標準誤差 ~ sigma / sqrt(2(n-1))
    out["resid_sd_se"] = out["resid_sd"] / np.sqrt(2 * (out["n"] - 1))
    return out.reset_index()


def trim_impact(
    kept: pd.DataFrame,
    removed: pd.DataFrame,
    trial_col: str = "trial_c",
    groups: str = "subject",
) -> dict:
    """試行除外が系統的な偏りを持たないか調べる。

    除外が試行位置に偏っていると、trial の傾き推定が機械的に歪む。
    """
    if removed is None or len(removed) == 0:
        return {"n_removed": 0, "note": "除外なし"}
    ks = stats.ks_2samp(removed[trial_col], kept[trial_col])
    n_blocks = 6
    span = kept[trial_col].max() + 1
    hist = np.histogram(removed[trial_col], bins=n_blocks, range=(0, span))[0]
    return {
        "n_removed": int(len(removed)),
        "removed_trial_mean": float(removed[trial_col].mean()),
        "kept_trial_mean": float(kept[trial_col].mean()),
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "position_biased": bool(ks.pvalue < 0.05),
        "block_counts": [int(x) for x in hist],
        "late_half_share": float(hist[n_blocks // 2:].sum() / hist.sum()) if hist.sum() else np.nan,
        "n_subjects_affected": int(removed[groups].nunique()),
    }


def dispersion_preservation(
    before: pd.DataFrame, after: pd.DataFrame, value: str = "logrt", groups: str = "subject"
) -> dict:
    """除外の前後で参加者ごとの散布度（= rt_cv の材料）が保たれているか。

    相対基準（MAD 等）の除外は、ばらつきの大きい参加者ほど多く削るため、
    MW 指標としての個人差を圧縮してしまう。絶対基準ならその心配がない。
    """
    sd_b = before.groupby(groups)[value].std()
    sd_a = after.groupby(groups)[value].std()
    joined = pd.concat([sd_b.rename("before"), sd_a.rename("after")], axis=1).dropna()
    return {
        "pearson_r": float(joined["before"].corr(joined["after"])),
        "spearman_r": float(joined["before"].corr(joined["after"], method="spearman")),
        "sd_mean_before": float(joined["before"].mean()),
        "sd_mean_after": float(joined["after"].mean()),
        "sd_range_before": [float(joined["before"].min()), float(joined["before"].max())],
        "sd_range_after": [float(joined["after"].min()), float(joined["after"].max())],
        "between_subject_sd_before": float(joined["before"].std(ddof=1)),
        "between_subject_sd_after": float(joined["after"].std(ddof=1)),
    }
