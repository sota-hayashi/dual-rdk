from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

from io_data.load import combine_subjects
from stats.metrics import permutation_spearman, permutation_mean_diff


# -------- 相関を調べる関数群 --------
def analyze_reward_learning(df: pd.DataFrame, n_perm: int = 5000) -> pd.DataFrame:
    """For each session, correlate num_trial with reward_points using Spearman + permutation test."""
    records: List[Dict[str, float]] = []
    for session_id, sub in df.groupby("num_session"):
        valid = sub.dropna(subset=["num_trial", "reward_points"])
        if valid["num_trial"].nunique() <= 1 or len(valid) < 3:
            continue
        stats = permutation_spearman(
            valid["num_trial"].to_numpy(),
            valid["reward_points"].to_numpy(),
            n_perm=n_perm,
            random_state=session_id
        )
        records.append({"num_session": session_id, **stats, "n_trials": len(valid)})
    return pd.DataFrame(records)


def analyze_reward_learning_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_perm: int = 5000
) -> pd.DataFrame:
    """
    Pool all subjects, then for each session compute Pearson r(num_trial, reward_points) with permutation p.
    This answers: 「セッション内で全被験者を平均的に見たとき、試行進行と報酬の傾向はあるか？」。
    """
    combined = combine_subjects(concat_list).dropna(subset=["num_trial", "reward_points", "num_session"])
    if combined.empty:
        return pd.DataFrame()
    rows = []
    for session_id, sub in combined.groupby("num_session"):
        stats = permutation_spearman(
            sub["num_trial"].to_numpy(),
            sub["reward_points"].to_numpy(),
            n_perm=n_perm,
            random_state=int(session_id)
        )
        rows.append({
            "num_session": session_id,
            **stats,
            "n_trials": len(sub),
            "n_subjects": sub["subject"].nunique()
        })
    return pd.DataFrame(rows)


def mean_reward_by_trial_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    df = combine_subjects(concat_list)
    agg = df.groupby(["num_session", "num_trial"])["reward_points"].mean().reset_index(name="mean_reward")
    rows = []
    for sess, sub in agg.groupby("num_session"):
        r, p = pearsonr(sub["num_trial"], sub["mean_reward"])
        rows.append({"num_session": sess, "rho": r, "p": p, "n_trials": len(sub)})
    return pd.DataFrame(rows)


# -------- 回帰を調べる関数群 --------
def analyze_reward_learning_regression(concat_list: pd.DataFrame) -> pd.DataFrame:
    combined = combine_subjects(concat_list).dropna(subset=["num_trial", "reward_points", "num_session"])
    for subj_id, sub in combined.groupby("num_session"):
        valid = sub.dropna(subset=["num_trial", "reward_points"])
        if valid["num_trial"].nunique() <= 1 or len(valid) < 3:
            continue
        X = sm.add_constant(valid["num_trial"])
        model = sm.OLS(valid["reward_points"], X)
        results = model.fit()
        print(f"Across all subjects, {subj_id} session regression results:")
        print(results.summary())
    return pd.DataFrame()


def analyze_reward_color_dependency(df: pd.DataFrame, n_perm: int = 5000) -> Dict[str, float]:
    """Check if reward_points differ between target_group levels via permutation mean-diff test."""
    valid = df.dropna(subset=["reward_points", "target_group"])
    groups = [(name, grp["reward_points"].to_numpy()) for name, grp in valid.groupby("target_group")]
    if len(groups) != 2:
        raise ValueError("Need exactly two target_group levels for dependency test.")
    (group_a, values_a), (group_b, values_b) = groups
    stats = permutation_mean_diff(values_a, values_b, n_perm=n_perm, random_state=42)
    stats.update({"group_a": group_a, "group_b": group_b, "n_a": len(values_a), "n_b": len(values_b)})
    return stats
