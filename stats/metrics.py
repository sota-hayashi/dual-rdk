from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2, ttest_ind, ttest_1samp
import statsmodels.api as sm
from statsmodels.formula.api import ols


from io_data.utils import combine_subjects
from common.config import TRIALS_PER_SESSION


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, random_state: int = 0) -> Dict[str, float]:
    """Compute Spearman r and permutation p-value (two-sided)."""
    rho, _ = spearmanr(x, y)
    rng = np.random.default_rng(random_state)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        permuted = rng.permutation(y)
        perm_rhos[i], _ = spearmanr(x, permuted)
    extreme = np.sum(np.abs(perm_rhos) >= abs(rho))
    p_perm = (extreme + 1) / (n_perm + 1)
    return {"rho": rho, "p_perm": p_perm}


def permutation_mean_diff(values_a: np.ndarray, values_b: np.ndarray, n_perm: int = 5000, random_state: int = 0) -> Dict[str, float]:
    """Permutation test for difference in means between two groups."""
    actual = values_a.mean() - values_b.mean()
    combined = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        diffs[i] = perm[:n_a].mean() - perm[n_a:].mean()
    extreme = np.sum(np.abs(diffs) >= abs(actual))
    p_perm = (extreme + 1) / (n_perm + 1)
    return {"diff": actual, "p_perm": p_perm}


def permutation_sign_test(values: np.ndarray, center: float = 0.0, n_perm: int = 5000, random_state: int = 0) -> Dict[str, float]:
    """
    One-sample sign-flip permutation test for mean(values - center) != 0.
    """
    rng = np.random.default_rng(random_state)
    diffs = values - center
    observed = diffs.mean()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_stats[i] = (diffs * signs).mean()
    extreme = np.sum(np.abs(perm_stats) >= abs(observed))
    p_perm = (extreme + 1) / (n_perm + 1)
    return {"mean": observed + center, "delta": observed, "p_perm": p_perm}


def cmh_test_2x2(tables: Iterable[Dict[str, int]]) -> Dict[str, float]:
    """
    Cochran-Mantel-Haenszel test for common odds ratio across strata (subjects).
    Each table is dict with keys ww, wb, bw, bb.
    """
    a_list, b_list, c_list, d_list, n_list = [], [], [], [], []
    for t in tables:
        a, b, c, d = t["ww"], t["wb"], t["bw"], t["bb"]
        n = a + b + c + d
        if n == 0:
            continue
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
        d_list.append(d)
        n_list.append(n)
    if not n_list:
        return {"chi2": np.nan, "p_value": np.nan, "dof": 1, "tables": len(n_list)}

    numer = 0.0
    denom = 0.0
    for a, b, c, d, n in zip(a_list, b_list, c_list, d_list, n_list):
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d
        expected_a = row1 * col1 / n
        var_a = (row1 * row2 * col1 * col2) / (n * n * (n - 1)) if n > 1 else 0
        numer += (a - expected_a)
        denom += var_a
    chi2_stat = (numer ** 2) / denom if denom > 0 else np.nan
    p_val = 1 - chi2.cdf(chi2_stat, df=1) if denom > 0 else np.nan
    return {"chi2": chi2_stat, "p_value": p_val, "dof": 1, "tables": len(n_list)}


def anova_rt_by_chosen_item(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, object]:
    """
    chosen_item別（-1/0/1）のRT差を一次元配置ANOVAで検定する。
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return {"anova": pd.DataFrame(), "group_stats": pd.DataFrame()}

    work = combined.dropna(subset=["rt", "chosen_item"]).copy()
    if work.empty:
        return {"anova": pd.DataFrame(), "group_stats": pd.DataFrame()}

    work["chosen_item"] = work["chosen_item"].astype(int)
    # work = work[work["chosen_item"].isin([0, 1])]
    model = ols("rt ~ C(chosen_item)", data=work).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    group_stats = work.groupby("chosen_item")["rt"].agg(
        n="count",
        mean="mean",
        std="std"
    ).reset_index()
    return {"anova": anova_table, "group_stats": group_stats}

def anova_reward_by_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 3
) -> Dict[str, object]:
    """
    タスクの前期/中期/後期の報酬ポイント差を一元配置ANOVAで検定する。
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return {"anova": pd.DataFrame(), "group_stats": pd.DataFrame()}

    work = combined.dropna(subset=["rt", "reward_points", "num_trial"]).copy()
    if work.empty:
        return {"anova": pd.DataFrame(), "group_stats": pd.DataFrame()}

    work["trial_period"] = pd.cut(
        work["num_trial"],
        bins=[-1, n_trial - 1, 2 * n_trial - 1, 3 * n_trial - 1],
        labels=["early", "middle", "late"]
    )
    work["trial_period"] = work["trial_period"].astype("category")
    model = ols("reward_points ~ C(trial_period)", data=work).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    group_stats = work.groupby("trial_period")["reward_points"].agg(
        n="count",
        mean="mean",
        std="std"
    ).reset_index()
    return {"anova": anova_table, "group_stats": group_stats}

def anova_count_of_target_choice_by_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 3
) -> Dict[str, object]:
    """
    タスクの前期/中期/後期のターゲット選択数の差を一元配置ANOVA検定で検定する。
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return {"anova": {}, "group_stats": pd.DataFrame()}

    work = combined.dropna(subset=["rt", "chosen_item", "num_trial"]).copy()
    if work.empty:
        return {"anova": {}, "group_stats": pd.DataFrame()}
    work["trial_period"] = pd.cut(
        work["num_trial"],
        bins=[-1, n_trial - 1, 2 * n_trial - 1, 3 * n_trial - 1],
        labels=["early", "middle", "late"]
    )
    work["trial_period"] = work["trial_period"].astype("category")
    work["is_target"] = (work["chosen_item"] == 1).astype(int)
    anova_table = sm.stats.anova_lm(
        ols("is_target ~ C(trial_period)", data=work).fit(),
        typ=2
    )
    group_stats = work.groupby("trial_period")["is_target"].agg(
        n="count",
        mean="mean",
        std="std"
    ).reset_index()
    return {"anova": anova_table, "group_stats": group_stats}

def t_test_rt_between_choices(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, float]:
    """
    ターゲット選択試行とディストラクター選択試行のRT差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    work = combined.dropna(subset=["rt", "chosen_item"]).copy()
    if work.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    target_rt = work.loc[work["chosen_item"] == 1, "rt"].astype(float)
    distractor_rt = work.loc[work["chosen_item"] == 0, "rt"].astype(float)
    if target_rt.empty or distractor_rt.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(target_rt, distractor_rt, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value}


def t_test_learning_rate_from_switch_probs(
    subject_prob_list: List[Tuple[str, List[float]]]
) -> Dict[str, float]:
    """
    被験者ごとの学習率 ((probs[-1] - probs[0]) / (n-1)) を計算し、
    0との一標本t検定を行う。
    """
    rates = []
    for subj_id, probs in subject_prob_list:
        if probs is None or len(probs) < 2:
            continue
        n = len(probs)
        rate = (probs[-1] - probs[0]) / (n - 1)
        if np.isfinite(rate):
            rates.append(rate)
        print(f"Subject {subj_id}: learning rate = {rate}")

    if len(rates) == 0:
        return {"t_stat": np.nan, "p_value": np.nan, "mean_rate": np.nan, "n": 0}

    rates_arr = np.array(rates, dtype=float)
    t_stat, p_value = ttest_1samp(rates_arr, 0.0)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_rate": float(rates_arr.mean()),
        "n": len(rates_arr),
    }

def t_test_count_target_choice_between_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    タスクの前半/後半のターゲット選択数の差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()
    df = df[df["chosen_item"].isin([0, 1])].copy()
    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_half_choices = df.loc[df["num_trial"] <= n_trial - 1, "chosen_item"]
    second_half_choices = df.loc[df["num_trial"] >= n_trial, "chosen_item"]

    if first_half_choices.empty or second_half_choices.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    t_stat, p_value = ttest_ind(
        first_half_choices,
        second_half_choices,
        equal_var=False
    )
    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_total": df["chosen_item"].mean(),
        "mean_first_half": first_half_choices.mean(),
        "mean_second_half": second_half_choices.mean()
    }
    return results

def t_test_reward_points_between_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    タスクの前半/後半の獲得報酬の差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()
    df = df[df["chosen_item"].isin([0, 1])].copy()
    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_half_points = df.loc[df["num_trial"] <= n_trial - 1, "reward_points"]
    second_half_points = df.loc[df["num_trial"] >= n_trial, "reward_points"]

    if first_half_points.empty or second_half_points.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(
        first_half_points,
        second_half_points,
        equal_var=False
    )
    results = {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_first_half": first_half_points.mean(),
            "mean_second_half": second_half_points.mean()
        }
    return results

def t_test_count_target_choice_between_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    first_group: List[str],
    second_group: List[str],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    参加者ごとのターゲット選択確率の前半→後半の変化量を算出し、
    その変化量の群間差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()
    df = df[df["chosen_item"].isin([0, 1])].copy()
    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_mask = df["num_trial"] <= n_trial - 1
    second_mask = df["num_trial"] >= n_trial

    per_subject = (
        df.assign(half=np.where(first_mask, "first", np.where(second_mask, "second", np.nan)))
          .dropna(subset=["half"])
          .groupby(["subject", "half"], as_index=False)["chosen_item"]
          .mean()
          .pivot(index="subject", columns="half", values="chosen_item")
    )
    per_subject = per_subject.dropna(subset=["first", "second"]).copy()
    per_subject["delta"] = per_subject["second"] - per_subject["first"]

    group1_delta = per_subject.loc[per_subject.index.isin(first_group), "delta"]
    group2_delta = per_subject.loc[per_subject.index.isin(second_group), "delta"]

    if group1_delta.empty or group2_delta.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(
        group1_delta,
        group2_delta,
        equal_var=False
    )
    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_delta_group1": float(group1_delta.mean()),
        "mean_delta_group2": float(group2_delta.mean()),
        "n_group1": int(group1_delta.shape[0]),
        "n_group2": int(group2_delta.shape[0])
    }
    return results
