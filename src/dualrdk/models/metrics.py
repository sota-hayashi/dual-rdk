from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, chi2, chi2_contingency, ttest_ind, ttest_rel, ttest_1samp, shapiro, mannwhitneyu, normaltest, lognorm, gamma, invgauss, kstest
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm


from dualrdk.io.utils import combine_subjects
from dualrdk.config import TRIALS_PER_SESSION

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

def t_test_rt_difference_between_target_distractor(
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
    return {
        "n_target": len(target_rt),
        "n_distractor": len(distractor_rt),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 5),
        "mean_target_rt": round(target_rt.mean()),
        "std_target_rt": round(target_rt.std()),
        "mean_distractor_rt": round(distractor_rt.mean()),
        "std_distractor_rt": round(distractor_rt.std()),
        "mean_diff_rt": round(target_rt.mean() - distractor_rt.mean())
    }

def t_test_rt_difference_between_white_black(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, float]:
    """
    白いターゲットと黒いターゲットのRT差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    work = combined.dropna(subset=["rt", "target_group"]).copy()
    if work.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    white_rt = work.loc[work["target_group"] == "white", "rt"].astype(float)
    black_rt = work.loc[work["target_group"] == "black", "rt"].astype(float)
    if white_rt.empty or black_rt.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(white_rt, black_rt, equal_var=False)
    return {
        "n_white": len(white_rt),
        "n_black": len(black_rt),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 5),
        "mean_white_rt": round(white_rt.mean()),
        "std_white_rt": round(white_rt.std()),
        "mean_black_rt": round(black_rt.mean()),
        "std_black_rt": round(black_rt.std()),
        "mean_diff_rt": round(white_rt.mean() - black_rt.mean())
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
    # df = df[df["chosen_item"].isin([0, 1])].copy()
    # df = df.copy()
    df["chosen_item"] = df["chosen_item"].replace({-1: 0, 0: 0, 1: 1})
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
        "n": len(concat_list),
        "t_stat": round(t_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_total": df["chosen_item"].mean(),
        "mean_first_half": round(first_half_choices.mean(), 3),
        "mean_second_half": round(second_half_choices.mean(), 3)
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
        "n": len(concat_list),
        "t_stat": round(t_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_total": round(df["reward_points"].mean(), 3),
        "mean_first_half": round(first_half_points.mean(), 3),
        "mean_second_half": round(second_half_points.mean(), 3)
        }
    return results

def t_test_Minimum_Angular_Error_between_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    タスクの前半/後半の獲得報酬の差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()

    df["error_value"] = np.where(
        df["angular_error_target"].abs() < df["angular_error_distractor"].abs(), # ターゲットAEとディストラクターAEのどちらを選択するか
        df["angular_error_target"].abs(),
        df["angular_error_distractor"].abs(),
    )
    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_half_points = df.loc[df["num_trial"] <= n_trial - 1, "error_value"].abs()
    second_half_points = df.loc[df["num_trial"] >= n_trial, "error_value"].abs()

    if first_half_points.empty or second_half_points.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(
        first_half_points,
        second_half_points,
        equal_var=False
    )
    results = {
        "n": len(concat_list),
        "t_stat": round(t_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_total": round(df["error_value"].abs().mean(), 2),
        "mean_first_half": round(first_half_points.mean(), 2),
        "mean_second_half": round(second_half_points.mean(), 2)
        }
    return results

def t_test_Target_Angular_Error_between_periods(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    タスクの前半/後半の獲得報酬の差をt検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()

    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_half_points = df.loc[df["num_trial"] <= n_trial - 1, "angular_error_target"].abs()
    second_half_points = df.loc[df["num_trial"] >= n_trial, "angular_error_target"].abs()

    if first_half_points.empty or second_half_points.empty:
        return {"t_stat": np.nan, "p_value": np.nan}

    t_stat, p_value = ttest_ind(
        first_half_points,
        second_half_points,
        equal_var=False
    )
    results = {
        "n": len(concat_list),
        "t_stat": round(t_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_total": round(df["angular_error_target"].abs().mean(), 2),
        "mean_first_half": round(first_half_points.mean(), 2),
        "mean_second_half": round(second_half_points.mean(), 2)
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
    # df = df[df["chosen_item"].isin([0, 1])].copy()
    # df = df.copy()
    df["chosen_item"] = df["chosen_item"].replace({-1: 0, 0: 0, 1: 1})
    if df.empty:
        return {"t_stat": np.nan, "p_value": np.nan}
    first_mask = df["num_trial"] <= n_trial - 1
    second_mask = df["num_trial"] >= n_trial

    per_subject = (
        df.assign(half=np.where(first_mask, "first", np.where(second_mask, "second", pd.NA)))
          .dropna(subset=["half"])
          .groupby(["subject", "half"], as_index=False)["chosen_item"]
          .mean()
          .pivot(index="subject", columns="half", values="chosen_item")
    )
    per_subject = per_subject.dropna(subset=["first", "second"]).copy()
    per_subject["delta"] = per_subject["second"]

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
        "t_stat": round(t_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_delta_group1": round(float(group1_delta.mean()), 3),
        "mean_delta_group2": round(float(group2_delta.mean()), 3),
        "n_group1": int(group1_delta.shape[0]),
        "n_group2": int(group2_delta.shape[0])
    }
    return results

def test_shapiro_wilk(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    シャピロ・ウィルク検定を実行して、データが正規分布に従うかどうかを評価する。
    サンプルサイズが小さい場合に特に有効。

    Args:
        data: 検定対象のデータ（リスト形式）。

    Returns:
        (W統計量, p値) のタプル。
        p値が有意水準（例: 0.05）より大きい場合、「データは正規分布に従う」という帰無仮説を棄却できない。
    """    
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()
    # df = df[df["chosen_item"].isin([0, 1])].copy()
    # df = df.copy()
    df["chosen_item"] = df["chosen_item"].replace({-1: 0, 0: 0, 1: 1})
    if df.empty:
        return {"w_stat": np.nan, "p_value": np.nan}
    first_mask = df["num_trial"] <= n_trial - 1
    second_mask = df["num_trial"] >= n_trial

    per_subject = (
        df.assign(half=np.where(first_mask, "first", np.where(second_mask, "second", pd.NA)))
          .dropna(subset=["half"])
          .groupby(["subject", "half"], as_index=False)["chosen_item"]
          .mean()
          .pivot(index="subject", columns="half", values="chosen_item")
    )
    per_subject = per_subject.dropna(subset=["first", "second"]).copy()
    per_subject["delta"] = per_subject["second"] - per_subject["first"]
    per_subject["sum"] = (per_subject["second"] + per_subject["first"]) / 2

    if per_subject["delta"].isna().sum() > 0:
        return {"w_stat": np.nan, "p_value": np.nan}

    w_stat, p_value = shapiro(per_subject["delta"].values)
    results = {
        "w_stat": round(w_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_delta": round(float(per_subject["delta"].mean()), 3),
        "n": int(per_subject["delta"].shape[0])
    }
    return results

def Mann_Whitney_U_test_count_target_choice_between_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    first_group: List[str],
    second_group: List[str],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, float]:
    """
    参加者ごとのターゲット選択確率の前半→後半の変化量を算出し、
    その変化量の群間差をMann-Whitney U検定で検定する。
    """
    combined = combine_subjects(concat_list)
    df = combined.dropna(subset=["chosen_item", "num_trial", "rt"]).copy()
    df["chosen_item"] = df["chosen_item"].replace({-1: 0})
    # df = df[df["chosen_item"].isin([0, 1])].copy()
    if df.empty:
        return {"u_stat": np.nan, "p_value": np.nan}
    first_mask = df["num_trial"] <= n_trial - 1
    second_mask = df["num_trial"] >= n_trial

    per_subject = (
        df.assign(half=np.where(first_mask, "first", np.where(second_mask, "second", pd.NA)))
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
        return {"u_stat": np.nan, "p_value": np.nan}

    u_stat, p_value = mannwhitneyu(group1_delta, group2_delta, alternative='two-sided')
    results = {
        "u_stat": round(u_stat, 2),
        "p_value": f"{p_value:.6f}",
        "mean_delta_group1": round(float(group1_delta.mean()), 3),
        "mean_delta_group2": round(float(group2_delta.mean()), 3),
        "n_group1": int(group1_delta.shape[0]),
        "n_group2": int(group2_delta.shape[0])
    }
    return results

def test_log_normality(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    log_arr = np.log(arr)

    shapiro_stat, shapiro_p = shapiro(log_arr)
    normaltest_stat, normaltest_p = normaltest(log_arr)

    return {
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "normaltest_stat": normaltest_stat,
        "normaltest_p": normaltest_p,
    }

def compare_distributions(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    results = {}

    # lognormal
    shape, loc, scale = lognorm.fit(arr, floc=0)
    ll = np.sum(lognorm.logpdf(arr, shape, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * ll   # shape, scale の2パラメータ（loc固定）
    results["lognorm"] = {"loglik": ll, "AIC": aic}

    # gamma
    a, loc, scale = gamma.fit(arr, floc=0)
    ll = np.sum(gamma.logpdf(arr, a, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * ll
    results["gamma"] = {"loglik": ll, "AIC": aic}

    # inverse Gaussian
    mu, loc, scale = invgauss.fit(arr, floc=0)
    ll = np.sum(invgauss.logpdf(arr, mu, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * ll
    results["invgauss"] = {"loglik": ll, "AIC": aic}

    return results

def evaluate_lognormal_fit(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    if arr.size < 3:
        raise ValueError("Not enough positive finite data.")

    log_arr = np.log(arr)
    mu = log_arr.mean()
    sigma = log_arr.std(ddof=1)

    # KS statistic
    ks_D, ks_p = kstest(arr, 'lognorm', args=(sigma, 0, np.exp(mu)))

    # Normality test on log-values
    shapiro_W, shapiro_p = shapiro(log_arr)

    # Log-likelihood
    loglik = np.sum(lognorm.logpdf(arr, s=sigma, loc=0, scale=np.exp(mu)))

    # AIC (2 parameters: mu, sigma)
    aic = 2 * 2 - 2 * loglik

    return {
        "n": len(arr),
        "mu_log": mu,
        "sigma_log": sigma,
        "ks_D": ks_D,
        "ks_p": ks_p,
        "shapiro_W_on_log": shapiro_W,
        "shapiro_p_on_log": shapiro_p,
        "loglik": loglik,
        "AIC": aic,
    }

import numpy as np
from scipy import stats
from typing import Any, Dict, Optional


def fit_exgaussian_and_evaluate(
    values,
    n_mc_samples: int = 5000,
    statistic: str = "ad",   # "ad", "ks", "cvm", "filliben"
    random_state: Optional[int] = 42,
    floc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    RTデータに ex-Gaussian (scipy.stats.exponnorm) を最尤推定でフィットし、
    Monte Carlo goodness-of-fit test により適合度を評価する。

    Parameters
    ----------
    values : array-like
        RTデータ（正の実数を想定）
    n_mc_samples : int
        goodness_of_fit に使う Monte Carlo サンプル数
    statistic : str
        goodness_of_fit の統計量
        "ad", "ks", "cvm", "filliben" から選択
    random_state : int or None
        乱数シード
    floc : float or None
        loc を固定したいときに指定。通常は None のままでよい。
        RTが必ず正で、理論上0起点に寄せたいなど特殊な意図がある場合だけ使う。

    Returns
    -------
    result : dict
        フィット結果と適合度指標をまとめた辞書
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 5:
        raise ValueError("有効なデータ数が少なすぎます。少なくとも5点以上を推奨します。")
    if np.any(arr <= 0):
        raise ValueError("RTデータとして正の値を想定しているため、0以下の値が含まれています。")

    # 1) MLE fit
    if floc is None:
        K, loc, scale = stats.exponnorm.fit(arr)
    else:
        K, loc, scale = stats.exponnorm.fit(arr, floc=floc)

    # ex-Gaussian のよくあるパラメータへ変換
    # SciPy: loc = mu, scale = sigma, tau = K * sigma
    mu = float(loc)
    sigma = float(scale)
    tau = float(K * scale)
    lam = float(1.0 / tau)

    # 2) 対数尤度・AIC・BIC
    logpdf = stats.exponnorm.logpdf(arr, K, loc=loc, scale=scale)
    loglik = float(np.sum(logpdf))
    k_params = 3 if floc is None else 2
    aic = float(2 * k_params - 2 * loglik)
    bic = float(k_params * np.log(arr.size) - 2 * loglik)

    # 3) 経験CDFと理論CDFの最大差（参考）
    # goodness_of_fit の "ks" と似た発想だが、ここでは単なる記述量として計算
    x_sorted = np.sort(arr)
    ecdf = np.arange(1, arr.size + 1) / arr.size
    fitted_cdf = stats.exponnorm.cdf(x_sorted, K, loc=loc, scale=scale)
    ks_distance_descriptive = float(np.max(np.abs(ecdf - fitted_cdf)))

    # 4) Monte Carlo goodness-of-fit test
    # SciPy が利用可能なら、推定後再フィットを含む正しい流れで p 値を返す
    gof = stats.goodness_of_fit(
        dist=stats.exponnorm,
        data=arr,
        statistic=statistic,
        n_mc_samples=n_mc_samples,
        rng=random_state,
    )

    # 5) 追加で KS / CvM / AD の記述的統計量だけ並べたい場合
    #   - p値は statistic で選んだもののみ返す
    #   - 他は「ズレ量の参考値」として扱う
    def _ks_stat(data):
        return stats.kstest(
            data,
            stats.exponnorm.cdf,
            args=(K, loc, scale)
        ).statistic

    def _cvm_stat(data):
        return stats.cramervonmises(
            data,
            lambda x: stats.exponnorm.cdf(x, K, loc=loc, scale=scale)
        ).statistic

    ks_stat_desc = float(_ks_stat(arr))
    cvm_stat_desc = float(_cvm_stat(arr))

    # Anderson-Darling は SciPy に ex-Gaussian 用の単独関数がないので、
    # 汎用 goodness_of_fit の statistic="ad" を使うのが自然
    result = {
        "n": int(arr.size),
        "fit_scipy": {
            "K": float(K),
            "loc": float(loc),
            "scale": float(scale),
        },
        "fit_exgaussian": {
            "mu": mu,
            "sigma": sigma,
            "tau": tau,
            "lambda": lam,
        },
        "fit_quality": {
            "loglik": loglik,
            "AIC": aic,
            "BIC": bic,
            "ks_distance_descriptive": ks_distance_descriptive,
            "ks_stat_descriptive": ks_stat_desc,
            "cvm_stat_descriptive": cvm_stat_desc,
        },
        "goodness_of_fit_test": {
            "statistic_name": statistic,
            "statistic_value": float(gof.statistic),
            "pvalue": float(gof.pvalue),
            "n_mc_samples": int(n_mc_samples),
        },
        "interpretation_hint": (
            "pvalue が小さいほど『この ex-Gaussian から来たとみなすにはズレが大きい』方向です。"
            "逆に pvalue が十分大きければ、少なくともこの検定では ex-Gaussian を棄却しません。"
            "ただし pvalue は『モデルが真である確率』ではありません。"
        ),
    }
    return result

def t_test_angular_error_paired(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, float]:
    """
    各参加者のOOZ/非OOZ試行のangular error平均を算出し，
    対応のあるt検定で全参加者にわたって検定する．
    """
    records = []
    for subj_id, df in concat_list:
        df = df.copy()
        df["angular_error"] = np.where(
            df["angular_error_target"].abs() < df["angular_error_distractor"].abs(),
            df["angular_error_target"].abs(),
            df["angular_error_distractor"].abs(),
        )
        work = df.dropna(subset=["angular_error", "ooz"])

        ooz_mean = work.loc[work["ooz"] == 1, "angular_error"].mean()
        ooz_std = work.loc[work["ooz"] == 1, "angular_error"].std()
        non_ooz_mean = work.loc[work["ooz"] == 0, "angular_error"].mean()
        non_ooz_std = work.loc[work["ooz"] == 0, "angular_error"].std()

        # OOZ試行数が少なすぎる参加者は除外
        n_ooz = (work["ooz"] == 1).sum()
        n_non_ooz = (work["ooz"] == 0).sum()
        if n_ooz < 3 or n_non_ooz < 3:
            continue
        if np.isnan(ooz_mean) or np.isnan(non_ooz_mean):
            continue

        records.append({
            "subject_id": subj_id,
            "ooz_mean": ooz_mean,
            "non_ooz_mean": non_ooz_mean,
            "ooz_std": ooz_std,
            "non_ooz_std": non_ooz_std,
            "diff": ooz_mean - non_ooz_mean,
            "n_ooz": n_ooz,
            "n_non_ooz": n_non_ooz,
        })

    if len(records) < 2:
        return {"t_stat": np.nan, "p_value": np.nan}

    result_df = pd.DataFrame(records)
    t_stat, p_value = ttest_rel(result_df["ooz_mean"], result_df["non_ooz_mean"])

    return {
        "n_participants": len(result_df),
        "mean_ooz": round(result_df["ooz_mean"].mean(), 2),
        "mean_non_ooz": round(result_df["non_ooz_mean"].mean(), 2),
        "std_ooz": round(result_df["ooz_std"].mean(), 2),
        "std_non_ooz": round(result_df["non_ooz_std"].mean(), 2),
        "mean_diff": round(result_df["diff"].mean(), 2),
        "std_diff": round(result_df["diff"].std(), 2),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 5),
    }

def lmm_angular_error_ooz(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, float]:
    """
    全試行のデータを使い，OOZ状態がangular_errorに与える影響を
    線形混合効果モデルで推定する．参加者をランダム切片として扱う．
    """
    rows = []
    for pid, df in concat_list:
        df = df.copy()
        df["angular_error"] = np.where(
            df["angular_error_target"].abs() < df["angular_error_distractor"].abs(),
            df["angular_error_target"].abs(),
            df["angular_error_distractor"].abs(),
        )
        work = df.dropna(subset=["angular_error", "ooz"])
        work["subject_id"] = pid
        rows.append(work[["subject_id", "angular_error", "ooz"]])

    if not rows:
        return {"coef_ooz": np.nan, "p_value": np.nan}

    all_data = pd.concat(rows, ignore_index=True)
    all_data["ooz"] = all_data["ooz"].astype(float)

    model = mixedlm(
        "angular_error ~ ooz",
        data=all_data,
        groups=all_data["subject_id"]
    )
    result = model.fit(reml=True)

    coef_ooz = result.params["ooz"]
    se_ooz = result.bse["ooz"]
    z_stat = result.tvalues["ooz"]   # mixedlmはz統計量
    p_value = result.pvalues["ooz"]

    return {
        "n_trials": len(all_data),
        "n_participants": all_data["subject_id"].nunique(),
        "coef_ooz": round(coef_ooz, 3),
        "se_ooz": round(se_ooz, 3),
        "z_stat": round(z_stat, 3),
        "p_value": round(p_value, 5),
    }

# =====================================================================
# MDP / POMDP 判定：状態の観測可能性（state observability）の定量化
# ---------------------------------------------------------------------
# 判定の枠組み（3本立て）
#   (1) 単一観測から状態が同定できるか      -> H(S | O_t)
#         MDP なら H(S|O_t) = 0（観測が状態を一意に決める）
#         POMDP なら H(S|O_t) > 0（最大は 1 bit = P(S|O) が一様）
#   (2) その状態が報酬に効くか（潜在変数を捨てられないか） -> I(R ; S | A)
#         (1) で H(S|O)>0 でも、S が報酬に無関係なら S を消して観測を
#         そのまま状態とみなせる（= MDP に落とせる）。落とせないことを示す。
#   (3) 履歴が状態情報を運ぶか              -> H(S | O_t, h_{1:t-1})
#         MDP なら履歴は無情報（現在観測が十分統計量）。
#         POMDP なら履歴で不確実性が減る = 信念状態が必要。
# =====================================================================

# experiment/index.html の templateRanges（セッション回転 r を引いた座標での区間）
TEMPLATE_RANGES = {
    "W_H": (0.0, 45.0),
    "B_H": (90.0, 135.0),
    "W_L": (180.0, 225.0),
    "B_L": (270.0, 315.0),
}


def _wrap_signed_180(x: np.ndarray) -> np.ndarray:
    """角度差を [-180, 180) に畳み込む。"""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def _entropy_bits(counts: np.ndarray) -> float:
    """カウントベクトルからエントロピー[bit]を計算する。"""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _conditional_entropy_bits(obs_codes: np.ndarray, states: np.ndarray) -> float:
    """H(S | O) [bit] を経験分布から計算する。"""
    obs_codes = np.asarray(obs_codes)
    states = np.asarray(states)
    n = len(states)
    if n == 0:
        return np.nan
    h_cond = 0.0
    for o in np.unique(obs_codes):
        mask = obs_codes == o
        sub = states[mask]
        counts = np.array([np.sum(sub == s) for s in np.unique(states)], dtype=float)
        h_cond += (mask.sum() / n) * _entropy_bits(counts)
    return float(h_cond)


def _map_decode_accuracy(obs_codes: np.ndarray, states: np.ndarray) -> float:
    """観測セルごとの多数決（MAP デコード）で状態を当てる正答率（in-sample）。"""
    obs_codes = np.asarray(obs_codes)
    states = np.asarray(states)
    if len(states) == 0:
        return np.nan
    correct = 0
    for o in np.unique(obs_codes):
        mask = obs_codes == o
        sub = states[mask]
        vals, counts = np.unique(sub, return_counts=True)
        correct += counts.max()
    return float(correct / len(states))


def build_state_observation_table(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    各試行を (観測 o_t, 状態 s_t, 行動 a_t, 報酬 r_t) の組に整形する。

    観測 o_t : 参加者がその試行で実際に受け取る情報 = 白ドット群の運動方向と
               黒ドット群の運動方向（画面座標）。ドット数は白黒とも 100 で
               等しく、教示上の「多い方」は実際には手掛かりにならない。
    状態 s_t : 報酬が与えられる側（ターゲット）の色。1 = white, 0 = black。
               これは方向テンプレート（W_H/B_H/...）とセッション回転 r で
               決まるが、r は参加者に一切呈示されない潜在変数である。
    行動 a_t : 参加者が回答した方向がどちらの色の運動方向に近かったか。
    報酬 r_t : reward_points。
    """
    required = ["target_group", "target_direction", "distractor_direction", "session_rotation"]
    frames = []
    for subj_id, df in concat_list:
        if any(col not in df.columns for col in required):
            continue
        work = df.dropna(subset=["target_group", "target_direction", "distractor_direction"]).copy()
        if work.empty:
            continue
        is_white_target = work["target_group"].astype(str).eq("white").to_numpy()
        target_dir = work["target_direction"].astype(float).to_numpy()
        distractor_dir = work["distractor_direction"].astype(float).to_numpy()
        white_dir = np.where(is_white_target, target_dir, distractor_dir)
        black_dir = np.where(is_white_target, distractor_dir, target_dir)
        rotation = work["session_rotation"].astype(float).to_numpy()

        frames.append(pd.DataFrame({
            "subject": subj_id,
            "num_trial": work["num_trial"].to_numpy() if "num_trial" in work.columns else np.arange(len(work)),
            "white_dir": white_dir % 360.0,
            "black_dir": black_dir % 360.0,
            # 観測の内部構造（2方向の相対角）: 状態に依存しないことの確認用
            "rel_dir": (white_dir - black_dir) % 360.0,
            # 実験者だけが知る回転量 r を差し引いた「整列座標」での観測
            "aligned_white_dir": (white_dir - rotation) % 360.0,
            "aligned_black_dir": (black_dir - rotation) % 360.0,
            "session_rotation": rotation,
            "state_white_target": is_white_target.astype(int),
            "chosen_color": work["chosen_color"].to_numpy() if "chosen_color" in work.columns else np.nan,
            "chosen_item": work["chosen_item"].to_numpy() if "chosen_item" in work.columns else np.nan,
            "response_angle_rdk": work["response_angle_rdk"].astype(float).to_numpy() if "response_angle_rdk" in work.columns else np.nan,
            "reward_points": work["reward_points"].astype(float).to_numpy() if "reward_points" in work.columns else np.nan,
        }))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def state_observability_from_single_observation(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_bins: int = 12,
    n_perm: int = 2000,
    random_state: int = 0,
) -> Dict[str, object]:
    """
    (1) 「1試行の観測 o_t だけで状態 s_t を同定できるか」を H(S|O) で測る。

    観測の符号化（フレーム）を 4 通り比べる:
      raw_white        : 白ドット方向のみ（画面座標）を n_bins 個に離散化
      raw_joint        : 白・黒両方向の同時ビン（画面座標）
      aligned_white    : セッション回転 r を差し引いた座標での白方向
                         （= 参加者が r を完全に知っている場合の観測）
      within_subject   : 被験者 ID × 画面座標の白方向
                         （= r を1人分だけ同定済みの観測に相当）

    P(S|O) が一様（0.5）なら H(S|O) = 1 bit で、その観測では状態を全く
    決められない。逆に H(S|O) = 0 なら観測が状態を一意に決める（= MDP）。
    """
    table = build_state_observation_table(concat_list)
    if table.empty:
        return {"error": "no data"}

    bin_width = 360.0 / n_bins
    white_bin = np.floor(table["white_dir"].to_numpy() / bin_width).astype(int)
    black_bin = np.floor(table["black_dir"].to_numpy() / bin_width).astype(int)
    aligned_bin = np.floor(table["aligned_white_dir"].to_numpy() / bin_width).astype(int)
    subj_codes = pd.factorize(table["subject"])[0]
    states = table["state_white_target"].to_numpy()

    frames = {
        "raw_white": white_bin,
        "raw_joint": white_bin * n_bins + black_bin,
        "aligned_white": aligned_bin,
        "within_subject": subj_codes * n_bins + white_bin,
    }

    h_state = _entropy_bits(np.array([np.sum(states == 0), np.sum(states == 1)]))
    rng = np.random.default_rng(random_state)

    results = {}
    for name, codes in frames.items():
        h_cond = _conditional_entropy_bits(codes, states)
        mi = h_state - h_cond
        acc = _map_decode_accuracy(codes, states)

        # 帰無分布: 被験者内で状態ラベルをシャッフル（回転量の被験者間差を保存）
        null_mi = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = states.copy()
            for s in np.unique(subj_codes):
                mask = subj_codes == s
                shuffled[mask] = rng.permutation(states[mask])
            null_mi[i] = h_state - _conditional_entropy_bits(codes, shuffled)
        p_perm = (np.sum(null_mi >= mi) + 1) / (n_perm + 1)

        results[name] = {
            "n_trials": int(len(states)),
            "n_obs_cells": int(len(np.unique(codes))),
            "H_S": round(h_state, 4),
            "H_S_given_O": round(h_cond, 4),
            "MI_bits": round(mi, 4),
            "MI_bits_bias_corrected": round(mi - float(null_mi.mean()), 4),
            "MI_null_mean": round(float(null_mi.mean()), 4),
            "p_perm": round(float(p_perm), 5),
            "map_decode_acc": round(acc, 4),
        }

    # P(S=white | 観測ビン) が一様(0.5)かどうかを直接示す
    cond_probs = {}
    for name in ["raw_white", "aligned_white"]:
        codes = frames[name]
        probs = []
        for o in np.unique(codes):
            mask = codes == o
            probs.append(float(states[mask].mean()))
        probs = np.array(probs)
        cond_probs[name] = {
            "P_white_target_given_bin": [round(float(p), 3) for p in probs],
            "mean": round(float(probs.mean()), 4),
            "sd": round(float(probs.std(ddof=1)), 4) if len(probs) > 1 else np.nan,
            "max_abs_dev_from_0.5": round(float(np.max(np.abs(probs - 0.5))), 4),
        }

    # 独立性の χ2 検定（生座標フレーム）
    ct = pd.crosstab(frames["raw_white"], states)
    chi2_stat, chi2_p, dof, _ = chi2_contingency(ct)

    # 観測の内部構造（白-黒の相対角）が状態に依存しないことの確認
    rel = table["rel_dir"].to_numpy()
    rel_white = rel[states == 1]
    rel_black = rel[states == 0]

    return {
        "frames": results,
        "conditional_probs": cond_probs,
        "chi2_raw_white": {
            "chi2": round(float(chi2_stat), 3),
            "dof": int(dof),
            "p_value": round(float(chi2_p), 5),
        },
        "relative_direction_check": {
            "mean_rel_dir_white_target": round(float(rel_white.mean()), 2),
            "mean_rel_dir_black_target": round(float(rel_black.mean()), 2),
            "ks_p": round(float(kstest(rel_white, rel_black).pvalue), 5),
        },
    }


def state_relevance_for_reward(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, object]:
    """
    (2) 潜在状態 S が報酬に効くか（= S を捨てて観測をそのまま状態にできないか）。

    行動 a_t（どちらの色の方向を回答したか）を固定しても、報酬は S に強く依存する。
    I(R ; S | A) > 0 であれば、S は報酬関数に不可欠な変数であり、
    「H(S|O) が高いから S を無視して観測=状態と見なす」ことは許されない。
    """
    table = build_state_observation_table(concat_list)
    if table.empty:
        return {"error": "no data"}

    work = table.dropna(subset=["reward_points"]).copy()
    work = work[work["chosen_color"].isin(["white", "black"])]
    if work.empty:
        return {"error": "no valid choices"}

    action = (work["chosen_color"] == "white").astype(int).to_numpy()  # 1=白方向を回答
    state = work["state_white_target"].to_numpy()                      # 1=白がターゲット
    reward = work["reward_points"].to_numpy()
    reward_bin = (reward > 2).astype(int)

    # セルごとの平均報酬
    cell_stats = []
    for a in [1, 0]:
        for s in [1, 0]:
            mask = (action == a) & (state == s)
            if mask.sum() == 0:
                continue
            cell_stats.append({
                "action": "report_white_dir" if a == 1 else "report_black_dir",
                "state": "white_is_target" if s == 1 else "black_is_target",
                "n": int(mask.sum()),
                "mean_reward": round(float(reward[mask].mean()), 3),
                "p_high_reward": round(float(reward_bin[mask].mean()), 3),
            })

    # I(R ; S | A) [bit]
    h_r_given_a = _conditional_entropy_bits(action, reward_bin)
    h_r_given_as = _conditional_entropy_bits(action * 2 + state, reward_bin)
    mi_cond = h_r_given_a - h_r_given_as

    return {
        "cells": pd.DataFrame(cell_stats),
        "H_R_given_A": round(float(h_r_given_a), 4),
        "H_R_given_A_S": round(float(h_r_given_as), 4),
        "MI_R_S_given_A_bits": round(float(mi_cond), 4),
    }


def _ideal_observer_filter(
    df: pd.DataFrame,
    lapse: float = 0.02,
    reward_max_error: float = 50.0,
) -> Dict[str, np.ndarray]:
    """
    セッション回転 r（潜在変数, 一様事前）に対する理想観測者ベイズフィルタ。

    r ∈ {0,...,359} の各仮説について、その試行の白方向を整列座標に直すと
    W_H 区間なら「白がターゲット」、W_L 区間なら「黒がターゲット」が含意される。
    この観測者は生成構造（テンプレート区間と報酬関数）を既知としており、
    実際の参加者より強い。したがってこの観測者の不確実性は、
    人間が到達しうる下限（＝同定可能性の上界）を与える。

    返り値は試行ごとの
      belief_white : 報酬観測前の信念 P(S_t = white_target | o_t, h_{1:t-1})
      entropy      : そのエントロピー[bit]
      correct      : MAP 予測が真の状態と一致したか
    """
    grid = np.arange(360, dtype=float)
    log_b = np.zeros(360)  # 一様事前（対数, 非正規化）

    w_lo, w_hi = TEMPLATE_RANGES["W_H"]
    l_lo, l_hi = TEMPLATE_RANGES["W_L"]
    bh_lo, bh_hi = TEMPLATE_RANGES["B_H"]
    bl_lo, bl_hi = TEMPLATE_RANGES["B_L"]

    beliefs, entropies, corrects = [], [], []

    for _, row in df.iterrows():
        wd, bd = row["white_dir"], row["black_dir"]
        aligned_w = (wd - grid) % 360.0
        aligned_b = (bd - grid) % 360.0

        # 仮説A: 白がターゲット（白は W_H, 黒は B_L から引かれている）
        mask_white = (aligned_w >= w_lo) & (aligned_w < w_hi) & (aligned_b >= bl_lo) & (aligned_b < bl_hi)
        # 仮説B: 黒がターゲット（白は W_L, 黒は B_H）
        mask_black = (aligned_w >= l_lo) & (aligned_w < l_hi) & (aligned_b >= bh_lo) & (aligned_b < bh_hi)

        b = np.exp(log_b - log_b.max())
        b = b / b.sum()
        p_white = b[mask_white].sum()
        p_black = b[mask_black].sum()
        denom = p_white + p_black
        belief_white = 0.5 if denom <= 0 else float(p_white / denom)

        beliefs.append(belief_white)
        entropies.append(_entropy_bits(np.array([belief_white, 1.0 - belief_white])))
        true_state = int(row["state_white_target"])
        if belief_white == 0.5:
            corrects.append(0.5)  # 完全に無情報 = コイン投げ
        else:
            corrects.append(float((belief_white > 0.5) == (true_state == 1)))

        # --- 観測（幾何的整合性）による更新 ---
        consistent = mask_white | mask_black
        log_b = log_b + np.log(np.where(consistent, 1.0 - lapse, lapse))

        # --- 報酬フィードバックによる更新 ---
        resp = row["response_angle_rdk"]
        obs_reward = row["reward_points"]
        if np.isfinite(resp) and np.isfinite(obs_reward):
            implied_target = np.where(mask_white, wd, np.where(mask_black, bd, np.nan))
            err = np.abs(_wrap_signed_180(resp - implied_target))
            pred_reward = np.maximum(0.0, np.floor(10.0 * (1.0 - err / reward_max_error)))
            match = np.isclose(pred_reward, obs_reward)
            match = np.where(np.isnan(implied_target), False, match)
            log_b = log_b + np.log(np.where(match, 1.0 - lapse, lapse))

        log_b = log_b - log_b.max()

    return {
        "belief_white": np.array(beliefs),
        "entropy": np.array(entropies),
        "correct": np.array(corrects),
    }


def state_observability_from_history(
    concat_list: List[Tuple[str, pd.DataFrame]],
    lapse: float = 0.02,
    early_late_split: int = 8,
) -> Dict[str, object]:
    """
    (3) 「履歴 h_{1:t-1} を足すと状態の不確実性が減るか」を理想観測者で測る。

    MDP であれば現在の観測が十分統計量なので履歴は無情報のはず。
    実際には H(S_t | o_t) = 1 bit（同定不能）から、履歴を足すと
    H(S_t | o_t, h_{1:t-1}) が単調に下がる。この差分こそが信念状態
    （belief state）が運ぶ情報量であり、POMDP の定義そのものである。
    """
    table = build_state_observation_table(concat_list)
    if table.empty:
        return {"error": "no data"}

    per_trial_entropy = {}
    per_trial_correct = {}
    subj_summary = []

    for subj_id, sub in table.groupby("subject", sort=False):
        sub = sub.sort_values("num_trial")
        out = _ideal_observer_filter(sub, lapse=lapse)
        n = len(out["entropy"])
        for t in range(n):
            per_trial_entropy.setdefault(t, []).append(out["entropy"][t])
            per_trial_correct.setdefault(t, []).append(out["correct"][t])
        subj_summary.append({
            "subject": subj_id,
            "n_trials": n,
            "H_first_trial": out["entropy"][0],
            "H_early": float(np.mean(out["entropy"][:early_late_split])),
            "H_late": float(np.mean(out["entropy"][early_late_split:])),
            "acc_early": float(np.mean(out["correct"][:early_late_split])),
            "acc_late": float(np.mean(out["correct"][early_late_split:])),
            "trials_to_H_below_0.1": int(np.argmax(out["entropy"] < 0.1) + 1)
                if np.any(out["entropy"] < 0.1) else np.nan,
        })

    summary_df = pd.DataFrame(subj_summary)
    trial_curve = pd.DataFrame({
        "num_trial": sorted(per_trial_entropy.keys()),
        "mean_entropy_bits": [float(np.mean(per_trial_entropy[t])) for t in sorted(per_trial_entropy)],
        "mean_decode_acc": [float(np.mean(per_trial_correct[t])) for t in sorted(per_trial_correct)],
        "n_subjects": [len(per_trial_entropy[t]) for t in sorted(per_trial_entropy)],
    })

    t_stat, p_value = ttest_rel(summary_df["H_early"], summary_df["H_late"])

    return {
        "trial_curve": trial_curve,
        "subject_summary": summary_df,
        "H_S_given_o_only": 1.0,  # 単一観測での上界（解析的に一様）
        "mean_H_first_trial": round(float(summary_df["H_first_trial"].mean()), 4),
        "mean_H_early": round(float(summary_df["H_early"].mean()), 4),
        "mean_H_late": round(float(summary_df["H_late"].mean()), 4),
        "info_from_history_bits": round(float(1.0 - summary_df["H_late"].mean()), 4),
        "mean_acc_early": round(float(summary_df["acc_early"].mean()), 4),
        "mean_acc_late": round(float(summary_df["acc_late"].mean()), 4),
        "median_trials_to_identify": float(summary_df["trials_to_H_below_0.1"].median()),
        "t_stat_early_vs_late": round(float(t_stat), 3),
        "p_value_early_vs_late": f"{p_value:.3e}",
        "n_subjects": int(len(summary_df)),
    }


def state_stability_across_phases(
    learning_list: List[Tuple[str, pd.DataFrame]],
    awareness_list: List[Tuple[str, pd.DataFrame]],
    n_bins: int = 12,
) -> Dict[str, object]:
    """
    補足: 整列座標（回転 r を既知とした観測）ですら、learning ブロックと
    awareness(probe) ブロックで同じ観測が逆の状態に対応する。
    つまり「r さえ分かれば MDP」も厳密には成立せず、状態は真に潜在である。
    """
    learn = build_state_observation_table(learning_list)
    aware = build_state_observation_table(awareness_list)
    if learn.empty or aware.empty:
        return {"error": "no data"}

    bin_width = 360.0 / n_bins
    out = {}
    for name, tbl in [("learning", learn), ("awareness_probe", aware)]:
        codes = np.floor(tbl["aligned_white_dir"].to_numpy() / bin_width).astype(int)
        states = tbl["state_white_target"].to_numpy()
        probs = {int(o): round(float(states[codes == o].mean()), 3) for o in np.unique(codes)}
        out[name] = {
            "n_trials": int(len(states)),
            "P_white_target_by_aligned_bin": probs,
        }

    pooled = pd.concat([learn, aware], ignore_index=True)
    codes = np.floor(pooled["aligned_white_dir"].to_numpy() / bin_width).astype(int)
    states = pooled["state_white_target"].to_numpy()
    h_state = _entropy_bits(np.array([np.sum(states == 0), np.sum(states == 1)]))
    h_cond = _conditional_entropy_bits(codes, states)
    out["pooled_learning_plus_probe"] = {
        "H_S": round(h_state, 4),
        "H_S_given_aligned_O": round(h_cond, 4),
        "MI_bits": round(h_state - h_cond, 4),
    }
    return out


def simulate_raw_frame_mi_null(
    n_subjects: int = 61,
    n_trials: int = TRIALS_PER_SESSION,
    n_bins: int = 12,
    n_sim: int = 500,
    random_state: int = 0,
) -> Dict[str, object]:
    """
    生座標フレームの相互情報量 I(S ; O_raw) に対する「デザイン水準の帰無分布」。

    参加者から見た P(S | o_t) は、回転量 r ~ Uniform(0,360) について周辺化すると
    解析的にちょうど 0.5（= MI 0 bit）になる。しかし実データは有限個
    （= 被験者数）の r しか含まないため、被験者間プールした経験 MI はゼロには
    ならない。ここでは experiment/index.html の生成過程をそのまま再現して、
    「真に一様な r のもとで n_subjects 人ぶん集めたときに偶然生じる MI」の
    分布を求め、実データの値がその範囲に収まるかを判定する。

    なおこの残差 MI は被験者間の r のばらつきに由来するものであり、
    自分の r しか知らない個々の参加者には原理的に利用できない。
    """
    rng = np.random.default_rng(random_state)
    bin_width = 360.0 / n_bins
    w_h, b_l = TEMPLATE_RANGES["W_H"], TEMPLATE_RANGES["B_L"]
    w_l, b_h = TEMPLATE_RANGES["W_L"], TEMPLATE_RANGES["B_H"]

    def _sample_pair(is_white_target: bool) -> Tuple[float, float]:
        # index.html と同じく「方向差 >= 90 度」を最大20回まで再サンプルする
        wr, br = (w_h, b_l) if is_white_target else (w_l, b_h)
        for _ in range(20):
            aw = np.floor(wr[0] + rng.random() * (wr[1] - wr[0]))
            ab = np.floor(br[0] + rng.random() * (br[1] - br[0]))
            raw = abs(aw - ab)
            diff = 360 - raw if raw > 180 else raw
            if diff >= 90:
                break
        return aw, ab

    mis = np.empty(n_sim)
    for i in range(n_sim):
        white_dirs, states = [], []
        for _ in range(n_subjects):
            rot = float(rng.integers(0, 360))
            labels = np.array([1] * (n_trials // 2) + [0] * (n_trials // 2))
            rng.shuffle(labels)
            for lab in labels:
                aw, _ab = _sample_pair(bool(lab))
                white_dirs.append((aw + rot) % 360.0)
                states.append(int(lab))
        codes = np.floor(np.array(white_dirs) / bin_width).astype(int)
        states = np.array(states)
        h_s = _entropy_bits(np.array([np.sum(states == 0), np.sum(states == 1)]))
        mis[i] = h_s - _conditional_entropy_bits(codes, states)

    return {
        "n_sim": n_sim,
        "n_subjects": n_subjects,
        "mean_MI_bits": round(float(mis.mean()), 4),
        "sd_MI_bits": round(float(mis.std(ddof=1)), 4),
        "q025": round(float(np.quantile(mis, 0.025)), 4),
        "q975": round(float(np.quantile(mis, 0.975)), 4),
        "samples": mis,
    }
