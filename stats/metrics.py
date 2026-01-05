from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2
import statsmodels.api as sm
from statsmodels.formula.api import ols

from io_data.load import combine_subjects


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
    model = ols("rt ~ C(chosen_item)", data=work).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    group_stats = work.groupby("chosen_item")["rt"].agg(
        n="count",
        mean="mean",
        std="std"
    ).reset_index()
    return {"anova": anova_table, "group_stats": group_stats}
