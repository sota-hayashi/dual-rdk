from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from io_data.load import combine_subjects
from common.config import TRIALS_PER_SESSION


def mixed_learning_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_perm: int = 100,
    random_state: int = 0
) -> Dict[str, object]:
    """
    Mixed model across subjects: reward_points ~ num_trial + (1|subject)
    Permutation p-values via shuffling reward_points.
    """
    combined = combine_subjects(concat_list)
    combined = combined.dropna(subset=["reward_points", "num_trial", "subject"])
    if combined.empty:
        return {"model": None, "permutation": None}

    rng = np.random.default_rng(random_state)
    model = smf.mixedlm("reward_points ~ num_trial", data=combined, groups=combined["subject"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fit = model.fit(reml=False, maxiter=500, disp=False)
    obs_beta = fit.params

    perm_beta = {k: [] for k in obs_beta.index}
    for _ in range(n_perm):
        perm_data = combined.copy()
        perm_data["reward_points"] = rng.permutation(perm_data["reward_points"])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                perm_fit = smf.mixedlm(
                    "reward_points ~ num_trial",
                    data=perm_data,
                    groups=perm_data["subject"]
                ).fit(reml=False, maxiter=300, disp=False)
            for k in obs_beta.index:
                perm_beta[k].append(perm_fit.params.get(k, np.nan))
        except Exception:
            continue

    perm_stats = {}
    for k, obs in obs_beta.items():
        vals = np.array([v for v in perm_beta[k] if np.isfinite(v)])
        if len(vals) == 0:
            perm_stats[k] = {"p_perm": np.nan, "n_perm": 0}
            continue
        extreme = np.sum(np.abs(vals) >= abs(obs))
        p_perm = (extreme + 1) / (len(vals) + 1)
        perm_stats[k] = {"p_perm": p_perm, "n_perm": len(vals)}

    return {
        "model_params": obs_beta.to_dict(),
        "model_tvalues": fit.tvalues.to_dict(),
        "permutation": perm_stats
    }

def logit_regression(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 2
) -> Dict[str, object]:
    """
    Logistic regression to predict target choice (0/1) from trial number.
    """
    results = {}
    for subj_id, combined in concat_list:
        combined = combined.dropna(subset=["chosen_item", "num_trial"])
        combined = combined[combined["chosen_item"].isin([0, 1])].copy()
        # combined = combined[combined["num_trial"] > n_trial - 1]
        if combined.empty:
            continue
        X = combined[["num_trial"]]
        X = sm.add_constant(X)
        y = combined["chosen_item"]

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        params = result.params
        pvalues = result.pvalues

        if pvalues.get('num_trial', 1.0) < 0.05:
            results[subj_id] = {
                "intercept": params.get('const'),
                "coef_num_trial": params.get('num_trial'),
                "p_intercept": pvalues.get('const'),
                "p_num_trial": pvalues.get('num_trial'),
                "model": result
            }
    return results

def linear_regression(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, object]:
    """
    Simple linear regression to predict reward_points from trial number for each subject.
    """
    results = {}
    for subj_id, combined in concat_list:
        combined = combined.dropna(subset=["reward_points", "num_trial"]).copy()
        if combined.empty:
            continue
            
        X = combined[["num_trial"]]
        X = sm.add_constant(X)
        y = combined["reward_points"]

        model = sm.OLS(y, X)
        result = model.fit()

        params = result.params
        pvalues = result.pvalues

        if pvalues.get('num_trial', 1.0) < 0.05:
            results[subj_id] = {
                "intercept": params.get('const'),
                "coef_num_trial": params.get('num_trial'),
                "p_intercept": pvalues.get('const'),
                "p_num_trial": pvalues.get('num_trial'),
                "model": result
            }
    return results
