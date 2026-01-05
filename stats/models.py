from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from io_data.load import combine_subjects


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
