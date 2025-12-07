from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2_contingency, chi2, pearsonr
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = Path("data/experiment_data_202511251648.csv")

PRACTICE_ROWS = 6
ROWS_PER_SESSION = 64
TRIALS_PER_SESSION = 32


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and report its shape/columns."""
    df = pd.read_csv(path)
    # print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def filter_task_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with target_group white/black and drop initial practice block."""
    task_mask = df["target_group"].isin(["white", "black"])
    task_df = df.loc[task_mask].copy()
    # print(f"Rows with target_group white/black: {len(task_df)}")
    trimmed_df = task_df.iloc[PRACTICE_ROWS:].reset_index(drop=True)
    # print(f"After dropping first {PRACTICE_ROWS} rows (practice): {len(trimmed_df)} rows remain")
    return trimmed_df


def annotate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add num_session and num_trial columns as described."""
    expected_rows = ROWS_PER_SESSION * 3
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows after trimming, got {len(df)}")

    df = df.copy()
    df["num_session"] = df.index // ROWS_PER_SESSION
    df["num_trial"] = (df.index % ROWS_PER_SESSION) // 2
    return df


def concatenate_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the two rows (RDK + response) that share the same num_session/num_trial.
    - The first row acts as the base.
    - reward_points / rotation_angle from the second row are copied into new columns.
    """
    combined_rows = []
    group_cols = ["num_session", "num_trial"]

    for key, group in df.groupby(group_cols, sort=False):
        group_sorted = group.sort_index()
        if len(group_sorted) != 2:
            raise ValueError(f"Expected 2 rows per trial for {key}, found {len(group_sorted)}")
        base = group_sorted.iloc[0].copy()
        follow = group_sorted.iloc[1]
        base["reward_points"] = follow.get("reward_points")
        base["angular_error_target"] = follow.get("angular_error_target")
        base["angular_error_distractor"] = follow.get("angular_error_distractor")
        combined_rows.append(base)

    combined_df = pd.DataFrame(combined_rows).reset_index(drop=True)
    return combined_df


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


def analyze_reward_learning(df: pd.DataFrame, n_perm: int = 5000) -> pd.DataFrame:
    """For each session, correlate num_trial with reward_points using Spearman + permutation test."""
    records: List[Dict[str, float]] = []
    for session_id, sub in df.groupby("num_session"):
        valid = sub.dropna(subset=["num_trial", "reward_points"])
        if valid["num_trial"].nunique() <= 1 or len(valid) < 3:
            continue
        stats = permutation_spearman(valid["num_trial"].to_numpy(), valid["reward_points"].to_numpy(), n_perm=n_perm, random_state=session_id)
        records.append({"num_session": session_id, **stats, "n_trials": len(valid)})
    return pd.DataFrame(records)


def analyze_reward_learning_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]], n_perm: int = 5000) -> pd.DataFrame:
    """
    Pool all subjects, then for each session compute Pearson r(num_trial, reward_points) with permutation p.
    This answers: 「セッション内で全被験者を平均的に見たとき、試行進行と報酬の傾向はあるか？」。
    """
    combined = combine_subjects(concat_list).dropna(subset=["num_trial", "reward_points", "num_session"])
    if combined.empty:
        return pd.DataFrame()
    rows = []
    for session_id, sub in combined.groupby("num_session"):
        stats = permutation_spearman(sub["num_trial"].to_numpy(), sub["reward_points"].to_numpy(), n_perm=n_perm, random_state=int(session_id))
        rows.append({"num_session": session_id, **stats, "n_trials": len(sub), "n_subjects": sub["subject"].nunique()})
    return pd.DataFrame(rows)

def mean_reward_by_trial_across_subjects(concat_list):
    df = combine_subjects(concat_list)
    agg = df.groupby(["num_session", "num_trial"])["reward_points"].mean().reset_index(name="mean_reward")
    rows = []
    for sess, sub in agg.groupby("num_session"):
        r, p = pearsonr(sub["num_trial"], sub["mean_reward"])
        rows.append({"num_session": sess, "rho": r, "p": p, "n_trials": len(sub)})
    return pd.DataFrame(rows)


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


def extract_aftered_color(dot_color_value) -> str:
    """
    dot_color には '["white","black"]' のような文字列が入るので、先頭要素を aftered_color として返す。
    """
    if isinstance(dot_color_value, list):
        return dot_color_value[0] if dot_color_value else None
    if isinstance(dot_color_value, str):
        try:
            parsed = json.loads(dot_color_value)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except json.JSONDecodeError:
            pass
    return None


def analyze_after_target_dependency(df: pd.DataFrame) -> Dict[str, object]:
    """
    Count combinations of aftered_color (first entry of dot_color) and target_color (target_group),
    and run chi-square to see if there is dependency. Also returns same/different counts.
    """
    df = df.copy()
    df["aftered_color"] = df["dot_color"].apply(extract_aftered_color)
    df["target_color"] = df["target_group"]
    valid = df.dropna(subset=["aftered_color", "target_color"])

    # Count combinations
    ww = len(valid[(valid["aftered_color"] == "white") & (valid["target_color"] == "white")])
    wb = len(valid[(valid["aftered_color"] == "white") & (valid["target_color"] == "black")])
    bw = len(valid[(valid["aftered_color"] == "black") & (valid["target_color"] == "white")])
    bb = len(valid[(valid["aftered_color"] == "black") & (valid["target_color"] == "black")])

    contingency = np.array([[ww, wb], [bw, bb]])
    chi2, p, dof, expected = chi2_contingency(contingency)

    # same vs different
    same = ww + bb
    diff = wb + bw

    return {
        "counts": {"ww": ww, "wb": wb, "bw": bw, "bb": bb},
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "expected": expected,
        "same_diff_counts": {"same": same, "different": diff},
    }


def compute_same_diff_stats(df: pd.DataFrame) -> Dict[str, int]:
    """Return same/different counts based on aftered_color vs target_color."""
    stats = analyze_after_target_dependency(df)
    return stats["same_diff_counts"]


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Full pipeline: load -> filter -> annotate -> concatenate."""
    df = load_data(path)
    trimmed = filter_task_rows(df)
    annotated = annotate_sessions(trimmed)
    concatenated = concatenate_trials(annotated)
    return concatenated


def load_all_concatenated(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load all csvs in data_dir and return list of (subject_id, concatenated_df)."""
    datasets = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        subj_id = csv_path.stem
        try:
            concat_df = load_and_prepare(csv_path)
            datasets.append((subj_id, concat_df))
        except Exception as e:
            print(f"Skipping {csv_path.name}: {e}")
    return datasets


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


def analyze_same_diff_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]], n_perm: int = 5000) -> Dict[str, object]:
    """
    Across subjects: compute same/total proportion for each subject, then sign-flip test vs 0.5.
    """
    proportions = []
    per_subject = []
    for subj_id, df in concat_list:
        same_diff = compute_same_diff_stats(df)
        total = same_diff["same"] + same_diff["different"]
        if total == 0:
            continue
        prop_same = same_diff["same"] / total
        proportions.append(prop_same)
        per_subject.append({"subject": subj_id, "same": same_diff["same"], "different": same_diff["different"], "prop_same": prop_same})
    if not proportions:
        return {"per_subject": per_subject, "test": None}
    test = permutation_sign_test(np.array(proportions), center=0.5, n_perm=n_perm, random_state=123)
    return {"per_subject": per_subject, "test": test}


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
        a_list.append(a); b_list.append(b); c_list.append(c); d_list.append(d); n_list.append(n)
    if not n_list:
        return {"chi2": np.nan, "p_value": np.nan, "dof": 1, "tables": len(n_list)}

    # CMH statistic
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


def analyze_after_target_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]]) -> Dict[str, object]:
    """
    Build per-subject 2x2 tables (aftered x target) and run CMH test for overall dependency.
    """
    per_subject = []
    tables = []
    for subj_id, df in concat_list:
        stats = analyze_after_target_dependency(df)
        per_subject.append({"subject": subj_id, **stats["counts"]})
        tables.append(stats["counts"])
    cmh_stats = cmh_test_2x2(tables)
    return {"per_subject": per_subject, "cmh": cmh_stats}


def combine_subjects(concat_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-subject concatenated data with a subject column."""
    rows = []
    for subj_id, df in concat_list:
        tmp = df.copy()
        tmp["subject"] = subj_id
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def mixed_learning_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]], n_perm: int = 500, random_state: int = 0) -> Dict[str, object]:
    """
    Mixed model across subjects: reward_points ~ num_trial + num_session + (1|subject)
    Permutation p-values via shuffling reward_points.
    """
    combined = combine_subjects(concat_list)
    combined = combined.dropna(subset=["reward_points", "num_trial", "num_session", "subject"])
    if combined.empty:
        return {"model": None, "permutation": None}

    rng = np.random.default_rng(random_state)
    model = smf.mixedlm("reward_points ~ num_trial + num_session", data=combined, groups=combined["subject"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fit = model.fit(reml=False, maxiter=500, disp=False)
    obs_beta = fit.params

    # track permuted betas for num_trial and num_session
    perm_beta = {k: [] for k in obs_beta.index}
    for _ in range(n_perm):
        perm_data = combined.copy()
        perm_data["reward_points"] = rng.permutation(perm_data["reward_points"])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                perm_fit = smf.mixedlm("reward_points ~ num_trial + num_session", data=perm_data, groups=perm_data["subject"]).fit(reml=False, maxiter=300, disp=False)
            for k in obs_beta.index:
                perm_beta[k].append(perm_fit.params.get(k, np.nan))
        except Exception:
            # skip failed fits
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

    return {"model_params": obs_beta.to_dict(), "model_tvalues": fit.tvalues.to_dict(), "permutation": perm_stats}


def add_is_target_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_target=1 if reward_points>0 else 0."""
    out = df.copy()
    out["is_target"] = np.where(out["reward_points"] > 0, 1, 0)
    return out


def analyze_target_choice_learning(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, logistic regression of is_target ~ num_trial.
    Returns slope and p-value (Wald).
    """
    rows = []
    for sess, sub in df.groupby("num_session"):
        if sub["is_target"].nunique() < 2:
            rows.append({"num_session": sess, "beta_trial": np.nan, "p_value": np.nan, "n": len(sub), "note": "no variance"})
            continue
        # add intercept and fit logit
        X = sm.add_constant(sub["num_trial"])
        try:
            fit = sm.Logit(sub["is_target"], X).fit(disp=False)
            beta = fit.params.get("num_trial", np.nan)
            pval = fit.pvalues.get("num_trial", np.nan)
            rows.append({"num_session": sess, "beta_trial": beta, "p_value": pval, "n": len(sub), "note": ""})
        except Exception as e:
            rows.append({"num_session": sess, "beta_trial": np.nan, "p_value": np.nan, "n": len(sub), "note": f"fit error: {e}"})
    return pd.DataFrame(rows)


def count_high_angular_error(df: pd.DataFrame, threshold: float = 45.0) -> int:
    """
    Count trials where angular_error >= threshold.
    Assumes df contains angular_error (e.g., filtered 192-row data, not concatenated).
    """
    if "angular_error" not in df.columns:
        raise ValueError("angular_error column not found in DataFrame.")
    valid = df.dropna(subset=["angular_error"])
    return int((valid["angular_error"].abs() >= threshold).sum())


def plot_reward_by_trial(df: pd.DataFrame, session_id: int, save_path: str = None, corner: str = "upper left"):
    """
    Scatter plot of trial vs reward for a given session with Pearson r and regression line (95% CI).
    Colors aligned with learning-more-code style: blue dots, red line, red-transparent CI.
    """
    sub = df[df["num_session"] == session_id].dropna(subset=["num_trial", "reward_points"]).copy()
    if sub.empty:
        print(f"No data for session {session_id}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
      x="num_trial",
      y="reward_points",
      data=sub,
      ax=ax,
      scatter_kws={"alpha": 0.5},
      line_kws={"color": "red"},
      ci=95
    )

    r, p = spearmanr(sub["num_trial"], sub["reward_points"])
    xpos = 0.95 if "right" in corner else 0.05
    ypos = 0.95
    ax.text(
        xpos, ypos,
        f"$r_{{s}}=${r:.3f}\n$p={p:.3f}$",
        transform=ax.transAxes,
        ha="right" if "right" in corner else "left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    ax.set_title(f"Subject3: Reward vs Trial (session {session_id+1})", fontsize=14)
    ax.set_xlabel("Trial (num_trial)", fontsize=12)
    ax.set_ylabel("Reward points", fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()