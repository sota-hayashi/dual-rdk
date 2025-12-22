from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Callable
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

# デフォルトのデータパスはJSONに差し替え（必要に応じて変更してください）
DATA_PATH = Path("data_online_experiment/596634e005f2df00017281ae.json")

PRACTICE_ROWS = 6
ROWS_PER_SESSION = 100
TRIALS_PER_SESSION = 50
ROWS_FOR_AWARENESS = 16


def load_data(path: Path) -> pd.DataFrame:
    """
    Load CSV or JSON (jsPsych export) into a DataFrame.
    - CSV: same挙動 as before.
    - JSON: expects an array of trial dictionaries (jsPsych.data.get().json()).
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with open(path, "r") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported file type: {suffix}")


def filter_task_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with target_group white/black and drop initial practice block."""
    task_mask = df["target_group"].isin(["white", "black"])
    task_df = df.loc[task_mask].copy()
    # print(f"Rows with target_group white/black: {len(task_df)}")
    trimmed_df = task_df.iloc[PRACTICE_ROWS:].reset_index(drop=True)
    trimmed_df_learning = trimmed_df.iloc[:-ROWS_FOR_AWARENESS].copy()
    trimmed_df_awareness = trimmed_df.iloc[-ROWS_FOR_AWARENESS:].copy()
    # print(f"After dropping first {PRACTICE_ROWS} rows (practice): {len(trimmed_df)} rows remain")
    return trimmed_df_learning, trimmed_df_awareness


def annotate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add num_session and num_trial columns as described."""
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
        base["rt"] = follow.get("rt")
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

# -------- 相関を調べる関数群 --------
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

# ------------------------------------

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


def analyze_color_accuracy_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    被験者内解析: 白/黒のどちらを選んだか推定し、色ごとに角度誤差の変化を回帰で検定。
    - df: load_and_prepare 済み（単一被験者）を想定。
    - 選択色は |angular_error_target| と |angular_error_distractor| を比較して決定。
      target_group=white かつ |target|<|distractor| -> white 選択, 逆なら black 選択（black target も同様）。
      ※誤って反対方向を選んだケースは一旦無視（必要なら別途フラグ化）。
    """
    needed = ["angular_error_target", "angular_error_distractor", "target_group", "reward_points"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")

    work = df.dropna(subset=needed + ["num_trial"]).copy()
    if work.empty:
        return pd.DataFrame()

    abs_t = work["angular_error_target"].abs()
    abs_d = work["angular_error_distractor"].abs()

    # 推定した選択色
    def infer_choice(row):
        if abs(row["angular_error_target"]) <= abs(row["angular_error_distractor"]):
            return row["target_group"]  # 目標色に近い
        # ここで「反対方向を選んだ可能性」を考慮する場合は別フラグを立てて判定する
        return "white" if row["target_group"] == "black" else "black"

    work["chosen_color"] = work.apply(infer_choice, axis=1)
    work["chosen_error"] = np.where(
        work["chosen_color"] == work["target_group"],
        work["angular_error_target"],
        work["angular_error_distractor"],
    )

    # 色ごとの試行インデックス（累積カウンタ）
    work["color_trial_index"] = work.groupby("chosen_color").cumcount()

    rows = []
    for color, sub in work.groupby("chosen_color"):
        if len(sub) < 3:
            rows.append({
                "color": color,
                "n": len(sub),
                "mean_abs_error": sub["chosen_error"].abs().mean(),
                "slope": np.nan,
                "p_value": np.nan,
                "note": "not enough trials"
            })
            continue
        X = sm.add_constant(sub["color_trial_index"])
        fit = sm.OLS(sub["chosen_error"], X).fit()
        slope = fit.params.get("color_trial_index", np.nan)
        pval = fit.pvalues.get("color_trial_index", np.nan)
        rows.append({
            "color": color,
            "n": len(sub),
            "mean_reward": sub["reward_points"].mean(),
            "mean_abs_error": sub["chosen_error"].abs().mean(),
            "slope": slope,
            "p_value": pval,
            "note": ""
        })

    return pd.DataFrame(rows)


def analyze_color_accuracy_change_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]] , alpha: float = 0.05) -> Dict[str, object]:
    """
    全被験者に対して analyze_color_accuracy_change を実行し、
    p値が alpha 以下のケースをカウントする。
    Returns:
      {
        "all_results": DataFrame(subject, color, n, mean_abs_error, slope, p_value, note),
        "significant": DataFrame(上記から p<=alpha を抽出),
        "n_sig": 件数（行数ベース）,
        "alpha": alpha
      }
    """
    per_subject_rows = []
    for subj_id, df in concat_list:
        try:
            res = analyze_color_accuracy_change(df)
            if res.empty:
                continue
            res = res.copy()
            res["subject"] = subj_id
            per_subject_rows.append(res)
        except Exception as e:
            print(f"Skipping {subj_id}: {e}")
            continue

    if not per_subject_rows:
        return {"all_results": pd.DataFrame(), "significant": pd.DataFrame(), "n_sig": 0, "alpha": alpha}

    all_results = pd.concat(per_subject_rows, ignore_index=True)
    significant = all_results.loc[all_results["p_value"] <= alpha].copy()
    return {
        "all_results": all_results,
        "significant": significant,
        "n_sig": len(significant),
        "alpha": alpha
    }

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
    trimmed_learning, trimmed_awareness = filter_task_rows(df)
    annotated_learning = annotate_sessions(trimmed_learning)
    concatenated_learning = concatenate_trials(annotated_learning)
    annotated_awareness = annotate_sessions(trimmed_awareness)
    concatenated_awareness = concatenate_trials(annotated_awareness)
    return concatenated_learning, concatenated_awareness

def load_all_concatenated(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load all csv/json in data_dir and return list of (subject_id, concatenated_df)."""
    datasets_learning = []
    datasets_awareness = []
    for file_path in sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))):
        subj_id = file_path.stem
        # print(f"Loading subject {subj_id} from {file_path.name}...")
        # if subj_id in [
        #             #      "666306b0bf2de127943c419f"
        #             #    , "667aca76f4fb2f1d50d80c2e"
        #             #    , "673757f92aa69c13b7841d90"
        #             #    , "673f0e83fbba6c167eebd6f7"
        #             #    , "677e4656af6e5525f72fc926"
        #             #    , "678f3b13379c83cf1027d2ed"
        #             ]:
        #     continue
        try:
            concat_df_learning, concat_df_awareness = load_and_prepare(file_path)
            datasets_learning.append((subj_id, concat_df_learning))
            datasets_awareness.append((subj_id, concat_df_awareness))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
    return datasets_learning, datasets_awareness

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


def mixed_learning_across_subjects(concat_list: List[Tuple[str, pd.DataFrame]], n_perm: int = 100, random_state: int = 0) -> Dict[str, object]:
    """
    Mixed model across subjects: reward_points ~ num_trial + num_session + (1|subject)
    Permutation p-values via shuffling reward_points.
    """
    combined = combine_subjects(concat_list)
    print(combined)
    combined = combined.dropna(subset=["reward_points", "num_trial", "subject"])
    if combined.empty:
        return {"model": None, "permutation": None}

    rng = np.random.default_rng(random_state)
    model = smf.mixedlm("reward_points ~ num_trial", data=combined, groups=combined["subject"])
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
                perm_fit = smf.mixedlm("reward_points ~ num_trial", data=perm_data, groups=perm_data["subject"]).fit(reml=False, maxiter=300, disp=False)
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

def extract_rts_from_online_data(data_dir: Path) -> List[float]:
    """
    指定されたディレクトリ内のすべてのJSONファイルから'rt'を抽出します。
    'rt'が存在し、nullでない試行のみを対象とします。
    """
    all_rts = []
    if not data_dir.is_dir():
        print(f"Error: Directory not found at {data_dir}")
        return all_rts

    for file_path in sorted(data_dir.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                trials = json.load(f)
                for trial in trials:
                    # trialが辞書型であり、'rt'キーを持つか確認
                    if isinstance(trial, dict) and 'rt' in trial:
                        rt_value = trial['rt']
                        # rt_valueがNoneでないことを確認
                        if rt_value is not None:
                            all_rts.append(float(rt_value))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path.name}")
        except Exception as e:
            print(f"Warning: An error occurred while processing {file_path.name}: {e}")

    return all_rts

def plot_regression_across_subjects(concat_list, session_id=0, save_path=None):
    df = combine_subjects(concat_list).dropna(subset=["num_trial", "reward_points", "num_session"])
    sub = df[df["num_session"] == session_id]
    if sub.empty:
        print(f"No data for session {session_id}")
        return

    # OLSで傾きなど計算
    X = sm.add_constant(sub["num_trial"])
    fit = sm.OLS(sub["reward_points"], X).fit()
    slope = fit.params["num_trial"]
    pval = fit.pvalues["num_trial"]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x="num_trial", y="reward_points", data=sub,
                scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}, ci=95, ax=ax)
    ax.set_title(f"Across subjects: Reward vs Trial (session {session_id+1})")
    ax.text(0.05, 0.95,
            f"slope={slope:.3f}\np={pval:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------- マインドワンダリング解析 --------
def compute_out_of_zone_ratio(df: pd.DataFrame, lower: float = 45.0, upper: float = 60.0) -> float:
    """
    各被験者のマインドワンダリング指標（out of the zone）の全試行に対する割合を計算する。
    
    out of the zone の定義:
      lower < |angular_error_target| < upper かつ lower < |angular_error_distractor| < upper
    を満たす試行を 1 (out of the zone) とラベル付けする。
    
    Args:
        df: 単一被験者の連結済みDataFrame（angular_error_target, angular_error_distractor列を含む）
        lower: 下限閾値（デフォルト45度）
        upper: 上限閾値（デフォルト60度）
    
    Returns:
        out of the zone 試行の割合 (0.0 ~ 1.0)
    """
    needed = ["angular_error_target", "angular_error_distractor"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"DataFrame lacks required columns: {needed}")
    
    valid = df.dropna(subset=needed).copy()
    if valid.empty:
        return np.nan
    
    abs_target = valid["angular_error_target"].abs()
    abs_distractor = valid["angular_error_distractor"].abs()
    
    # out of the zone: lower < |error| < upper for BOTH target and distractor
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
    
    Args:
        df: 単一被験者の連結済みDataFrame（angular_error_distractor列を含む）
        max_abs_angle: スケーリング用に想定する最大絶対角度。デフォルトは180度。
    
    Returns:
        distractorに対する平均角度誤差（0〜1にスケールした値）
    """
    if "angular_error_distractor" not in df.columns:
        raise ValueError("DataFrame lacks 'angular_error_distractor' column.")
    if max_abs_angle <= 0:
        raise ValueError("max_abs_angle must be positive.")
    
    valid = df[df["angular_error_target"].abs() > 45]["angular_error_distractor"].dropna()
    if valid.empty:
        return np.nan
    
    mean_abs_error = valid.abs().mean()
    # 角度範囲を max_abs_angle とみなし 0〜1 にスケール（上限をクリップ）
    scaled = mean_abs_error / max_abs_angle
    return len(valid), float(np.clip(scaled, 0.0, 1.0))


def plot_mind_wandering_vs_reward(
    concat_list: List[Tuple[str, pd.DataFrame]],
    save_path: str = None,
    ooz_index: str = "default",
    ooz_fn: Callable[[pd.DataFrame, float, float], float] = None, 
) -> pd.DataFrame:
    """
    各被験者のout of the zone割合と平均獲得報酬の関係をプロットする。
    
    横軸: out of the zone 割合
    縦軸: 平均獲得報酬
    回帰直線、傾き、p値、95%信頼区間を表示。
    
    Args:
        concat_list: (subject_id, DataFrame) のリスト
        lower: out of the zone判定の下限閾値（デフォルト45度）
        upper: out of the zone判定の上限閾値（デフォルト60度）
        save_path: 保存先パス（Noneなら保存しない）
    
    Returns:
        被験者ごとの集計結果DataFrame (subject, out_of_zone_ratio, mean_reward)
    """

    ooz_methods = {
        "default": compute_out_of_zone_ratio,
        "AngularError_distractor": compute_mean_distractor_AngularError_ratio,
        # "alt1": compute_out_of_zone_ratio_alt,  # 追加したい場合ここに足す
    }
    func = ooz_fn or ooz_methods.get(ooz_index)
    if func is None:
        raise ValueError(f"Unknown ooz_index: {ooz_index}")

    rows = []
    for subj_id, df in concat_list:
        try:
            length, out_ratio = func(df)
            valid = df.dropna(subset=["reward_points"])
            mean_reward = valid["reward_points"].mean() if not valid.empty else np.nan
            rows.append({
                "subject": subj_id,
                "out_of_zone_ratio": out_ratio,
                "mean_reward": mean_reward
            })
            print(f"Subject {subj_id}: out_of_zone_ratio={out_ratio:.4f}, mean_reward={mean_reward:.2f}, n_trials={length}")
        except Exception as e:
            print(f"Skipping {subj_id}: {e}")
            continue
    
    result_df = pd.DataFrame(rows).dropna()
    
    if result_df.empty or len(result_df) < 3:
        print("Not enough data for regression analysis.")
        return result_df
    
    # OLS回帰
    X = sm.add_constant(result_df["out_of_zone_ratio"])
    fit = sm.OLS(result_df["mean_reward"], X).fit()
    slope = fit.params.get("out_of_zone_ratio", np.nan)
    pval = fit.pvalues.get("out_of_zone_ratio", np.nan)
    
    # プロット
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="out_of_zone_ratio",
        y="mean_reward",
        data=result_df,
        ax=ax,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95
    )
    
    ax.set_xlabel("Out of the Zone Ratio", fontsize=12)
    ax.set_ylabel("Mean Reward Points", fontsize=12)
    ax.set_title("Mind Wandering vs Reward (Across Subjects)", fontsize=14)
    
    # 統計情報をプロット上に表示
    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(result_df)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()
    
    return result_df
