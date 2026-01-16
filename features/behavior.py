from typing import Dict, List, Tuple, Iterable
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm

from io_data.utils import combine_subjects
from stats.metrics import permutation_sign_test, cmh_test_2x2


def summarize_chosen_item_errors(concat_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    被験者×chosen_itemごとに、選択に応じた角度誤差の平均と分散を算出する。
    chosen_item=1 -> angular_error_target
    chosen_item=0 -> angular_error_distractor
    """
    combined = combine_subjects(concat_list)
    if combined.empty:
        return pd.DataFrame()

    work = combined[combined["chosen_item"].isin([0, 1])].copy()
    work["error_value"] = np.where(
        work["chosen_item"] == 1,
        work["angular_error_target"],
        work["angular_error_distractor"],
    )
    work = work.dropna(subset=["subject", "error_value"])

    result = (
        work.groupby(["subject", "chosen_item"])["error_value"]
        .agg(
            mean_abs_error=lambda x: x.abs().mean(),
            std_error=lambda x: x.std(ddof=1),
            n="count",
        )
        .reset_index()
    )
    return result

def calculate_rt_moving_mean(
    concat_list: List[Tuple[str, pd.DataFrame]],
    window: int = 3
) -> List[Tuple[str, pd.DataFrame]]:
    """
    直前試行のRTを使った移動平均 M_t を計算する。
    M_t = mean(RT_{t-1}, RT_{t-2}, RT_{t-3})
    """
    updated = []
    for subj_id, df in concat_list:
        work = df.copy()
        if "rt" not in work.columns:
            raise ValueError("DataFrame lacks 'rt' column.")
        rt = pd.to_numeric(work["rt"], errors="coerce")
        work["rt_moving_mean"] = rt.shift(1).rolling(window=window, min_periods=1).mean()
        updated.append((subj_id, work))
    return updated


def calculate_rt_deviance_mean(
    concat_list: List[Tuple[str, pd.DataFrame]],
    window: int = 3
) -> List[Tuple[str, pd.DataFrame]]:
    """
    RTの平均との差の絶対値 D_t を計算し、直前試行で平滑化する。
    D_t_smooth = mean(D_{t-1}, D_{t-2}, D_{t-3})
    """
    updated = []
    for subj_id, df in concat_list:
        work = df.copy()
        if "rt" not in work.columns:
            raise ValueError("DataFrame lacks 'rt' column.")
        rt = pd.to_numeric(work["rt"], errors="coerce")
        rt_mean = rt.mean()
        dev = (rt - rt_mean).abs()
        work["rt_deviance"] = dev
        work["rt_deviance_mean"] = dev.shift(1).rolling(window=window, min_periods=1).mean()
        updated.append((subj_id, work))
    return updated


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
    # work = work[work["num_trial"] >= 16]
    if work.empty:
        return pd.DataFrame()

    abs_t = work["angular_error_target"].abs()
    abs_d = work["angular_error_distractor"].abs()

    def infer_choice(row):
        if row["chosen_item"] == 1:
            return row["target_group"]  # ターゲット色
        if row["chosen_item"] == 0:
            return "white" if row["target_group"] == "black" else "black"  # 反対色=ディストラクター色
        return np.nan  # -1は除外


    work["chosen_color"] = work.apply(infer_choice, axis=1)
    work["chosen_error"] = np.where(
        work["chosen_color"] == work["target_group"],
        work["angular_error_target"],
        work["angular_error_distractor"],
    )

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


def analyze_color_accuracy_change_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    alpha: float = 0.05
) -> Dict[str, object]:
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

    ww = len(valid[(valid["aftered_color"] == "white") & (valid["target_color"] == "white")])
    wb = len(valid[(valid["aftered_color"] == "white") & (valid["target_color"] == "black")])
    bw = len(valid[(valid["aftered_color"] == "black") & (valid["target_color"] == "white")])
    bb = len(valid[(valid["aftered_color"] == "black") & (valid["target_color"] == "black")])

    contingency = np.array([[ww, wb], [bw, bb]])
    chi2, p, dof, expected = chi2_contingency(contingency)

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


def analyze_same_diff_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_perm: int = 5000
) -> Dict[str, object]:
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
        per_subject.append({
            "subject": subj_id,
            "same": same_diff["same"],
            "different": same_diff["different"],
            "prop_same": prop_same
        })
    if not proportions:
        return {"per_subject": per_subject, "test": None}
    test = permutation_sign_test(np.array(proportions), center=0.5, n_perm=n_perm, random_state=123)
    return {"per_subject": per_subject, "test": test}


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


def analyze_target_choice_learning(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, logistic regression of is_target ~ num_trial.
    Returns slope and p-value (Wald).
    """
    rows = []
    for sess, sub in df.groupby("num_session"):
        if sub["is_target"].nunique() < 2:
            rows.append({
                "num_session": sess,
                "beta_trial": np.nan,
                "p_value": np.nan,
                "n": len(sub),
                "note": "no variance"
            })
            continue
        X = sm.add_constant(sub["num_trial"])
        try:
            fit = sm.Logit(sub["is_target"], X).fit(disp=False)
            beta = fit.params.get("num_trial", np.nan)
            pval = fit.pvalues.get("num_trial", np.nan)
            rows.append({
                "num_session": sess,
                "beta_trial": beta,
                "p_value": pval,
                "n": len(sub),
                "note": ""
            })
        except Exception as e:
            rows.append({
                "num_session": sess,
                "beta_trial": np.nan,
                "p_value": np.nan,
                "n": len(sub),
                "note": f"fit error: {e}"
            })
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

def categorize_subjects_from_hmm_summary(
    hmm_summary: pd.DataFrame,
    state_label_col: str = "states",
    switch_count_label_col: str = "switch_count",
    frac_exploit_label_col: str = "frac_exploit"
) -> pd.DataFrame:
    """
    HMMの状態ラベルと各状態の割合に基づいて、被験者をカテゴリ分けする。
    """
    def categorize(states: List[str], switch_count: int, frac_exploit: float) -> str:
        if states[0] == 0 and states[-1] == 1 and switch_count == 1:
            return "explore-to-exploit"
        elif states[0] == 1 and switch_count == 0:
            return "immediate-exploit"
        elif switch_count > 1:
            return "explore-exploit-cycling"
        else:
            return "other"

    categorized = hmm_summary.copy()
    categorized["category"] = categorized.apply(lambda row: categorize(row[state_label_col], row[switch_count_label_col], row[frac_exploit_label_col]), axis=1)
    return categorized
