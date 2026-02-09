from typing import Dict, List, Tuple
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
    needed = ["angular_error_target", "angular_error_distractor", "target_group", "reward_points", "chosen_item"]
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
                "mean_target_choice": sub["chosen_item"].mean(),
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
            "mean_target_choice": sub["chosen_item"].mean(),
            "mean_reward": sub["reward_points"].mean(),
            "mean_abs_error": sub["chosen_error"].abs().mean(),
            "slope": slope,
            "p_value": pval,
            "note": ""
        })

    return pd.DataFrame(rows)


def compute_exploit_target_prob_by_switch(
    subject_states_list: List[Tuple[str, str, np.ndarray, dict, np.ndarray]]
) -> List[Tuple[str, List[float]]]:
    """
    各被験者について、explore -> exploit の切り替え後の exploit 区間ごとに
    target(=1)選択確率を算出する。-1 は平均から除外するが系列は崩さない。
    Returns: [(subject_id, [prob1, prob2, ...]), ...]
    """
    results = []
    for subj_id, category, states, state_labels, observations in subject_states_list:
        if not isinstance(state_labels, dict):
            raise ValueError("state_labels must be a dict like {state_id: label}")

        label_to_state = {v: k for k, v in state_labels.items()}
        if "explore" not in label_to_state or "exploit" not in label_to_state:
            print(f"Skipping {subj_id}: missing explore/exploit labels")
            continue

        explore_state = label_to_state["explore"]
        exploit_state = label_to_state["exploit"]

        obs = np.array(observations)
        st = np.array(states)
        if len(obs) != len(st):
            raise ValueError(f"{subj_id}: observations and states length mismatch")

        probs = []
        t = 1
        while t < len(st):
            if st[t - 1] == exploit_state and st[t] == exploit_state:
                start = t - 1
                end = t
                while end < len(st) and st[end] == exploit_state:
                    end += 1
                seg = obs[start:end]
                valid = seg[seg != -1] # -1（ターゲット・ディストラクター以外を選択） を除外
                if valid.size == 0:
                    probs.append(np.nan)
                else:
                    probs.append(float(np.mean(valid == 1)))
                t = end
            elif st[t - 1] == explore_state and st[t] == exploit_state:
                start = t
                end = t + 1
                while end < len(st) and st[end] == exploit_state:
                    end += 1
                seg = obs[start:end]
                valid = seg[seg != -1] # -1（ターゲット・ディストラクター以外を選択） を除外
                if valid.size == 0:
                    probs.append(np.nan)
                else:
                    probs.append(float(np.mean(valid == 1)))
                t = end
            else:
                t += 1

        results.append((subj_id, probs))

    return results


def compute_target_choice_rate_per_subject(
    all_df_awareness: List[Tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    各被験者のターゲット選択確率（chosen_item==1）を算出する。
    chosen_item は 0/1 のみを対象にする。
    """
    rows = []
    for subj_id, df in all_df_awareness:
        if "chosen_item" not in df.columns:
            raise ValueError("DataFrame lacks 'chosen_item' column.")
        valid = df[df["chosen_item"].isin([0, 1])]
        if valid.empty:
            rate = np.nan
        else:
            rate = float((valid["chosen_item"] == 1).mean())
        rows.append({"subject": subj_id, "target_choice_rate": rate})

    result_df = pd.DataFrame(rows)
    # for _, row in result_df.iterrows():
        # print(f"{row['subject']}: {row['target_choice_rate']}")
    return result_df


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

def relabel_hmm_states(hmm_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel HMM states based on a global criterion and recalculate summaries.

    Args:
        hmm_results_df: A DataFrame loaded from the HMM summary, 
                        containing columns like 'B', 'state_labels', 'states'.

    Returns:
        A DataFrame with summaries where states are aligned across subjects.
    """
    if hmm_results_df.empty:
        return pd.DataFrame()

    # 1. 全被験者のB行列を平均し、グローバルな基準を決定
    all_b_matrices = hmm_results_df["B"].tolist()
    if not all_b_matrices:
        return pd.DataFrame()
    
    b_avg = np.mean(all_b_matrices, axis=0)
    
    # big_AE_cat = 0 (大きなエラー) の放出確率が高い方を「探索」状態とする
    big_ae_cat = 0
    global_explore_idx = int(np.argmax(b_avg[:, big_ae_cat])) # off
    global_exploit_idx = 1 - global_explore_idx # on        

    # 2. 各被験者の結果をループし、必要ならラベルを反転・再計算
    recalculated_summaries = []
    for index, row in hmm_results_df.iterrows():
        subj_id = row["subject"]
        
        # ローカル（この被験者だけ）の探索状態のインデックスを取得
        # 'state_labels' is like {0: 'explore', 1: 'exploit'}
        state_map = row["state_labels"]
        local_explore_idx = [k for k, v in state_map.items() if v == 'explore'][0]
        
        states = row["states"]
        
        # ラベルがグローバル基準と不一致なら状態系列を反転
        if local_explore_idx != global_explore_idx:
            # 0を1に、1を0に反転
            states = 1 - np.array(states)
        
        # 新しい(正規化された)statesに基づいて統計量を再計算
        frac_exploit = float(np.mean(states == global_exploit_idx))
        switch_count = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0

        category = "off-to-off"
        if len(states) > 1:
            # 動的に決定されたインデックスを使用する
            if states[0] == global_explore_idx and states[-1] == global_exploit_idx and switch_count == 1:
                category = "off-to-on"
            elif states[0] == global_exploit_idx and switch_count == 0:
                category = "on-to-on"
            elif switch_count > 1:
                category = "on-off-cycling"
            elif states[0] == global_exploit_idx and states[-1] == global_explore_idx and switch_count == 1:
                category = "on-to-off"
        
        # run_lengthsの再計算
        runs = {k: [] for k in range(2)}
        if len(states) > 0:
            current = states[0]
            length = 1
            for s in states[1:]:
                if s == current:
                    length += 1
                else:
                    runs[current].append(length)
                    current = s
                    length = 1
            runs[current].append(length)
        
        mean_run_explore = np.mean(runs.get(global_explore_idx, [])) if runs.get(global_explore_idx) else np.nan
        mean_run_exploit = np.mean(runs.get(global_exploit_idx, [])) if runs.get(global_exploit_idx) else np.nan
        global_state_labels = {
            global_explore_idx: "explore",
            global_exploit_idx: "exploit"
        }

        summary = {
            "subject": subj_id,
            "frac_exploit": frac_exploit,
            "switch_count": switch_count,
            "mean_run_explore": mean_run_explore,
            "mean_run_exploit": mean_run_exploit,
            "A": json.dumps(row["A"].tolist()),
            "B": json.dumps(row["B"].tolist()),
            "pi": json.dumps(row["pi"].tolist()),
            "state_labels": global_state_labels,
            "states": json.dumps(states.tolist()),
            "observations": json.dumps(row["observations"].tolist()),
            "mapped_observations": json.dumps(row["mapped_observations"].tolist()),
            "category": category,
            "loglik": row["loglik"],
        }
        recalculated_summaries.append(summary)
        
    return pd.DataFrame(recalculated_summaries)
