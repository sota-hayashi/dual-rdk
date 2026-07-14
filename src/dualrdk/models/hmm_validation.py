"""
HMM妥当性検証

仕様: hmm_validation_spec.md

観点A：Recovery Analysis
  既知パラメータからデータを生成し、HMMで復元できるかを確認（48試行の妥当性）

観点B：External Validation
  推定状態系列が外部指標（log_rt）と関係するかを確認（モデルの妥当性）
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────
# 観点A：Recovery Analysis
# ─────────────────────────────────────────────

# 検証する真のパラメータ（3パターン）
TRUE_PARAMS_LIST = {
    "pattern_A": {
        "means":     np.array([[10.0], [35.0]]),
        "covars":    np.array([[[50.0]], [[300.0]]]),
        "transmat":  np.array([[0.95, 0.05],
                               [0.10, 0.90]]),
        "startprob": np.array([0.8, 0.2]),
    },
    "pattern_B": {
        "means":     np.array([[10.0], [30.0]]),
        "covars":    np.array([[[50.0]], [[300.0]]]),
        "transmat":  np.array([[0.90, 0.10],
                               [0.20, 0.80]]),
        "startprob": np.array([0.8, 0.2]),
    },
    "pattern_C": {
        "means":     np.array([[9.5],  [28.5]]),
        "covars":    np.array([[[47.6]], [[325.5]]]),
        "transmat":  np.array([[0.866, 0.134],
                               [0.304, 0.696]]),
        "startprob": np.array([0.8, 0.2]),
    },
}


def _make_generative_model(params: Dict) -> hmm.GaussianHMM:
    """既知パラメータを持つ生成モデルを作成する。"""
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        init_params="",
        params="",  # EMを走らせない（生成専用）
    )
    model.startprob_ = params["startprob"].copy()
    model.transmat_  = params["transmat"].copy()
    model.means_     = params["means"].copy()
    model.covars_    = params["covars"].copy()
    return model


def _fit_recovery_model(
    X: np.ndarray,
    n_iter: int = 200,
    tol: float = 1e-4,
    n_init: int = 10,
) -> hmm.GaussianHMM:
    """
    シミュレートデータにHMMを適用する（グローバルfittingと同じ設定）。
    n_init回の異なるシードで最大対数尤度のモデルを返す。
    """
    best_score = -np.inf
    best_model: Optional[hmm.GaussianHMM] = None

    for seed in range(n_init):
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        try:
            score = model.score(X)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_model = model

    return best_model  # type: ignore[return-value]


def _align_recovered_model(
    model: hmm.GaussianHMM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    μの大小で0=engaged, 1=disengagedに統一したパラメータを返す。

    Returns
    -------
    means_aligned     : (2, 1)
    covars_aligned    : (2, 1, 1)
    transmat_aligned  : (2, 2)
    label_map         : (2,) 元のインデックス順序（engaged, disengaged）
    """
    means_flat = model.means_.flatten()
    engaged_idx    = int(np.argmin(means_flat))
    disengaged_idx = int(np.argmax(means_flat))
    idx_order = [engaged_idx, disengaged_idx]

    means_aligned    = model.means_[idx_order]
    covars_aligned   = model.covars_[idx_order]
    transmat_aligned = model.transmat_[np.ix_(idx_order, idx_order)]

    return means_aligned, covars_aligned, transmat_aligned, np.array(idx_order)


def run_recovery_analysis(
    n_trials: int = 48,
    n_simulations: int = 100,
    true_params_list: Optional[Dict] = None,
    random_seed: int = 0,
) -> Dict:
    """
    Recovery Analysis を実行する。

    Parameters
    ----------
    n_trials       : int   シミュレートする試行数（デフォルト: 48）
    n_simulations  : int   各パターンで繰り返すシミュレーション回数（デフォルト: 100）
    true_params_list : dict or None  検証するパラメータパターン（Noneのとき TRUE_PARAMS_LIST を使用）
    random_seed    : int   再現性のためのシード（デフォルト: 0）

    Returns
    -------
    dict : 仕様書 §2-3 の recovery_results 形式
    """
    if true_params_list is None:
        true_params_list = TRUE_PARAMS_LIST

    rng = np.random.default_rng(random_seed)
    recovery_results: Dict = {}

    for pattern_name, true_params in true_params_list.items():
        print(f"[Recovery Analysis] {pattern_name}...")

        records = []

        for sim in range(n_simulations):
            sim_seed = int(rng.integers(0, 2**31))

            # Step 1: 既知パラメータからデータを生成
            gen_model = _make_generative_model(true_params)
            X_sim, states_true = gen_model.sample(n_samples=n_trials, random_state=sim_seed)
            # X_sim: (n_trials, 1), states_true: (n_trials,)

            # Step 2: シミュレートデータにHMMを適用
            rec_model = _fit_recovery_model(X_sim)
            if rec_model is None:
                continue

            # Step 3: アライメント
            means_al, covars_al, transmat_al, idx_order = _align_recovered_model(rec_model)

            # states_estimated もアライメント
            states_raw = rec_model.predict(X_sim)
            label_map = {int(idx_order[0]): 0, int(idx_order[1]): 1}
            states_estimated = np.vectorize(label_map.get)(states_raw)

            # Step 4: 比較を記録
            accuracy = float(np.mean(states_true == states_estimated))
            # ラベルが逆転している場合は反転した方が高い一致率を採用
            accuracy_flipped = float(np.mean(states_true == (1 - states_estimated)))
            if accuracy_flipped > accuracy:
                accuracy = accuracy_flipped

            records.append({
                "mean_engaged":    float(means_al[0, 0]),
                "mean_disengaged": float(means_al[1, 0]),
                "covar_engaged":   float(covars_al[0, 0, 0]),
                "covar_disengaged":float(covars_al[1, 0, 0]),
                "transmat_EE":     float(transmat_al[0, 0]),
                "transmat_DD":     float(transmat_al[1, 1]),
                "state_accuracy":  accuracy,
            })

        if not records:
            print(f"  Warning: no valid simulations for {pattern_name}")
            continue

        df_rec = pd.DataFrame(records)

        def _stats(key: str, true_val: float) -> Dict:
            vals = df_rec[key].to_numpy()
            est_mean = float(np.mean(vals))
            est_std  = float(np.std(vals, ddof=1))
            return {
                "true":           true_val,
                "estimated_mean": est_mean,
                "estimated_std":  est_std,
                "bias":           est_mean - true_val,
            }

        true = true_params
        recovery_results[pattern_name] = {
            "mean_engaged":     _stats("mean_engaged",     float(true["means"][0, 0])),
            "mean_disengaged":  _stats("mean_disengaged",  float(true["means"][1, 0])),
            "transmat_EE":      _stats("transmat_EE",      float(true["transmat"][0, 0])),
            "transmat_DD":      _stats("transmat_DD",      float(true["transmat"][1, 1])),
            "state_accuracy": {
                "mean": float(df_rec["state_accuracy"].mean()),
                "std":  float(df_rec["state_accuracy"].std(ddof=1)),
            },
        }

        print(f"  state_accuracy: {recovery_results[pattern_name]['state_accuracy']['mean']:.3f} "
              f"± {recovery_results[pattern_name]['state_accuracy']['std']:.3f}")

    return recovery_results


def evaluate_recovery_results(recovery_results: Dict) -> Dict:
    """
    仕様書 §2-5 の判断基準に従って各パターンを評価する。

    Returns
    -------
    dict : パターンごとの判断結果
        {
            "pattern_A": {
                "bias_ok":     bool,   # 全パラメータで |bias| < true * 0.10
                "std_ok":      bool,   # 全パラメータで std < true * 0.20
                "accuracy_ok": bool,   # state_accuracy >= 0.70
                "overall_ok":  bool,   # 3つすべてOK
                "details":     dict,   # 各パラメータの詳細
            },
            ...
        }
    """
    eval_results: Dict = {}

    for pattern_name, result in recovery_results.items():
        param_keys = ["mean_engaged", "mean_disengaged", "transmat_EE", "transmat_DD"]
        details: Dict = {}
        bias_ok_all = True
        std_ok_all  = True

        for key in param_keys:
            p = result[key]
            true_val = p["true"]
            bias_ok = abs(p["bias"]) < abs(true_val) * 0.10
            std_ok  = p["estimated_std"] < abs(true_val) * 0.20
            print(f"  {key}: true={p['true']:.3f}, est_mean={p['estimated_mean']:.3f}, est_std={p['estimated_std']:.3f}"
                  f", bias={p['bias']:.3f} -> bias_ok={bias_ok}, std_ok={std_ok}")
            details[key] = {
                "bias_ok": bias_ok,
                "std_ok":  std_ok,
            }
            if not bias_ok:
                bias_ok_all = False
            if not std_ok:
                std_ok_all = False

        acc_mean = result["state_accuracy"]["mean"]
        accuracy_ok = acc_mean >= 0.70

        eval_results[pattern_name] = {
            "bias_ok":     bias_ok_all,
            "std_ok":      std_ok_all,
            "accuracy_ok": accuracy_ok,
            "overall_ok":  bias_ok_all and std_ok_all and accuracy_ok,
            "details":     details,
            "state_accuracy_mean": acc_mean,
        }

    return eval_results


# ─────────────────────────────────────────────
# 観点B：External Validation
# ─────────────────────────────────────────────

def _build_merged_df(
    hmm_results: List[Dict],
    df_original: pd.DataFrame,
) -> pd.DataFrame:
    """
    HMM結果（viterbi_states）と元DataFrameのlog_rtを
    participant_id × trial でマージする。

    df_original には以下の列が必要:
        participant_id, trial（0-indexed）, log_rt, abs_angular_error
    """
    rows = []
    for r in hmm_results:
        pid    = r["participant_id"]
        states = r["viterbi_states"]  # shape (n_trials,)
        n      = len(states)
        for t_idx, state in enumerate(states):
            rows.append({
                "subject": pid,
                "trial":          t_idx,
                "hmm_state":      int(state),
            })

    df_states = pd.DataFrame(rows)

    # log_rt が未計算の場合は計算する
    if "log_rt" not in df_original.columns and "rt" in df_original.columns:
        df_original = df_original.dropna(subset=["rt"])
        df_original["log_rt"] = np.log(df_original["rt"])

    # trial 列の型を揃える
    df_states["trial"]    = df_states["trial"].astype(int)
    df_original           = df_original.copy()
    df_original["trial"]  = df_original["num_trial"].astype(int)

    df_merged = pd.merge(
        df_states,
        df_original[["subject", "trial", "log_rt", "abs_angular_error"]
                     if "abs_angular_error" in df_original.columns
                     else ["subject", "trial", "log_rt"]],
        on=["subject", "trial"],
        how="inner",
    )

    return df_merged


def _compute_local_cv_rt(
    df_merged: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    各参加者ごとに局所的なCV(rt)（前後2試行の移動SD / 移動mean）を計算する。
    NaNが生じる端の試行は除外する。
    """
    dfs = []
    for pid, group in df_merged.groupby("subject"):
        g = group.sort_values("trial").copy()
        # log_rt の rolling を rt スケールで計算するため exp に戻す
        rt_raw = np.exp(g["log_rt"])
        rolling_std  = rt_raw.rolling(window, center=True).std()
        rolling_mean = rt_raw.rolling(window, center=True).mean()
        g["local_cv_rt"] = rolling_std / rolling_mean
        dfs.append(g)

    return pd.concat(dfs, ignore_index=True)


def _cohen_d_from_groups(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d（Welch風の分母）を返す。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _mixed_effects_p_value(
    df: pd.DataFrame,
    dependent: str,
    group_col: str = "hmm_state",
    subject_col: str = "subject",
) -> float:
    """
    混合線形モデルを使って、hmm_state の効果のp値を返す。
    statsmodels が使えない場合は被験者内対応のt検定で代替する。
    """
    try:
        import statsmodels.formula.api as smf
        formula = f"{dependent} ~ C({group_col})"
        model = smf.mixedlm(formula, df.dropna(subset=[dependent]), groups=df.dropna(subset=[dependent])[subject_col])
        result = model.fit(reml=False, disp=False)
        # C(hmm_state)[T.1] の p 値
        pval = float(result.pvalues.get(f"C({group_col})[T.1]", np.nan))
        return pval
    except Exception:
        pass

    # フォールバック：参加者ごとの平均差でペアt検定
    engaged    = df[df[group_col] == 0].groupby(subject_col)[dependent].mean().dropna()
    disengaged = df[df[group_col] == 1].groupby(subject_col)[dependent].mean().dropna()
    common_subs = engaged.index.intersection(disengaged.index)
    if len(common_subs) < 2:
        return float("nan")
    _, pval = scipy_stats.ttest_rel(
        disengaged.loc[common_subs].values,
        engaged.loc[common_subs].values,
    )
    return float(pval)


def _trial_level_validation(df_cv: pd.DataFrame) -> Dict:
    """
    試行レベルの検証:
    engaged / disengaged 試行のlog_rt・local_cv_rtの差を検定する。
    """
    engaged_logrt    = df_cv[df_cv["hmm_state"] == 0]["log_rt"].dropna().to_numpy()
    disengaged_logrt = df_cv[df_cv["hmm_state"] == 1]["log_rt"].dropna().to_numpy()

    p_logrt = _mixed_effects_p_value(df_cv, "log_rt")
    d_logrt = _cohen_d_from_groups(disengaged_logrt, engaged_logrt)

    # local_cv_rt が計算されているかチェック
    df_cv_nonan = df_cv.dropna(subset=["local_cv_rt"])
    engaged_cv    = df_cv_nonan[df_cv_nonan["hmm_state"] == 0]["local_cv_rt"].to_numpy()
    disengaged_cv = df_cv_nonan[df_cv_nonan["hmm_state"] == 1]["local_cv_rt"].to_numpy()

    p_cv = _mixed_effects_p_value(df_cv_nonan, "local_cv_rt")
    d_cv = _cohen_d_from_groups(disengaged_cv, engaged_cv)

    return {
        "log_rt_by_state": {
            "engaged_mean":    float(np.mean(engaged_logrt))    if len(engaged_logrt)    > 0 else float("nan"),
            "disengaged_mean": float(np.mean(disengaged_logrt)) if len(disengaged_logrt) > 0 else float("nan"),
            "effect_size":     d_logrt,
            "p_value":         p_logrt,
        },
        "local_cv_rt_by_state": {
            "engaged_mean":    float(np.mean(engaged_cv))    if len(engaged_cv)    > 0 else float("nan"),
            "disengaged_mean": float(np.mean(disengaged_cv)) if len(disengaged_cv) > 0 else float("nan"),
            "effect_size":     d_cv,
            "p_value":         p_cv,
        },
    }


def _participant_level_validation(
    df_merged: pd.DataFrame,
    df_original: pd.DataFrame,
) -> Dict:
    """
    参加者レベルの検証:
    disengaged割合とCV(rt)・学習指標の相関を計算する。
    """
    # 各参加者のdisengaged割合
    disengaged_rate = (
        df_merged.groupby("subject")["hmm_state"]
        .apply(lambda x: float(np.mean(x == 1)))
        .rename("disengaged_rate")
    )

    result: Dict = {}

    # CV(rt) との相関
    if "rt" in df_original.columns:
        cv_rt = (
            df_original.groupby("subject")["rt"]
            .apply(lambda x: float(x.std(ddof=1) / x.mean()) if len(x) > 1 else float("nan"))
            .rename("cv_rt")
        )
        merged = pd.concat([disengaged_rate, cv_rt], axis=1).dropna()
        if len(merged) >= 3:
            r, p = scipy_stats.pearsonr(merged["disengaged_rate"], merged["cv_rt"])
            result["disengaged_rate_vs_cv_rt"] = {"r": float(r), "p_value": float(p)}
        else:
            result["disengaged_rate_vs_cv_rt"] = {"r": float("nan"), "p_value": float("nan")}
    else:
        result["disengaged_rate_vs_cv_rt"] = {"r": float("nan"), "p_value": float("nan")}

    # 学習指標（target_choice_rate_diff）との相関
    if "target_choice_rate_diff" in df_original.columns:
        learning = (
            df_original.groupby("subject")["target_choice_rate_diff"]
            .mean()
            .rename("target_choice_rate_diff")
        )
        merged2 = pd.concat([disengaged_rate, learning], axis=1).dropna()
        if len(merged2) >= 3:
            r2, p2 = scipy_stats.pearsonr(merged2["disengaged_rate"], merged2["target_choice_rate_diff"])
            result["disengaged_rate_vs_learning"] = {"r": float(r2), "p_value": float(p2)}
        else:
            result["disengaged_rate_vs_learning"] = {"r": float("nan"), "p_value": float("nan")}
    else:
        result["disengaged_rate_vs_learning"] = {"r": float("nan"), "p_value": float("nan")}

    return result


def _transition_level_validation(df_merged: pd.DataFrame) -> Dict:
    """
    状態遷移レベルの検証:
    遷移直後の試行のlog_rtをベースライン（全試行の平均）と比較する。
    """
    baseline_logrt = df_merged["log_rt"].dropna().mean()

    logrt_after_to_disengaged = []
    logrt_after_to_engaged    = []

    for pid, group in df_merged.groupby("subject"):
        g = group.sort_values("trial").reset_index(drop=True)
        states = g["hmm_state"].to_numpy()
        logrt  = g["log_rt"].to_numpy()

        for t in range(1, len(states)):
            if np.isnan(logrt[t]):
                continue
            prev, curr = states[t - 1], states[t]
            if prev == 0 and curr == 1:  # engaged → disengaged
                logrt_after_to_disengaged.append(logrt[t])
            elif prev == 1 and curr == 0:  # disengaged → engaged
                logrt_after_to_engaged.append(logrt[t])

    def _vs_baseline(vals: List[float]) -> Dict:
        if len(vals) < 2:
            return {"mean": float("nan"), "vs_baseline_p_value": float("nan")}
        arr = np.array(vals)
        # 1サンプルt検定（ベースライン平均との比較）
        _, p = scipy_stats.ttest_1samp(arr, baseline_logrt, nan_policy="omit")
        return {"mean": float(np.mean(arr)), "vs_baseline_p_value": float(p)}

    return {
        "log_rt_after_transition_to_disengaged": _vs_baseline(logrt_after_to_disengaged),
        "log_rt_after_transition_to_engaged":    _vs_baseline(logrt_after_to_engaged),
    }


def run_external_validation(
    hmm_results: List[Dict],
    df_original: pd.DataFrame,
    cv_window: int = 5,
) -> Dict:
    """
    External Validation を実行する。

    Parameters
    ----------
    hmm_results   : List[Dict]   run_gaussian_hmm() の出力
    df_original   : DataFrame    元のDataFrame（log_rtまたはrt列を含む）
    cv_window     : int          局所CV計算のウィンドウ幅（デフォルト: 5）

    Returns
    -------
    dict : 仕様書 §3-3 の validation_results 形式
    """
    # Step 1: 状態系列とlog_rtを試行レベルで結合
    df_merged = _build_merged_df(hmm_results, df_original)
    print(f"[External Validation] Merged {len(df_merged)} trial rows across "
          f"{df_merged['subject'].nunique()} participants.")

    # Step 2: 局所的なCV(rt)を計算
    df_cv = _compute_local_cv_rt(df_merged, window=cv_window)

    # Step 3: 試行レベルの検証
    trial_level = _trial_level_validation(df_cv)
    print(f"  log_rt: engaged={trial_level['log_rt_by_state']['engaged_mean']:.4f}, "
          f"disengaged={trial_level['log_rt_by_state']['disengaged_mean']:.4f}, "
          f"d={trial_level['log_rt_by_state']['effect_size']:.3f}, "
          f"p={trial_level['log_rt_by_state']['p_value']:.4f}")

    # Step 4: 参加者レベルの検証
    participant_level = _participant_level_validation(df_merged, df_original)

    # Step 5: 遷移レベルの検証
    transition_level = _transition_level_validation(df_merged)
    print(f"  Transitions to disengaged: n={len([])}, "
          f"mean_logrt={transition_level['log_rt_after_transition_to_disengaged']['mean']:.4f}")

    return {
        "trial_level":       trial_level,
        "participant_level": participant_level,
        "transition_level":  transition_level,
    }


# ─────────────────────────────────────────────
# 統合判断
# ─────────────────────────────────────────────

def print_validation_summary(
    recovery_eval: Dict,
    validation_results: Dict,
) -> None:
    """
    仕様書 §4 の統合判断に基づいて結果を表示する。
    """
    recovery_ok  = all(v["overall_ok"] for v in recovery_eval.values())
    external_ok  = (
        validation_results["trial_level"]["log_rt_by_state"]["p_value"] < 0.05
        and validation_results["trial_level"]["log_rt_by_state"]["effect_size"] > 0.3
    )

    print("\n" + "=" * 60)
    print("HMM Validation Summary")
    print("=" * 60)

    print("\n[Recovery Analysis]")
    for pattern_name, ev in recovery_eval.items():
        status = "OK" if ev["overall_ok"] else "NG"
        print(f"  {pattern_name}: {status} "
              f"(bias_ok={ev['bias_ok']}, std_ok={ev['std_ok']}, "
              f"accuracy_ok={ev['accuracy_ok']}, "
              f"state_acc={ev['state_accuracy_mean']:.3f})")

    print("\n[External Validation]")
    tl = validation_results["trial_level"]
    pl = validation_results["participant_level"]
    tr = validation_results["transition_level"]
    print(f"  Trial level  - log_rt effect: d={tl['log_rt_by_state']['effect_size']:.3f}, "
          f"p={tl['log_rt_by_state']['p_value']:.4f}")
    print(f"  Trial level  - CV(rt) effect: d={tl['local_cv_rt_by_state']['effect_size']:.3f}, "
          f"p={tl['local_cv_rt_by_state']['p_value']:.4f}")
    print(f"  Participant  - disengaged vs CV(rt):  r={pl['disengaged_rate_vs_cv_rt']['r']:.3f}, "
          f"p={pl['disengaged_rate_vs_cv_rt']['p_value']:.4f}")
    print(f"  Participant  - disengaged vs learning: r={pl['disengaged_rate_vs_learning']['r']:.3f}, "
          f"p={pl['disengaged_rate_vs_learning']['p_value']:.4f}")
    print(f"  Transition   - to disengaged: mean={tr['log_rt_after_transition_to_disengaged']['mean']:.4f}, "
          f"p={tr['log_rt_after_transition_to_disengaged']['vs_baseline_p_value']:.4f}")
    print(f"  Transition   - to engaged:    mean={tr['log_rt_after_transition_to_engaged']['mean']:.4f}, "
          f"p={tr['log_rt_after_transition_to_engaged']['vs_baseline_p_value']:.4f}")

    print("\n[統合判断]")
    if recovery_ok and external_ok:
        print("  → 両方OK: HMMの結果を信頼して解析を進める")
    elif not recovery_ok and external_ok:
        print("  → RecoveryのみNG: 推定の不安定性を明示した上でExternal Validationの結果を報告する（探索的解析）")
    elif recovery_ok and not external_ok:
        print("  → ExternalのみNG: モデルの仮定が現実と合っていない → 状態数・入力変数の変更を検討")
    else:
        print("  → 両方NG: HMMの適用を断念し、変化点検出など別の手法を検討する")

    print("=" * 60)
