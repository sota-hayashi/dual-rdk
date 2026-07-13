import numpy as np
import pandas as pd
from functools import reduce
from scipy import stats

from features.behavior import (
    cancatenate_necessary_behavioral_df,
)
from stats.metrics import (
    t_test_reward_points_between_periods,
    t_test_count_target_choice_between_periods,
    t_test_rt_difference_between_target_distractor,
    t_test_Target_Angular_Error_between_periods,
    t_test_Minimum_Angular_Error_between_periods,
)
from stats.models import (
    evaluate_q_learning,
    run_q_learning_ooz,
    run_directional_q_learning,
    evaluate_directional_q_learning,
)
from stats.q_learning_ooz_map import fit_q_learning_ooz_map_group
from viz.plots import (
    plot_exp_obj_with_linear_fit,
)


def get_subjects_by_behavior_data(
    all_data_learning,
    threshold: float = 0.8
):
    """
    行動指標に基づいて被験者を抽出する。
    rt_cv の上位10%を分離する。
    """
    behavioral_df = cancatenate_necessary_behavioral_df(all_data_learning)

    subjects_behavior_1 = behavioral_df.loc[
        (behavioral_df["rt_cv"] < behavioral_df["rt_cv"].quantile(0.90)),
        "subject"
    ].tolist()
    subjects_behavior_2 = behavioral_df.loc[
        (behavioral_df["rt_cv"] >= behavioral_df["rt_cv"].quantile(0.90)),
        "subject"
    ].tolist()

    return subjects_behavior_1, subjects_behavior_2, behavioral_df


# =============================================================================
# 前処理: データ準備
# =============================================================================

def prepare_analysis_df(
    all_data_learning,
    behavioral_df=None,
):
    """
    行動指標・Q-learningパラメータを統合した解析用DataFrameを構築する。

    Returns
    -------
    all_df : pd.DataFrame
        behavioral + Q-learning を結合した解析用DataFrame
    behavioral_df : pd.DataFrame
        行動指標の要約DataFrame
    results_df : pd.DataFrame
        Q-learning fitting結果
    metrics : pd.DataFrame
        Q-learningモデル評価指標
    """
    # Q-learning fitting
    results_df, _ = run_q_learning_ooz(fit=False, concat_list=all_data_learning)
    metrics = evaluate_q_learning(
        method="ooz_map",
        concat_list=all_data_learning,
        results_df=results_df,
        n_params=4,
    )
    # results_df, _ = run_directional_q_learning(fit=False, concat_list=all_data_learning)
    # metrics = evaluate_directional_q_learning(
    #     concat_list=all_data_learning, 
    #     results_df=results_df, 
    #     n_params=4,
    # )

    # 行動指標
    if behavioral_df is None:
        behavioral_df = cancatenate_necessary_behavioral_df(all_data_learning)

    # DataFrame結合
    results_df_slim = results_df.loc[:, ["subject", "alpha_0", "alpha_1", "beta", "c_black"]]
    data_frames = [behavioral_df, results_df_slim]
    all_df = reduce(
        lambda left, right: pd.merge(left, right, on='subject', how='inner'),
        data_frames,
    )

    # 派生特徴量
    all_df["alpha_diff"] = np.log(all_df["alpha_1"] / all_df["alpha_0"])
    all_df["ooz_rate"] = all_df["ooz"].apply(
        lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else np.nan
    )

    return all_df, behavioral_df, results_df, metrics


# =============================================================================
# 解析: 統計検定
# =============================================================================

def run_statistical_tests(all_data_learning, all_df):
    """
    統計検定を実行し、結果を辞書で返す。

    Returns
    -------
    dict
        テスト名をキーとした結果辞書
    """
    results = {}

    results["target_choice_between_periods"] = (
        t_test_count_target_choice_between_periods(all_data_learning)
    )
    results["reward_between_periods"] = (
        t_test_reward_points_between_periods(all_data_learning)
    )
    results["min_angular_error_between_periods"] = (
        t_test_Minimum_Angular_Error_between_periods(all_data_learning)
    )
    results["target_angular_error_between_periods"] = (
        t_test_Target_Angular_Error_between_periods(all_data_learning)
    )
    results["rt_target_vs_distractor"] = (
        t_test_rt_difference_between_target_distractor(all_data_learning)
    )

    # OOZ rate の one-sample t-test
    alpha_diffs = all_df["alpha_diff"].dropna().tolist()
    t_stat, p_value = stats.ttest_1samp(alpha_diffs, popmean=0)
    results["alpha_diff_one_sample"] = {"mean": np.mean(alpha_diffs), "sd": np.std(alpha_diffs, ddof=1), "t_stat": t_stat, "p_value": p_value}

    return results


def _print_period_test(name, results):
    """前半/後半比較のt検定結果を表示する。"""
    print(f"\nT-test: {name}")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )


def _print_test_results(test_results):
    """全統計検定結果をフォーマットして表示する。"""
    period_tests = [
        ("Target choice count between early and late periods", "target_choice_between_periods"),
        ("Reward points between early and late periods", "reward_between_periods"),
        ("Minimum Angular Error between early and late periods", "min_angular_error_between_periods"),
        ("Target Angular Error between early and late periods", "target_angular_error_between_periods"),
    ]
    for label, key in period_tests:
        _print_period_test(label, test_results[key])

    # RT target vs distractor
    r = test_results["rt_target_vs_distractor"]
    print("\nT-test: RT difference between target and distractor choices")
    print(f"n_target = {r['n_target']}, n_distractor = {r['n_distractor']}, T-statistic: {r['t_stat']}, P-value: {r['p_value']}")
    print(
        f"Mean RT (target choices): {r['mean_target_rt']} ms, "
        f"Std RT (target choices): {r['std_target_rt']} ms, "
        f"\nMean RT (distractor choices): {r['mean_distractor_rt']} ms, "
        f"Std RT (distractor choices): {r['std_distractor_rt']} ms, "
        f"\nMean RT difference (target - distractor): {r['mean_diff_rt']} ms"
    )

    # Alpha diff one-sample t-test
    r = test_results["alpha_diff_one_sample"]
    print(f"\nAlpha diff one-sample t-test: μ={r['mean']:.4f}, σ={r['sd']:.4f}, t={r['t_stat']:.4f}, p={r['p_value']:.4f}")


# =============================================================================
# 解析: 要約統計
# =============================================================================

def compute_summary_statistics(all_df):
    """
    解析用DataFrameから要約統計量を計算する。

    Returns
    -------
    dict
        ooz_rates, combined_variances, quartile_stats を含む辞書
    """
    ooz_rates = all_df["ooz_rate"].dropna().tolist()
    combined_variances = list(zip(
        all_df["alpha_diff"].tolist(),
        all_df["alpha_diff_directional"].tolist(),
    ))

    arr = np.array(ooz_rates)
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1

    quartile_stats = {
        "Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr,
        "minimum": np.min(arr), "maximum": np.max(arr),
        "lower_fence": q1 - 1.5 * iqr,
        "upper_fence": q3 + 1.5 * iqr,
    }

    return {
        "ooz_rates": ooz_rates,
        "combined_variances": combined_variances,
        "quartile_stats": quartile_stats,
    }


# =============================================================================
# 解析: ブートストラップによる群レベル alpha_diff 検定
# =============================================================================

def run_bootstrap_alpha_diff(
    all_data_learning,
    n_samples=30,
    n_iterations=100,
    random_state=None,
):
    """
    ランダムサンプリング + 群レベルQ-learning推定によるalpha_diffの検定。

    各イテレーションで n_samples 人をランダムに抽出し、
    群全体で共有するalpha_0, alpha_1をMAP推定して alpha_diff を得る。
    全イテレーションの alpha_diffs に対して1標本t検定 (H0: mean = 0) を行う。

    Parameters
    ----------
    all_data_learning : list of (subject_id, DataFrame)
    n_samples : int
        各イテレーションでサンプリングする参加者数
    n_iterations : int
        リサンプリングの繰り返し回数
    random_state : int or None
        再現性のための乱数シード

    Returns
    -------
    dict
        alpha_diffs: list of float (各イテレーションの alpha_diff)
        t_stat, p_value: 1標本t検定の結果
        mean, std, ci_95: alpha_diff の要約統計
    """
    rng = np.random.default_rng(random_state)
    n_total = len(all_data_learning)

    if n_samples > n_total:
        raise ValueError(
            f"n_samples ({n_samples}) > total participants ({n_total})"
        )

    alpha_diffs = []
    fit_summaries = []
    for i in range(n_iterations):
        indices = rng.choice(n_total, size=n_samples, replace=False)
        sampled = [all_data_learning[j] for j in indices]

        result = fit_q_learning_ooz_map_group(sampled)
        alpha_diffs.append(result["alpha_diff"])
        fit_summaries.append(result["fit_summary"])

        pct = (i + 1) / n_iterations
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {i+1}/{n_iterations} "
            f"alpha_diff={result['alpha_diff']:+.4f} "
            f"mean_LL={result['fit_summary']['mean_ll']:.2f}",
            end="", flush=True,
        )
    print()

    alpha_diffs = np.array(alpha_diffs)
    t_stat, p_value = stats.ttest_1samp(alpha_diffs, popmean=0)
    mean_diff = np.mean(alpha_diffs)
    std_diff = np.std(alpha_diffs, ddof=1)
    ci_95 = (
        mean_diff - 1.96 * std_diff / np.sqrt(n_iterations),
        mean_diff + 1.96 * std_diff / np.sqrt(n_iterations),
    )

    # 全イテレーションにわたる適合度の集約
    all_mean_ll = [s["mean_ll"] for s in fit_summaries]
    all_min_ll = [s["min_ll"] for s in fit_summaries]
    all_max_ll = [s["max_ll"] for s in fit_summaries]

    print(f"\n=== Bootstrap alpha_diff results ===")
    print(f"n_iterations={n_iterations}, n_samples={n_samples}")
    print(f"Mean alpha_diff: {mean_diff:.4f} (SD={std_diff:.4f})")
    print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"One-sample t-test (H0: mean=0): μ={mean_diff:.4f}, t={t_stat:.4f}, p={p_value:.4f}")
    print(f"\n--- Per-participant fit (across iterations) ---")
    print(f"Mean LL:  avg={np.mean(all_mean_ll):.2f}, min={np.min(all_mean_ll):.2f}, max={np.max(all_mean_ll):.2f}")
    print(f"Worst LL: avg={np.mean(all_min_ll):.2f}, min={np.min(all_min_ll):.2f}, max={np.max(all_min_ll):.2f}")
    print(f"Best LL:  avg={np.mean(all_max_ll):.2f}, min={np.min(all_max_ll):.2f}, max={np.max(all_max_ll):.2f}")

    return {
        "alpha_diffs": alpha_diffs.tolist(),
        "t_stat": t_stat,
        "p_value": p_value,
        "mean": mean_diff,
        "std": std_diff,
        "ci_95": ci_95,
        "fit_summaries": fit_summaries,
        "n_iterations": n_iterations,
        "n_samples": n_samples,
    }


# =============================================================================
# 解析: 色別方向価値Q学習（OOZ依存学習率）の alpha_diff 検定
# =============================================================================

def run_directional_alpha_diff_test(
    all_data_learning,
    fit=True,
    N=4,
    kappa=2.0,
):
    """色別方向価値Q学習を各参加者にフィットし，log(alpha_1/alpha_0) の
    個人差を1標本t検定（H0: mean=0）で検定する。

    Parameters
    ----------
    all_data_learning : list of (subject_id, DataFrame)
        各 df は rt, chosen_color, response_angle_css, target_direction,
        distractor_direction, target_group, reward_points, ooz を含む。
    fit : bool
        True で推定を実行，False で保存済み結果を読み込む。
    N, kappa : int, float
        von Mises 基底のパラメータ。

    Returns
    -------
    dict
        results_df, alpha_diffs, t_stat, p_value, mean, sd, n を含む。
    """
    results_df, _ = run_directional_q_learning(
        fit=fit, concat_list=all_data_learning, N=N, kappa=kappa,
    )

    results_df = results_df.copy()
    results_df["alpha_diff"] = np.log(results_df["alpha_1"] / results_df["alpha_0"])
    alpha_diffs = results_df["alpha_diff"].replace([np.inf, -np.inf], np.nan).dropna()

    t_stat, p_value = stats.ttest_1samp(alpha_diffs, popmean=0)
    mean_diff = float(np.mean(alpha_diffs))
    sd_diff   = float(np.std(alpha_diffs, ddof=1))

    print("\n=== Directional Q-learning: alpha_diff one-sample t-test ===")
    print(f"n = {len(alpha_diffs)}, mean = {mean_diff:.4f}, sd = {sd_diff:.4f}")
    print(f"t = {t_stat:.4f}, p = {p_value:.4f}")

    return {
        "results_df":  results_df,
        "alpha_diffs": alpha_diffs.tolist(),
        "t_stat":      float(t_stat),
        "p_value":     float(p_value),
        "mean":        mean_diff,
        "sd":          sd_diff,
        "n":           int(len(alpha_diffs)),
    }


# =============================================================================
# オーケストレータ
# =============================================================================

def run_behavior(
    all_data_learning,
    all_data_awareness=None,
    subjects_behavior_on=None,
    subjects_behavior_off=None,
    hmm_df=None,
    behavioral_df=None,
):
    """行動解析のオーケストレータ: 前処理 → 統計検定 → 可視化。"""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)

    # 1) 前処理
    all_df, behavioral_df, results_df, metrics = prepare_analysis_df(
        all_data_learning, behavioral_df=behavioral_df
    )
    print(metrics.describe())

    # 2) 行動指標の要約表示
    print("\nTask relevant choice rate per subject:")
    print(behavioral_df.loc[:, [
        "subject", "valid_trial_count", "rt_cv",
        "task_relevant_choice_rate", "target_choice_rate", "target_choice_rate_diff",
    ]])
    print(f"\nMean win-stay rate: {behavioral_df['win_stay_rate'].mean()}")
    print(f"Mean lose-switch rate: {behavioral_df['lose_switch_rate'].mean()}")

    # 3) 統計検定
    test_results = run_statistical_tests(all_data_learning, all_df)
    _print_test_results(test_results)

    test_results = run_directional_alpha_diff_test(all_data_learning, fit=False, N=4, kappa=2.0)
    # _print_test_results(test_results)
    directional_alpha_df = test_results["results_df"][["subject", "alpha_diff"]].rename(
        columns={"alpha_diff": "alpha_diff_directional"}
    )
    all_df = all_df.merge(directional_alpha_df, on="subject", how="left")

    # 4) 要約統計 + 可視化
    summary = compute_summary_statistics(all_df)
    qs = summary["quartile_stats"]
    print(
        f"\nOOZ rate: Q1={qs['Q1']:.4f}, Q2={qs['Q2']:.4f}, Q3={qs['Q3']:.4f}, "
        f"IQR={qs['IQR']:.4f}, min={qs['minimum']:.4f}, max={qs['maximum']:.4f}, "
        f"lower={qs['lower_fence']:.4f}, upper={qs['upper_fence']:.4f}"
    )

    # plot_exp_obj_with_linear_fit(
    #     summary["combined_variances"],
    #     title="Alpha diff vs Target choice rate diff",
    #     xlabel="alpha_diff (learning rate difference)",  # ここは alpha_diff を示すように変更
    #     ylabel="target_choice_rate_diff (early)",  # ここは target_choice_rate_diff を示すように変更
    #     # save_path="./fig/rt_mean_vs_angular_error.pdf",
    # )

    # # 5) ブートストラップ: 群レベル alpha_diff 検定
    # bootstrap_results = run_bootstrap_alpha_diff(
    #     all_data_learning,
    #     n_samples=30,
    #     n_iterations=100,
    #     random_state=42,
    # )

    return {
        "all_data_learning": all_data_learning,
        "all_data_awareness": all_data_awareness,
        "behavioral_df": behavioral_df,
    }
