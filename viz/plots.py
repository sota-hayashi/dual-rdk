from typing import List, Tuple, Callable
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from io_data.utils import combine_subjects
from features.lapses import (
    compute_out_of_zone_ratio_by_rt, 
    compute_out_of_zone_ratio_by_AE, 
    compute_out_of_zone_ratio_of_mean_AE,
    compute_out_of_zone_ratio_by_task_irrelevant_rate,
    compute_out_of_the_zone_ratio_by_rt_moving,
    label_if_ooz,
)
from features.behavior import summarize_chosen_item_errors
from io_data.load import load_hmm_summary
from common.config import TRIALS_PER_SESSION

def plot_frac_ae_target_distractor_by_trial(
    df: pd.DataFrame,
    save_path: str = None,
    window: int = 3
):
    """
    Plot the fraction of AE (Angular Error) for target and distractor items by trial.
    """
    sub = df.dropna(subset=["num_trial", "angular_error_target", "angular_error_distractor"]).copy()
    if sub.empty:
        print(f"No data for session frac AE plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # choose smaller of target/distractor angular error and apply exponential(normalized by 180)
    sub["frac_ae"] = 1 / np.exp(
        sub["angular_error_target"].abs() / 180.0
        # np.where(
        #     sub["angular_error_target"].abs() < sub["angular_error_distractor"].abs(),
        #     sub["angular_error_target"].abs() / 180.0,
        #     sub["angular_error_distractor"].abs() / 180.0,
        # )
    )

    if window > 1:
        plot_values = sub["frac_ae"].rolling(window=window, min_periods=1).mean()
        label = f'frac_ae (smoothed, window={window})'
    else:
        plot_values = sub["frac_ae"]
        label = 'frac_ae'

    
    ax.plot(sub["num_trial"], plot_values, label=label)

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("The inverse of exponential of target and distractor angular error", fontsize=12)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_rt_by_trial(
    df: pd.DataFrame,
    save_path: str = None,
    window: int = 3
):
    """Plot RT by trial with optional smoothing."""
    sub = df.dropna(subset=["num_trial", "rt"]).copy()
    if sub.empty:
        print(f"No data for session RT plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    if window > 1:
        plot_values = sub["rt"].rolling(window=window, min_periods=1).mean()
        label = f'RT (smoothed, window={window})'
    else:
        plot_values = sub["rt"]
        label = 'RT'

    ax.plot(sub["num_trial"], plot_values, label=label)
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Reaction Time (ms)", fontsize=12)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_reward_by_trial(
    df: pd.DataFrame,
    save_path: str = None,
    corner: str = "upper left"
):
    """
    Scatter plot of trial vs reward for a given session with Pearson r and regression line (95% CI).
    Colors aligned with learning-more-code style: blue dots, red line, red-transparent CI.
    """
    sub = df.dropna(subset=["num_trial", "reward_points", "rt"]).copy()
    sub = sub[sub["chosen_item"].isin([0, 1])]
    if sub.empty:
        print(f"No data for session reward vs trial plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
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
    ax.set_title(f"Subject3: Reward vs Trial", fontsize=14)
    ax.set_xlabel("Trial (num_trial)", fontsize=12)
    ax.set_ylabel("Reward points", fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_logistic_regression_per_subject(
    df: pd.DataFrame,
    save_path: str = None
):
    """
    For a single DataFrame, plot logistic regression of chosen_item (0/1) vs num_trial for a given session.
    """
    sub = df.dropna(subset=["num_trial", "chosen_item", "rt"]).copy()
    # sub = sub[sub["num_trial"] > 17]
    sub_irrelevant = sub[sub["chosen_item"] == -1]
    sub_relevant = sub[sub["chosen_item"].isin([0, 1])]
    if sub_relevant.empty:
        print(f"No data for session logistic regression plot.")
        return

    X = sm.add_constant(sub_relevant["num_trial"])
    model = sm.Logit(sub_relevant["chosen_item"], X)
    fit = model.fit(disp=False)
    slope = fit.params["num_trial"]
    pval = fit.pvalues["num_trial"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # chosen_item == -1 の点を赤色でプロット
    if not sub_irrelevant.empty:
        sns.scatterplot(
            x="num_trial",
            y="chosen_item",
            data=sub_irrelevant,
            ax=ax,
            color="red",
            alpha=0.5,
            label="Irrelevant Choice (-1)"
        )

    # chosen_item が 0 または 1 の点をデフォルト色（青）でプロット
    sns.scatterplot(
        x="num_trial",
        y="chosen_item",
        data=sub_relevant,
        ax=ax,
        color="blue",
        alpha=0.5,
        label="Relevant Choice (0 or 1)"
    )

    # 回帰直線と信頼区間のみを描画 (点は描画しない)
    sns.regplot(
        x="num_trial",
        y="chosen_item",
        data=sub_relevant,
        ax=ax,
        logistic=True,
        scatter=False,  # 点は描画しない
        line_kws={"color": "red"},
        ci=95
    )
    ax.legend(loc="lower left")
    ax.set_title(f"Subject: Logistic Regression of Chosen Item vs Trial", fontsize=14)
    ax.set_xlabel("Trial (num_trial)", fontsize=12)
    ax.set_ylabel("Chosen Item (0/1)", fontsize=12)

    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(sub_relevant)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()
    return


def plot_regression_across_subjects(concat_list, session_id=0, save_path=None):
    df = combine_subjects(concat_list).dropna(subset=["num_trial", "reward_points", "num_session"])
    sub = df[df["num_session"] == session_id]
    if sub.empty:
        print(f"No data for session {session_id}")
        return

    X = sm.add_constant(sub["num_trial"])
    fit = sm.OLS(sub["reward_points"], X).fit()
    slope = fit.params["num_trial"]
    pval = fit.pvalues["num_trial"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x="num_trial",
        y="reward_points",
        data=sub,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95,
        ax=ax
    )
    ax.set_title(f"Across subjects: Reward vs Trial (session {session_id+1})")
    ax.text(
        0.05, 0.95,
        f"slope={slope:.3f}\np={pval:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_rt_histogram_all_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    bin_ms: int = 100,
    save_path: str = None,
    fit_lognormal: bool = True,
    cutoff_ms: float = 10000.0
):
    """
    all_data_learning の rt をまとめてヒストグラム表示する。
    bin_ms はミリ秒単位のビン幅。
    """
    combined = combine_subjects(concat_list)
    if combined.empty or "rt" not in combined.columns:
        print("No RT data available.")
        return

    rt_values = combined["rt"].dropna().astype(float)
    if rt_values.empty:
        print("No RT data available.")
        return

    bin_width = float(bin_ms)
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    bins = np.arange(min_rt, max_rt + bin_width, bin_width)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rt_values, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white", density=fit_lognormal)
    ax.set_xlabel("RT (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("RT Histogram (All Subjects)", fontsize=14)

    if fit_lognormal:
        log_rt = np.log(rt_values)
        mu = log_rt.mean()
        sigma = log_rt.std(ddof=1)
        x = np.linspace(min_rt, max_rt, 400)
        pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        ax.plot(x, pdf, color="red", linewidth=2)

        if cutoff_ms is not None and cutoff_ms > 0:
            cutoff_log = np.log(cutoff_ms)
            tail_prob = 1 - 0.5 * (1 + np.math.erf((cutoff_log - mu) / (sigma * np.sqrt(2))))
            ax.axvline(cutoff_ms, color="gray", linestyle="--", linewidth=1)
            ax.text(
                0.98, 0.95,
                f"cutoff={cutoff_ms:.0f}ms\nlognormal tail≈{tail_prob:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
            )
        ax.legend(loc="upper right", fontsize=11)
    else:
        ax.legend([f"n={len(rt_values)}"], loc="upper right", fontsize=11)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_MW_vs_target_choice_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    save_path: str = None,
    ooz_index: str = "rt_moving_based",
    n_trial: int = TRIALS_PER_SESSION,
    ooz_fn: Callable[[pd.DataFrame, float, float], float] = None
) -> pd.DataFrame:
    """
    各被験者のout of the zone割合と平均獲得報酬の関係をプロットする。
    横軸: out of the zone 割合
    縦軸: 平均獲得報酬
    回帰直線、傾き、p値、95%信頼区間を表示。
    """
    ooz_methods = {
        "ae_based": compute_out_of_zone_ratio_by_AE,
        "ae_mean_based": compute_out_of_zone_ratio_of_mean_AE,
        "rt_based": compute_out_of_zone_ratio_by_rt,
        "task_irrelevant_based": compute_out_of_zone_ratio_by_task_irrelevant_rate,
        "rt_moving_based": compute_out_of_the_zone_ratio_by_rt_moving,

    }
    func = ooz_fn or ooz_methods.get(ooz_index)
    if func is None:
        raise ValueError(f"Unknown ooz_index: {ooz_index}")
    if ooz_index == "rt_moving_based":
        concat_list = label_if_ooz(concat_list)

    rows = []
    for subj_id, df in concat_list:
        try:
            out_ratio = func(df, n_trial=n_trial)
            valid = df.dropna(subset=["rt"])
            # 分割する試行数を指定することで任意の期間の平均報酬を計算できるようにする
            if n_trial == TRIALS_PER_SESSION:
                pass
            else:
                valid = valid[valid["num_trial"] > n_trial - 1]
            # valid["chosen_item"] = valid["chosen_item"].replace({-1: 0, 0: 0, 1: 1})
            # valid = valid[valid["chosen_item"].isin([0, 1])]
            mean_target = (
                # np.where(
                #     valid["angular_error_target"].abs() < valid["angular_error_distractor"].abs(),
                #     valid["angular_error_target"].abs(),
                #     valid["angular_error_distractor"].abs()
                # ).mean()
                valid["angular_error_target"].abs().mean()
                # - valid.loc[valid["num_trial"] < n_trial//2,"chosen_item"].mean() if not valid.empty else np.nan
            )
            rows.append({
                "subject": subj_id,
                "out_of_zone_ratio": out_ratio,
                "mean_target": mean_target
            })
        except Exception as e:
            print(f"Skipping {subj_id}: {e}")
            continue

    result_df = pd.DataFrame(rows).dropna()

    if result_df.empty or len(result_df) < 3:
        print("Not enough data for regression analysis.")
        return result_df

    X = sm.add_constant(result_df["out_of_zone_ratio"])
    fit = sm.OLS(result_df["mean_target"], X).fit()
    slope = fit.params["out_of_zone_ratio"]
    pval = fit.pvalues["out_of_zone_ratio"]

    # rho, pval = spearmanr(result_df["out_of_zone_ratio"], result_df["mean_target"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="out_of_zone_ratio",
        y="mean_target",
        data=result_df,
        ax=ax,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95
    )

    ax.set_xlabel("Out of the Zone Ratio", fontsize=12)
    ax.set_ylabel("Mean Target Choice Rate", fontsize=12)
    ax.set_title("Mind Wandering vs Target Choice (Across Subjects)", fontsize=14)

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


def plot_std_distractor_ae_vs_mean_reward(
    concat_list: List[Tuple[str, pd.DataFrame]],
    save_path: str = None
) -> pd.DataFrame:
    """
    chosen_item=0 の angular_error_distractor 標準偏差を out_of_the_zone_ratio として使用し、
    被験者ごとの mean_reward_points との関係をプロットする。
    横軸: std_dis_AE (ratio)
    縦軸: mean_reward_points
    """
    stats = summarize_chosen_item_errors(concat_list)
    if stats.empty:
        print("No data for chosen_item error stats.")
        return stats

    dis_stats = stats[stats["chosen_item"] == 0].copy()
    if dis_stats.empty:
        print("No chosen_item=0 data.")
        return dis_stats

    ratio_df = dis_stats[["subject", "std_error"]].rename(
        columns={"std_error": "std_dis_AE"}
    ).dropna()

    combined = combine_subjects(concat_list)
    reward_df = (
        combined.dropna(subset=["reward_points"])
        .groupby("subject")["reward_points"]
        .mean()
        .reset_index(name="mean_reward_points")
    )

    result_df = ratio_df.merge(reward_df, on="subject", how="inner").dropna()
    if result_df.empty or len(result_df) < 3:
        print("Not enough data for regression analysis.")
        return result_df

    X = sm.add_constant(result_df["std_dis_AE"])
    fit = sm.OLS(result_df["mean_reward_points"], X).fit()
    slope = fit.params.get("std_dis_AE", np.nan)
    pval = fit.pvalues.get("std_dis_AE", np.nan)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="std_dis_AE",
        y="mean_reward_points",
        data=result_df,
        ax=ax,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95
    )
    ax.set_xlabel("out_of_the_zone_ratio (std_dis_AE)", fontsize=12)
    ax.set_ylabel("mean_reward_points", fontsize=12)
    ax.set_title("std_dis_AE vs mean_reward_points (Across Subjects)", fontsize=14)

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

# def plot_rt_by_trial(
#     df: pd.DataFrame,
#     save_path: str = None
# ):
#     """
#     For a single DataFrame, plot reaction time (rt) vs trial (num_trial) scatter plot.
#     """
#     sub = df.dropna(subset=["rt", "num_trial"]).copy()
#     if sub.empty:
#         print("No data for RT vs Trial plot.")
#         return
#     X = sm.add_constant(sub["num_trial"])
#     fit = sm.OLS(sub["rt"], X).fit()
#     slope = fit.params.get("num_trial", np.nan)
#     pval = fit.pvalues.get("num_trial", np.nan)
    
#     plt.style.use("seaborn-v0_8-whitegrid")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.regplot(
#         x="num_trial",
#         y="rt",
#         data=sub,
#         ax=ax,
#         scatter_kws={"alpha": 0.5},
#         line_kws={"color": "red"},
#         ci=95
#     )
#     ax.set_title("Reaction Time vs Trial", fontsize=14)
#     ax.set_xlabel("Trial (num_trial)", fontsize=12)
#     ax.set_ylabel("Reaction Time (rt)", fontsize=12)

#     sig_marker = "*" if pval < 0.05 else ""
#     ax.text(
#         0.05, 0.95,
#         f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(sub)}",
#         transform=ax.transAxes,
#         va="top",
#         fontsize=11,
#         bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
#     )
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"Saved plot to {save_path}")
#     plt.show()
#     return


def plot_hmm_subject_result(
    subject_id: str,
    hmm_result: dict
):
    """
    Plot chosen_item series and HMM state sequence, plus emission/transition summaries.
    """
    obs = hmm_result["observations"]
    states = hmm_result["states"]
    A = hmm_result["A"]
    B = hmm_result["B"]
    labels = hmm_result["state_labels"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 2]})

    axes[0].plot(obs, color="#4C72B0", linewidth=1)
    axes[0].set_title(f"HMM states for subject {subject_id}")
    axes[0].set_ylabel("chosen_item")
    axes[0].set_yticks([-1, 0, 1])

    axes[1].plot(states, color="#DD8452", linewidth=1)
    axes[1].set_ylabel("state")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels([labels.get(0, "state0"), labels.get(1, "state1")])

    bar_x = np.arange(B.shape[1])
    axes[2].bar(bar_x - 0.15, B[0], width=0.3, label=labels.get(0, "state0"))
    axes[2].bar(bar_x + 0.15, B[1], width=0.3, label=labels.get(1, "state1"))
    axes[2].set_xticks(bar_x)
    axes[2].set_xticklabels(["else", "distractor", "target"])
    axes[2].set_ylabel("emission prob")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    im = ax2.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax2.set_title("Transition matrix")
    ax2.set_xlabel("to")
    ax2.set_ylabel("from")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.show()


def plot_exploit_target_prob_by_switch(
    subject_prob_list: List[Tuple[str, List[float]]]
):
    """
    各被験者の explore->exploit 切り替え後の target選択確率を時系列で表示。
    横軸: switch回数, 縦軸: target選択確率
    """
    if not subject_prob_list:
        print("No subject data for plotting.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    for subj_id, probs in subject_prob_list:
        if not probs:
            continue
        x = np.arange(1, len(probs) + 1)
        ax.plot(x, probs, marker="o", linewidth=1, alpha=0.7, label=subj_id)

    ax.set_xlabel("Switch count", fontsize=12)
    ax.set_ylabel("Mean target choice probability", fontsize=12)
    ax.set_title("Target choice probability after explore->exploit switches", fontsize=14)
    # ax.set_ylim(0, 1)
    if len(subject_prob_list) <= 10:
        ax.legend(fontsize=9, loc="best")
    plt.show()


def plot_frac_exploit_vs_valid_choice_rate(
    summary_path: Path,
    subjects_include: List[str] = None
):
    """
    hmm_normalized_summary.csv を読み込み、frac_exploit と
    mean(observations in {0,1}) の散布図を描画する。
    """
    df = load_hmm_summary(summary_path)
    if df.empty:
        print("No HMM summary data.")
        return

    def valid_choice_rate(obs):
        arr = np.array(obs)
        valid = arr[arr != -1] # -1（ターゲット・ディストラクター以外を選択） を除外
        # print(np.sum(arr[0:23] == -1))
        # print(np.sum(arr[24:] == -1))
        # print(np.sum(arr == -1))
        if valid.size == 0:
            return np.nan
        valid1 = valid[:23]
        valid2 = valid[23:]
        print(valid.size)
        return float(np.mean(valid))
    if subjects_include is not None:
        print(f"Filtering subjects: {subjects_include}")
        df = df[df["subject"].isin(subjects_include)].copy()
    plot_df = df[["subject", "frac_exploit", "observations", "states"]].copy()
    # plot_df["frac_explore"] = 1.0 - plot_df["frac_exploit"]
    plot_df["frac_explore"] = 1 - plot_df["states"].apply(lambda s: np.mean(np.array(s) == 1))
    plot_df["valid_choice_rate"] = plot_df["observations"].apply(valid_choice_rate)

    if len(plot_df) < 3:
        print("Not enough data for regression analysis.")
        return

    X = sm.add_constant(plot_df["frac_explore"])
    fit = sm.OLS(plot_df["valid_choice_rate"], X).fit()
    slope = fit.params.get("frac_explore", np.nan)
    pval = fit.pvalues.get("frac_explore", np.nan)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="frac_explore",
        y="valid_choice_rate",
        data=plot_df,
        ax=ax,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"},
        ci=95
    )
    ax.set_xlabel("frac_explore", fontsize=12)
    ax.set_ylabel("mean(observations in {0,1})", fontsize=12)
    ax.set_title("frac_explore vs valid choice rate", fontsize=14)
    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(plot_df)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    plt.show()


def plot_hist_with_lognormal_fit(
    values: List[float],
    bin_size: float,
    title: str = "Histogram with Gaussian Fit",
    xlabel: str = "Value",
    ylabel: str = "Density"
):
    """
    指定したデータとビン幅でヒストグラムを描画し、ガウス分布をフィットして重ねる。
    さらに第1四分位〜第3四分位の範囲をガウス分布の面積として薄く表示する。
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print("No valid data for histogram.")
        return

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    mu = float(arr.mean())
    log_ae = np.log(arr)
    mu = log_ae.mean()
    sigma = log_ae.std(ddof=1)
    if sigma == 0.0:
        print("Standard deviation is zero; Gaussian fit is not informative.")

    min_val = arr.min()
    max_val = arr.max()
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    q1, q3 = np.percentile(arr, [25, 75])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white", density=True)

    x = np.linspace(min_val, max_val, 400)
    if sigma > 0:
        pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        ax.plot(x, pdf, color="red", linewidth=2, label="Log-normal fit")
        mask = (x >= q1) & (x <= q3)
        ax.fill_between(x[mask], 0, pdf[mask], color="red", alpha=0.15, label="IQR area")

    text_str = f"Q1 (25%): {q1:.2f}\nQ3 (75%): {q3:.2f}\nMean: {arr.mean():.2f}\nStd: {arr.std(ddof=1):.2f}"
    ax.text(0.50, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.show()

def plot_exp_obj_with_linear_fit(
    values: List[Tuple[float, float]],
    title: str = "Plot of Explanatory variable and Objective variable with Linear Fit",
    xlabel: str = "Explanatory variable",
    ylabel: str = "Objective variable"
):
    """
    指定したデータで散布図を描画し、線形回帰をフィットして重ねる。
    """
    if not values:
        print("No data for plotting.")
        return

    x = np.array([v[0] for v in values], dtype=float)
    y = np.array([v[1] for v in values], dtype=float)

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    fit = model.fit(disp=False)
    slope = fit.params[1] if len(fit.params) > 1 else np.nan
    pval = fit.pvalues[1] if len(fit.pvalues) > 1 else np.nan
    print(f"Linear fit results: slope={slope:.4f}, p-value={pval:.4f}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}, ci=95)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(values)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    plt.show()