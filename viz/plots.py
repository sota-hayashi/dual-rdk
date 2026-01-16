from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from io_data.utils import combine_subjects
from features.lapses import compute_out_of_zone_ratio_by_rt, compute_out_of_zone_ratio_by_AE, compute_out_of_zone_ratio_of_mean_AE
from features.behavior import summarize_chosen_item_errors
from common.config import TRIALS_PER_SESSION


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
    sub = sub[sub["chosen_item"].isin([0, 1])]
    if sub.empty:
        print(f"No data for session logistic regression plot.")
        return

    X = sm.add_constant(sub["num_trial"])
    model = sm.Logit(sub["chosen_item"], X)
    fit = model.fit(disp=False)
    slope = fit.params["num_trial"]
    pval = fit.pvalues["num_trial"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x="num_trial",
        y="chosen_item",
        data=sub,
        ax=ax,
        logistic=True,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95
    )
    ax.set_title(f"Subject: Logistic Regression of Chosen Item vs Trial", fontsize=14)
    ax.set_xlabel("Trial (num_trial)", fontsize=12)
    ax.set_ylabel("Chosen Item (0/1)", fontsize=12)

    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(sub)}",
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
    save_path: str = None
):
    """
    all_data_learning の rt をまとめてヒストグラム表示する。
    bin_ms はミリ秒単位のビン幅。
    """
    combined = combine_subjects(concat_list)
    if combined.empty or "rt" not in combined.columns:
        print("No RT data available.")
        return

    rt_values = combined.loc[combined["chosen_item"] == -1, "rt"].dropna().astype(float)
    if rt_values.empty:
        print("No RT data available.")
        return

    bin_width = float(bin_ms)
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    bins = np.arange(min_rt, max_rt + bin_width, bin_width)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rt_values, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.set_xlabel("RT (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("RT Histogram (All Subjects)", fontsize=14)
    ax.legend([f"n={len(rt_values)}"], loc="upper right", fontsize=11)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_MW_vs_reward_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    save_path: str = None,
    ooz_index: str = "ae_mean_based",
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
    }
    func = ooz_fn or ooz_methods.get(ooz_index)
    if func is None:
        raise ValueError(f"Unknown ooz_index: {ooz_index}")

    rows = []
    for subj_id, df in concat_list:
        try:
            out_ratio = func(df, n_trial=n_trial)
            valid = df.dropna(subset=["reward_points"])
            # 分割する試行数を指定することで任意の期間の平均報酬を計算できるようにする
            if n_trial == TRIALS_PER_SESSION:
                pass
            else:
                valid = valid[valid["num_trial"] > n_trial - 1]
            valid = valid[valid["chosen_item"].isin([0, 1])]
            mean_reward = valid["reward_points"].mean() if not valid.empty else np.nan
            rows.append({
                "subject": subj_id,
                "out_of_zone_ratio": out_ratio,
                "mean_reward": mean_reward
            })
        except Exception as e:
            print(f"Skipping {subj_id}: {e}")
            continue

    result_df = pd.DataFrame(rows).dropna()

    if result_df.empty or len(result_df) < 3:
        print("Not enough data for regression analysis.")
        return result_df

    X = sm.add_constant(result_df["out_of_zone_ratio"])
    fit = sm.OLS(result_df["mean_reward"], X).fit()
    slope = fit.params["out_of_zone_ratio"]
    pval = fit.pvalues["out_of_zone_ratio"]

    # rho, pval = spearmanr(result_df["out_of_zone_ratio"], result_df["mean_reward"])

    plt.style.use("seaborn-v0_8-whitegrid")
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

def plot_rt_by_trial(
    df: pd.DataFrame,
    save_path: str = None
):
    """
    For a single DataFrame, plot reaction time (rt) vs trial (num_trial) scatter plot.
    """
    sub = df.dropna(subset=["rt", "num_trial"]).copy()
    if sub.empty:
        print("No data for RT vs Trial plot.")
        return
    X = sm.add_constant(sub["num_trial"])
    fit = sm.OLS(sub["rt"], X).fit()
    slope = fit.params.get("num_trial", np.nan)
    pval = fit.pvalues.get("num_trial", np.nan)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x="num_trial",
        y="rt",
        data=sub,
        ax=ax,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=95
    )
    ax.set_title("Reaction Time vs Trial", fontsize=14)
    ax.set_xlabel("Trial (num_trial)", fontsize=12)
    ax.set_ylabel("Reaction Time (rt)", fontsize=12)

    sig_marker = "*" if pval < 0.05 else ""
    ax.text(
        0.05, 0.95,
        f"slope = {slope:.4f}\np = {pval:.4f}{sig_marker}\nn = {len(sub)}",
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
