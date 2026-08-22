"""混合モデル結果の可視化。

1 枚のサマリ図に、モデルの主張と限界の両方が見えるようにする。
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dualrdk.mixed.config import CJK_FONT_CANDIDATES, FIG_DPI, FIG_STYLE
from dualrdk.mixed.results import MixedResult


def _style():
    try:
        plt.style.use(FIG_STYLE)
    except Exception:
        plt.style.use("default")
    _use_cjk_font()


def _use_cjk_font():
    """日本語グリフを持つフォントを選ぶ。無ければ既定のまま（英字は出る）。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            # 既定の Type 3 埋め込みは非 ASCII グリフで UnicodeEncodeError に
            # なるため、PDF/EPS では TrueType (42) を使う
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42
            return name
    return None


def plot_lmm_summary(
    res: MixedResult,
    df: pd.DataFrame,
    out_path: Path,
    boot_ci: Optional[pd.DataFrame] = None,
    diag: Optional[dict] = None,
    y: str = "logrt",
    x: str = "trial_c",
    title: str = "",
) -> Path:
    """6 パネルのサマリ図。

    A 個人の軌跡と集団平均   B 個人 BLUP 傾きの分布
    C 分散成分の分解         D ブートストラップ分布
    E 残差診断               F 参加者ごとの残差 SD（等分散仮定の検証）
    """
    _style()
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    b0 = res.fixed.loc["Intercept", "estimate"]
    b1 = res.fixed.loc[x, "estimate"] if x in res.fixed.index else 0.0
    xs = np.linspace(df[x].min(), df[x].max(), 50)

    # ---- A: 軌跡 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    blup = res.random_effects
    for s in blup.index:
        i = b0 + blup.loc[s, "Intercept"]
        sl = b1 + (blup.loc[s, x] if x in blup.columns else 0.0)
        ax.plot(xs, i + sl * xs, color="0.75", lw=0.6, alpha=0.7, zorder=1)
    blk = df.assign(_b=(df[x] // 8).astype(int)).groupby("_b").agg(m=(y, "mean"), c=(x, "mean"))
    ax.plot(blk["c"], blk["m"], "o", color="#c0392b", ms=7, zorder=3, label="実測ブロック平均")
    ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=2.5, zorder=4, label="集団平均")
    ax.set_xlabel("trial（1試行目 = 0）")
    ax.set_ylabel("log RT")
    ax.set_title("A  個人の軌跡（BLUP）と集団平均", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper right")

    # ---- B: 傾きの分布 ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if x in blup.columns:
        sl = (b1 + blup[x]).values
        ax.hist(sl, bins=18, color="#5b8ca8", edgecolor="white")
        ax.axvline(0, color="0.35", ls=":", lw=1.4)
        ax.axvline(b1, color="#c0392b", lw=2, label=f"集団平均 {b1:+.5f}")
        tau1 = res.varcomp.loc[x, "sd"]
        ax.axvspan(b1 - tau1, b1 + tau1, color="#c0392b", alpha=0.12, label=f"±1 tau1 ({tau1:.5f})")
        ax.set_xlabel("参加者ごとの傾き")
        ax.set_ylabel("人数")
        n_pos = int((sl > 0).sum())
        ax.set_title(f"B  個人傾き（遅くなる人 {n_pos}/{len(sl)}）", fontsize=11, loc="left")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "ランダム傾きなし", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B  個人傾き", fontsize=11, loc="left")

    # ---- C: 分散分解 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    r2 = res.r2(df)
    if r2:
        parts = [
            ("固定効果\n(trial)", r2["var_fixed"], "#c0392b"),
            ("ランダム効果\n(参加者間)", r2["var_random"], "#5b8ca8"),
            ("残差\n(参加者内)", r2["var_residual"], "#95a5a6"),
        ]
        tot = sum(p[1] for p in parts)
        bottom = 0
        for lab, v, c in parts:
            ax.bar(0, v, bottom=bottom, color=c, width=0.55, edgecolor="white")
            if v / tot > 0.02:
                ax.text(0, bottom + v / 2, f"{lab}\n{100*v/tot:.1f}%", ha="center",
                        va="center", fontsize=8.5, color="white", weight="bold")
            bottom += v
        ax.text(0.42, r2["var_fixed"] / 2, f"← {100*r2['var_fixed']/tot:.2f}%",
                fontsize=8.5, va="center", color="#c0392b", weight="bold")
        ax.set_xlim(-0.5, 1.0)
        ax.set_xticks([])
        ax.set_ylabel("分散")
        ax.set_title(
            f"C  分散の分解  R²m={r2['r2_marginal']:.4f} / R²c={r2['r2_conditional']:.3f}",
            fontsize=11, loc="left")

    # ---- D: ブートストラップ ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    if boot_ci is not None and len(boot_ci):
        show = boot_ci[boot_ci["parameter"].str.startswith(("sd[", "beta["))].copy()
        show = show[~show["parameter"].str.contains("Intercept")]
        show = show.reset_index(drop=True)
        # パラメータごとにスケールが桁違いなので、点推定で割って
        # 「点推定の何倍か」に揃える。点は必ず 1.0、0 は 0 のまま。
        labels = []
        for k, r in show.iterrows():
            s = r["point"] if pd.notna(r["point"]) and r["point"] != 0 else 1.0
            lo, hi = sorted([r["ci_low"] / s, r["ci_high"] / s])
            ax.plot([lo, hi], [k, k], color="#5b8ca8", lw=3, solid_capstyle="round")
            ax.plot(1.0, k, "o", color="#c0392b", ms=7, zorder=3)
            labels.append(f"{r['parameter']}\n{r['point']:+.5f}")
        ax.axvline(0, color="#c0392b", ls="--", lw=1.4, label="0（効果なし）")
        ax.axvline(1, color="0.5", ls=":", lw=1.0)
        ax.set_yticks(np.arange(len(show)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("点推定の何倍か（0 をまたがなければ有意）")
        ax.set_title("D  参加者ブートストラップ 95% CI", fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="lower right")

    # ---- E: 残差 ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    resid = np.asarray(res.raw.resid)
    stats.probplot(resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2.0)
    ax.get_lines()[0].set_color("#5b8ca8")
    ax.get_lines()[1].set_color("#c0392b")
    ax.set_title("")  # probplot が中央に付ける "Probability Plot" を消す
    t = "E  残差の正規 Q-Q"
    if diag:
        t += f"  skew={diag['skew']:+.2f} kurt={diag['kurtosis']:+.2f}"
    ax.set_title(t, fontsize=11, loc="left")
    ax.set_xlabel("理論分位点")
    ax.set_ylabel("残差")

    # ---- F: 等分散仮定 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    rs = pd.DataFrame({"s": df[res.spec.groups].values, "r": resid})
    per = rs.groupby("s")["r"].std().sort_values()
    ax.bar(range(len(per)), per.values, color="#5b8ca8", width=1.0)
    assumed = np.sqrt(res.scale) if res.scale else np.nan
    ax.axhline(assumed, color="#c0392b", lw=2, label=f"モデルの仮定 σ={assumed:.3f}")
    ax.set_xlabel("参加者（残差 SD 順）")
    ax.set_ylabel("参加者ごとの残差 SD")
    t = "F  等分散仮定の検証"
    if diag:
        t += f"  Levene p={diag['levene_p']:.1e}（比 {diag['resid_sd_ratio']:.2f}倍）"
    ax.set_title(t, fontsize=11, loc="left")
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=0.985)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(
    cmp_table: pd.DataFrame, out_path: Path, title: str = ""
) -> Path:
    """AIC / BIC によるモデル比較の図。"""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = cmp_table["model"].tolist()
    for ax, col in zip(axes, ["AIC", "BIC"]):
        v = cmp_table[col].values
        colors = ["#c0392b" if x == v.min() else "#95a5a6" for x in v]
        ax.bar(labels, v - v.min(), color=colors, width=0.5)
        ax.set_ylabel(f"Δ{col}（最良モデルからの差）")
        ax.set_title(f"{col}  最良: {labels[int(np.argmin(v))]}", fontsize=11, loc="left")
        for i, x in enumerate(v - v.min()):
            ax.text(i, x, f"{x:+.2f}", ha="center", va="bottom", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
