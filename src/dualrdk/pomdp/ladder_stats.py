"""`ladder.py` の per_subject.csv に対する検定（§R-1 の感度分析）。

`ladder.py` が出す 4 段の推定値を、参加者を標本として 2 つの問いで検定する。

**問1: Delta alpha = alpha_out - alpha_in は 0 と異なるか**

    本来の科学的問い（注意状態で学習率が違うか）を、段ごとに独立に検定する。
    4 段は同じデータの 4 通りの解析なので、これは 4 つの独立な仮説検定では
    なく**感度分析**である。「解析上の恣意的な選択（座標系・事前分布・データ
    整形）で結論が変わるか」を見るもので、多重比較補正の対象にはしない
    （補正は独立な検定族を前提とする）。Holm 補正列も出すが参考値である。

    有意でなかったときのために**同等性の検定**も併記する。Delta = 0 という
    帰無仮説の棄却に失敗しても「差が無い」ことにはならないので、TOST の
    論理で「このデータが支持できる最小の同等マージン」を報告する。90% 信頼
    区間が (-d, d) に収まる最小の d がそれで、「差があってもこの大きさ未満」
    と言える上限になる。

**問2: 参加者ごとの推定値の順位は段間で保存されるか**

    「beta と c_black はデータが決めているが alpha は事前分布の産物」という
    主張の直接の検証。段を変えると alpha の順位だけが崩れ、beta / c_black の
    順位は保たれる、という予測になる。Spearman の rho をブートストラップ信頼
    区間つきで出し、さらに **rho(alpha) - rho(beta) の差**を同じリサンプルで
    直接推定する（同一参加者上の従属な 2 相関の比較なので、解析的な検定より
    ブートストラップが素直）。

    python -m dualrdk.pomdp.ladder_stats \\
        --input outputs/pomdp/ladder/per_subject.csv \\
        --out   outputs/pomdp/ladder/stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PARAMS = ("alpha_in", "alpha_out", "beta", "c_black", "delta_alpha")
RANK_PARAMS = ("alpha_in", "alpha_out", "beta", "c_black")
N_BOOT = 10000
CI_LEVEL = 0.95


# --------------------------------------------------------------------------
# 入力
# --------------------------------------------------------------------------
def load_wide(path: Path) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """per_subject.csv を段ごとの DataFrame に割り、全段に揃う参加者だけ残す。

    段によって参加者集合がずれると対応のある検定が組めないので、共通部分を
    取る。落ちた参加者は呼び出し側で報告する。
    """
    df = pd.read_csv(path)
    missing = [c for c in ("rung", "subject_id", *PARAMS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 必要な列が無い: {missing}")
    df["subject_id"] = df["subject_id"].astype(str)

    rungs = list(dict.fromkeys(df["rung"]))  # ファイル中の出現順を保つ
    wide = {r: d.set_index("subject_id").sort_index() for r, d in df.groupby("rung")}

    common = set.intersection(*(set(w.index) for w in wide.values()))
    dropped = sorted(set(df["subject_id"]) - common)
    common = sorted(common)
    return {r: wide[r].loc[common] for r in rungs}, common, dropped


# --------------------------------------------------------------------------
# 問1: Delta alpha の検定
# --------------------------------------------------------------------------
def _boot_mean_ci(x: np.ndarray, rng, n_boot=N_BOOT, level=CI_LEVEL):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def test_delta_alpha(wide: dict[str, pd.DataFrame], *, seed=0, n_boot=N_BOOT) -> pd.DataFrame:
    """段ごとに Delta alpha = 0 を検定し、同等性の上限も出す。"""
    rng = np.random.default_rng(seed)
    rows = []
    for rung, d in wide.items():
        x = d["delta_alpha"].to_numpy(dtype=float)
        n = len(x)
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        se = sd / np.sqrt(n)

        t_res = stats.ttest_1samp(x, 0.0)
        ci95 = t_res.confidence_interval(0.95)
        # 同等性: 90% CI が (-d, d) に収まる最小の d（TOST と等価）
        ci90 = t_res.confidence_interval(0.90)
        equiv = float(max(abs(ci90.low), abs(ci90.high)))

        nz = x[x != 0.0]
        if len(nz) >= 1:
            w_res = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        else:
            w_stat, w_p = np.nan, np.nan

        n_pos, n_neg = int((x > 0).sum()), int((x < 0).sum())
        sign_p = (
            float(stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue)
            if n_pos + n_neg > 0 else np.nan
        )

        b_lo, b_hi = _boot_mean_ci(x, rng, n_boot=n_boot)
        rows.append({
            "rung": rung, "n": n,
            "mean_delta": mean, "sd": sd, "se": se, "median_delta": float(np.median(x)),
            "t": float(t_res.statistic), "df": n - 1, "p_t": float(t_res.pvalue),
            "cohens_dz": mean / sd if sd > 0 else np.nan,
            "ci95_lo": float(ci95.low), "ci95_hi": float(ci95.high),
            "boot_ci95_lo": b_lo, "boot_ci95_hi": b_hi,
            "wilcoxon_W": w_stat, "p_wilcoxon": w_p,
            "n_positive": n_pos, "n_negative": n_neg, "n_zero": int((x == 0).sum()),
            "p_sign": sign_p,
            "equivalence_bound_supported": equiv,
        })

    out = pd.DataFrame(rows)
    out["p_t_holm"] = _holm(out["p_t"].to_numpy())
    return out


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni。段どうしは独立でないので参考値（docstring 参照）。"""
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------
# 問2: 順位の保存
# --------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def _boot_rho(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([_spearman(x[i], y[i]) for i in idx])


def _pct_ci(v: np.ndarray, level=CI_LEVEL) -> tuple[float, float, int]:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return float(np.percentile(v, lo)), float(np.percentile(v, hi)), len(v)


def test_rank_preservation(
    wide: dict[str, pd.DataFrame], *, seed=0, n_boot=N_BOOT, reference=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """段の対ごとに Spearman rho と、rho(alpha) - rho(beta) の差を出す。

    リサンプルは段・パラメータをまたいで**共通の参加者インデックス**を使う。
    そうしないと相関どうしの差のブートストラップ分布が意味を持たない。
    """
    rng = np.random.default_rng(seed)
    rungs = list(wide)
    n = len(next(iter(wide.values())))
    idx = rng.integers(0, n, size=(n_boot, n))

    ref = reference or rungs[0]
    pairs = [(ref, r) for r in rungs if r != ref]
    pairs += [(a, b) for a, b in zip(rungs, rungs[1:]) if (a, b) not in pairs]

    rho_rows, boot_store = [], {}
    for a, b in pairs:
        for p in RANK_PARAMS:
            xa = wide[a][p].to_numpy(dtype=float)
            xb = wide[b][p].to_numpy(dtype=float)
            boot = _boot_rho(xa, xb, idx)
            lo, hi, n_ok = _pct_ci(boot)
            boot_store[(a, b, p)] = boot
            rho_rows.append({
                "rung_x": a, "rung_y": b, "parameter": p, "n": n,
                "spearman_rho": _spearman(xa, xb),
                "pearson_r": float(np.corrcoef(xa, xb)[0, 1]) if np.ptp(xa) and np.ptp(xb) else np.nan,
                "boot_ci95_lo": lo, "boot_ci95_hi": hi, "n_boot_valid": n_ok,
            })

    # alpha の順位崩れが beta / c_black より大きいか（同じリサンプル上の差）
    diff_rows = []
    for a, b in pairs:
        for target in ("alpha_in", "alpha_out"):
            for anchor in ("beta", "c_black"):
                d = boot_store[(a, b, target)] - boot_store[(a, b, anchor)]
                lo, hi, n_ok = _pct_ci(d)
                dv = d[np.isfinite(d)]
                diff_rows.append({
                    "rung_x": a, "rung_y": b,
                    "contrast": f"rho({target}) - rho({anchor})",
                    "estimate": float(
                        _spearman(wide[a][target].to_numpy(float), wide[b][target].to_numpy(float))
                        - _spearman(wide[a][anchor].to_numpy(float), wide[b][anchor].to_numpy(float))
                    ),
                    "boot_ci95_lo": lo, "boot_ci95_hi": hi,
                    "P(diff<0)": float((dv < 0).mean()) if n_ok else np.nan,
                    "n_boot_valid": n_ok,
                })
    return pd.DataFrame(rho_rows), pd.DataFrame(diff_rows)


# --------------------------------------------------------------------------
# 報告
# --------------------------------------------------------------------------
def _fmt(v, nd=3):
    return "  nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def print_report(delta: pd.DataFrame, rho: pd.DataFrame, diff: pd.DataFrame,
                 subjects: list[str], dropped: list[str]) -> None:
    print(f"\n参加者 n = {len(subjects)}" + (f"（全段に揃わず除外: {dropped}）" if dropped else ""))

    print("\n=== 問1: Delta alpha = alpha_out - alpha_in は 0 と異なるか ===")
    cols = ["rung", "n", "mean_delta", "ci95_lo", "ci95_hi", "cohens_dz",
            "p_t", "p_wilcoxon", "p_sign", "n_positive", "n_negative"]
    print(delta[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  ブートストラップ平均の 95% CI（t 区間の頑健性チェック）:")
    for _, r in delta.iterrows():
        print(f"    {r['rung']:<24s} [{r['boot_ci95_lo']:+.4f}, {r['boot_ci95_hi']:+.4f}]")

    print("\n  同等性（このデータが支持できる最小マージン d: |Delta| < d と言える上限）:")
    for _, r in delta.iterrows():
        print(f"    {r['rung']:<24s} d = {r['equivalence_bound_supported']:.3f}")
    print("  ※ p > 0.05 は「差が無い」ではない。言えるのは「差があっても d 未満」まで。")

    print("\n=== 問2: 参加者ごとの推定値の順位は段間で保存されるか（Spearman rho）===")
    piv = rho.pivot_table(index=["rung_x", "rung_y"], columns="parameter",
                          values="spearman_rho", sort=False)
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  95% ブートストラップ CI:")
    for _, r in rho.iterrows():
        print(f"    {r['rung_x']:<24s} -> {r['rung_y']:<24s} {r['parameter']:<10s} "
              f"rho={_fmt(r['spearman_rho'])} [{_fmt(r['boot_ci95_lo'])}, {_fmt(r['boot_ci95_hi'])}]")

    print("\n=== 問2b: alpha の順位崩れは beta / c_black より大きいか ===")
    print("  （負で 0 を含まなければ「alpha だけが崩れている」と言える）")
    for _, r in diff.iterrows():
        star = " *" if np.isfinite(r["boot_ci95_hi"]) and r["boot_ci95_hi"] < 0 else ""
        print(f"    {r['rung_x']:<24s} -> {r['rung_y']:<24s} {r['contrast']:<28s} "
              f"{_fmt(r['estimate'])} [{_fmt(r['boot_ci95_lo'])}, {_fmt(r['boot_ci95_hi'])}]"
              f"  P(<0)={_fmt(r['P(diff<0)'])}{star}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="ladder.py の per_subject.csv を検定する")
    ap.add_argument("--input", type=Path, default=Path("outputs/pomdp/ladder/per_subject.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/ladder/stats"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default=None,
                    help="順位比較の基準にする段（既定はファイル中の最初の段）")
    args = ap.parse_args(argv)

    wide, subjects, dropped = load_wide(args.input)
    print(f"[load] {args.input}  段 {list(wide)}  参加者 {len(subjects)} 名")

    delta = test_delta_alpha(wide, seed=args.seed, n_boot=args.n_boot)
    rho, diff = test_rank_preservation(
        wide, seed=args.seed, n_boot=args.n_boot, reference=args.reference
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    delta.to_csv(out / "delta_alpha_tests.csv", index=False)
    rho.to_csv(out / "rank_preservation.csv", index=False)
    diff.to_csv(out / "rho_contrasts.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "input": str(args.input),
        "n_subjects": len(subjects),
        "subjects_dropped": dropped,
        "rungs": list(wide),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "reference": args.reference or list(wide)[0],
        "note": "段どうしは同一データの再解析なので独立な検定族ではない。"
                "多重比較補正（p_t_holm）は参考値。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(delta, rho, diff, subjects, dropped)
    print(f"\n[out] {out}/delta_alpha_tests.csv, rank_preservation.csv, rho_contrasts.csv, meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
