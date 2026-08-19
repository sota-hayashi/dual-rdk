"""モデル評価の共通指標（FR-6.5）。

**すべてのモデル族・すべての推定法に対して同じ指標を出す**ことがこのモジュール
の目的である。M1 系（2 値選択）と M2 / M3 系（円環上の応答角）は観測空間が
違うため、`log_pi` を直接比べてはならない。共通通貨は次の 2 つだけである。

    一致率          モデルの予測選択が参加者の実際の選択と一致した割合
    選択のみ対数尤度 log P(参加者が実際に選んだ雲)  ← 確率質量。族をまたげる

「ターゲット選択確率の平均」は共通指標に**しない**。参加者が実際にターゲットを
選んだ割合（本データで 0.617）を超えていても、それは過大予測を意味するだけで
妥当性の証拠にならないため。評価は常に「参加者の実際の選択」に対して行う。

入力は fit.py が書く latent.csv（subject_id, trial, p_w, ...）と tidy テーブル。
どちらの推定法の出力でも同じ形なので、NUTS でも MAP でもそのまま渡せる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dualrdk.pomdp import data as pdata

# 報酬 > 0 の境界（§0.3）。「参加者がターゲットを取れたか」の判定に使う。
REWARD_RADIUS_DEG = 45.0
CALIBRATION_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
BLOCK_SIZE = 8


def circ_dist_deg(a, b):
    d = np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0)
    return np.minimum(d, 360.0 - d)


def add_observed_choice(tidy: pd.DataFrame) -> pd.DataFrame:
    """参加者の観測選択と正誤を tidy に付与する。

    chose_white  : 応答角がどちらの雲に近いか（等距離は白としない）
    white_target : 白が高報酬側か
    correct      : ターゲット側の雲を選んだか
    rewarded     : 実際に報酬を得たか（角度誤差 45 度以内）
    """
    out = tidy.copy()
    d_w = circ_dist_deg(out["response_deg"], out["theta_white_deg"])
    d_b = circ_dist_deg(out["response_deg"], out["theta_black_deg"])
    out["chose_white"] = d_w < d_b
    out["white_target"] = (
        circ_dist_deg(out["target_deg"], out["theta_white_deg"]) < 1e-9
    )
    out["correct"] = out["chose_white"] == out["white_target"]
    out["rewarded"] = circ_dist_deg(out["response_deg"], out["target_deg"]) <= REWARD_RADIUS_DEG
    out["abs_err_deg"] = np.where(
        out["chose_white"],
        circ_dist_deg(out["response_deg"], out["theta_white_deg"]),
        circ_dist_deg(out["response_deg"], out["theta_black_deg"]),
    )
    return out


def merge_latent(latent: pd.DataFrame, tidy: pd.DataFrame) -> pd.DataFrame:
    """latent.csv と tidy を突き合わせ、有効試行だけを返す。"""
    obs = add_observed_choice(tidy)
    cols = ["subject_id", "trial", "p_w"]
    extra = [c for c in ("d", "q", "log_pi") if c in latent.columns]
    df = obs.merge(latent[cols + extra], on=["subject_id", "trial"], how="inner")
    if len(df) != len(obs):
        raise ValueError(
            f"latent と tidy の突き合わせに失敗（tidy {len(obs)} 行 -> 結合 {len(df)} 行）"
        )
    return df[df["valid"]].reset_index(drop=True)


def _choice_log_lik(p_w: np.ndarray, chose_white: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_obs = np.where(chose_white, p_w, 1.0 - p_w)
    return np.log(np.clip(p_obs, eps, 1.0))


def _summary_block(df: pd.DataFrame) -> dict:
    p_w = df["p_w"].to_numpy(dtype=float)
    cw = df["chose_white"].to_numpy(dtype=bool)
    p_obs = np.where(cw, p_w, 1.0 - p_w)
    ll = _choice_log_lik(p_w, cw)
    n = len(df)
    ll_null = n * np.log(0.5)
    return {
        "n_trials": int(n),
        "n_subjects": int(df["subject_id"].nunique()),
        "agreement": float(np.mean((p_w > 0.5) == cw)),
        "mean_p_observed": float(p_obs.mean()),
        "choice_log_lik": float(ll.sum()),
        "choice_log_lik_per_trial": float(ll.mean()),
        "mcfadden_r2": float(1.0 - ll.sum() / ll_null),
        # ベースライン。モデルはこれらを超えて初めて意味を持つ
        "baseline_chance": 0.5,
        "baseline_majority": float(max(df["correct"].mean(), 1.0 - df["correct"].mean())),
        "participant_correct_rate": float(df["correct"].mean()),
        "participant_rewarded_rate": float(df["rewarded"].mean()),
        # モデルが予測する正解率。参加者の実測より高ければ過大予測
        "predicted_correct_rate": float(
            np.where(df["white_target"].to_numpy(dtype=bool), p_w, 1.0 - p_w).mean()
        ),
    }


def calibration_table(df: pd.DataFrame, bins=CALIBRATION_BINS) -> pd.DataFrame:
    """p_w のビンごとに、予測と実際の白選択率を並べる。

    一致率だけでは「どの確率帯でずれているか」が見えないため（FR-5.8 と同じ
    理由で、単一のスカラーに落とさない）。
    """
    work = df.copy()
    work["bin"] = pd.cut(work["p_w"], bins, include_lowest=True)
    tab = work.groupby("bin", observed=True).agg(
        n=("chose_white", "size"),
        predicted=("p_w", "mean"),
        observed=("chose_white", "mean"),
    )
    # 二項の標準誤差で「ずれが偶然か」を判断できるようにする
    tab["se"] = np.sqrt(tab["predicted"] * (1 - tab["predicted"]) / tab["n"])
    tab["z"] = (tab["observed"] - tab["predicted"]) / tab["se"]
    return tab.reset_index()


def learning_curve(df: pd.DataFrame, block: int = BLOCK_SIZE) -> pd.DataFrame:
    """ブロックごとの観測正解率とモデル予測正解率。水準のずれを見る。"""
    work = df.copy()
    work["block"] = (work["trial"] - 1) // block
    p_correct = np.where(
        work["white_target"].to_numpy(dtype=bool),
        work["p_w"].to_numpy(dtype=float),
        1.0 - work["p_w"].to_numpy(dtype=float),
    )
    work["p_correct"] = p_correct
    return (
        work.groupby("block")
        .agg(n=("correct", "size"), observed=("correct", "mean"), predicted=("p_correct", "mean"))
        .reset_index()
    )


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    p_w = work["p_w"].to_numpy(dtype=float)
    cw = work["chose_white"].to_numpy(dtype=bool)
    work["match"] = (p_w > 0.5) == cw
    work["p_obs"] = np.where(cw, p_w, 1.0 - p_w)
    work["cll"] = _choice_log_lik(p_w, cw)
    return (
        work.groupby("subject_id")
        .agg(
            n_trials=("match", "size"),
            agreement=("match", "mean"),
            mean_p_observed=("p_obs", "mean"),
            choice_log_lik=("cll", "sum"),
            correct_rate=("correct", "mean"),
        )
        .reset_index()
    )


def evaluate(latent: pd.DataFrame, tidy: pd.DataFrame, *, holdout_from: int | None = None) -> dict:
    """共通指標一式を計算する。

    holdout_from : 指定するとその試行番号以降だけの指標も併記する。MAP と階層
        ベイズを同じインサンプル一致率で比べると、参加者ごとに自由に当てはめる
        MAP が構造的に必ず勝つ（それは妥当性ではなく過適合）。前半で推定し
        後半で評価した数字を並べないと、この 2 つは比較できない。
    """
    df = merge_latent(latent, tidy)
    out = {"overall": _summary_block(df)}

    by_zone = {}
    for z, sub in df.groupby("zone"):
        label = "out_of_zone" if z == 1 else "in_zone"
        by_zone[label] = _summary_block(sub)
    out["by_zone"] = by_zone

    if holdout_from is not None:
        held = df[df["trial"] >= holdout_from]
        if len(held):
            out["holdout"] = {"from_trial": int(holdout_from), **_summary_block(held)}

    ps = per_subject(df)
    out["per_subject"] = {
        "agreement_mean": float(ps["agreement"].mean()),
        "agreement_median": float(ps["agreement"].median()),
        "agreement_min": float(ps["agreement"].min()),
        "agreement_max": float(ps["agreement"].max()),
        "n_below_chance": int((ps["agreement"] < 0.5).sum()),
    }

    # 角度の再現（M1 系では p_w しか無いので参考値）
    err = df["abs_err_deg"].to_numpy(dtype=float)
    rad = np.deg2rad((df["response_deg"].to_numpy(dtype=float) - np.where(
        df["chose_white"], df["theta_white_deg"], df["theta_black_deg"]
    )) % 360.0)
    R = np.abs(np.mean(np.exp(1j * rad)))
    out["angular"] = {
        "median_abs_error_deg": float(np.median(err)),
        "circular_sd_deg": float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12))))),
    }
    return out


def plot_learning_curve(curve: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    x = curve["block"] * BLOCK_SIZE + BLOCK_SIZE / 2
    ax.plot(x, curve["observed"], "o-", label="observed")
    ax.plot(x, curve["predicted"], "s--", label="model")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xlabel("trial")
    ax.set_ylabel("P(target chosen)")
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="モデルと参加者の選択の一致を評価する（推定法・モデル族に非依存）"
    )
    ap.add_argument("--fit", type=Path, required=True, help="fit.py の出力ディレクトリ")
    ap.add_argument("--input", type=Path, required=True, help="tidy テーブル")
    ap.add_argument(
        "--holdout-from", type=int, default=None,
        help="この試行番号以降をホールドアウトとして別集計する（例: 25）",
    )
    args = ap.parse_args(argv)

    latent = pd.read_csv(args.fit / "latent.csv")
    tidy = pdata.load_tidy(args.input)
    df = merge_latent(latent, tidy)

    summary = evaluate(latent, tidy, holdout_from=args.holdout_from)
    (args.fit / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calibration_table(df).to_csv(args.fit / "calibration.csv", index=False)
    curve = learning_curve(df)
    curve.to_csv(args.fit / "learning_curve.csv", index=False)
    per_subject(df).to_csv(args.fit / "per_subject.csv", index=False)
    plot_learning_curve(curve, args.fit / "learning_curve.png")

    o = summary["overall"]
    print(f"[evaluate] {args.fit}")
    print(f"  一致率              {o['agreement']:.3f}")
    print(f"  平均予測確率        {o['mean_p_observed']:.3f}")
    print(f"  選択のみ対数尤度    {o['choice_log_lik']:.1f}  ({o['choice_log_lik_per_trial']:.3f}/試行)")
    print(f"  McFadden pseudo-R2  {o['mcfadden_r2']:.3f}")
    print(f"  偶然 / 多数派       {o['baseline_chance']:.3f} / {o['baseline_majority']:.3f}")
    print(f"  参加者正解率        {o['participant_correct_rate']:.3f}"
          f"  モデル予測 {o['predicted_correct_rate']:.3f}")
    if "holdout" in summary:
        h = summary["holdout"]
        print(f"  ホールドアウト(試行{h['from_trial']}以降) 一致率 {h['agreement']:.3f}")
    print(f"  一致率が偶然未満の参加者 {summary['per_subject']['n_below_chance']}"
          f"/{o['n_subjects']}")
    print(f"[saved] {args.fit}/evaluation.json, calibration.csv, learning_curve.*, per_subject.csv")
    return summary


if __name__ == "__main__":
    main()
