"""モデル比較（§5.5）と、フィット前のモデル非依存チェック（FR-3.2 / AC-2）。

PSIS-LOO の観測単位は **参加者**（leave-one-subject-out）。試行はパラメータ
条件付きでも独立でなく V_t / b_t を通じて逐次依存しているため、
leave-one-trial-out は不正である（FR-5.7）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from dualrdk.pomdp import data as pdata
from dualrdk.pomdp.task_reward import wrap_deg


# --------------------------------------------------------------------------
# 事前チェック：2 峰性（FR-3.2 / AC-2）
# --------------------------------------------------------------------------
def bimodality_check(tidy: pd.DataFrame, out_png: Path | None = None) -> dict:
    """反応角を「2 クラウドの相対座標」に写してヒストグラムを描く。

    2 峰性は方策の **仮定** であって検証結果ではない。中間に質量があると
    P1 も P2 も不適切なので、フィット前に必ず確認する。

    相対座標: 白の方向を 0、黒の方向を Delta_t に写した座標系での a_t。
    Delta_t は試行ごとに変わるので、白基準の符号付き差 e_w と、黒基準の
    符号付き差 e_b の両方を返す。
    """
    v = tidy[tidy["valid"]]
    e_w = wrap_deg(v["response_deg"].to_numpy() - v["theta_white_deg"].to_numpy())
    e_b = wrap_deg(v["response_deg"].to_numpy() - v["theta_black_deg"].to_numpy())
    nearest = np.minimum(np.abs(e_w), np.abs(e_b))

    report = {
        "n_valid": int(len(v)),
        "frac_within_45_of_a_cloud": float(np.mean(nearest <= 45.0)),
        "frac_intermediate_45_to_90": float(np.mean((nearest > 45.0) & (nearest <= 90.0))),
        "frac_far_gt_90": float(np.mean(nearest > 90.0)),
        "median_nearest_error_deg": float(np.median(nearest)),
    }

    if out_png is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].hist(e_w, bins=72, range=(-180, 180), color="0.3")
        axes[0].axvline(0, color="C0", lw=1)
        axes[0].set_title("response - white direction [deg]")
        axes[1].hist(nearest, bins=36, range=(0, 180), color="0.3")
        axes[1].axvline(45, color="C3", lw=1, ls="--")
        axes[1].set_title("distance to nearest cloud [deg]")
        for ax in axes:
            ax.set_ylabel("count")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    return report


# --------------------------------------------------------------------------
# PSIS-LOO
# --------------------------------------------------------------------------
def loo_table(trace_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    loos, rows = {}, []
    for name, path in trace_paths.items():
        idata = az.from_netcdf(path)
        res = az.loo(idata, var_name="subject", pointwise=True)
        loos[name] = res
        k = np.asarray(getattr(res, "pareto_k", np.array([])))
        rows.append(
            {
                "model": name,
                "elpd_loo": float(res.elpd),
                "se": float(res.se),
                "p_loo": float(res.p),
                "n_obs": int(res.n_data_points),
                "pareto_k_max": float(k.max()) if k.size else float("nan"),
                "pareto_k_gt_0.7_frac": float(np.mean(k > 0.7)) if k.size else float("nan"),
            }
        )
    df = pd.DataFrame(rows).sort_values("elpd_loo", ascending=False).reset_index(drop=True)
    return df, loos


# --------------------------------------------------------------------------
# 族をまたぐ共通表（FR-6.5）
# --------------------------------------------------------------------------
def common_table(fits: dict[str, Path], tidy: pd.DataFrame, holdout_from: int | None = None):
    """一致率と選択のみ対数尤度で、全モデル・全推定法を 1 枚の表にする。

    ELPD をここに含めないのは意図的である。M1 系は 2 値選択の確率質量、
    M2 / M3 系は円環上の確率密度を予測しており、単位が違うので同じ列に
    並べてはならない。角度込みの比較は同じ族の中だけで行うこと。
    """
    from dualrdk.pomdp import evaluate as ev
    from dualrdk.pomdp.models import BINARY_MODELS, build_model

    rows = []
    for label, path in fits.items():
        path = Path(path)
        latent = pd.read_csv(path / "latent.csv")
        cfg = {}
        if (path / "fit_config.json").exists():
            cfg = json.loads((path / "fit_config.json").read_text(encoding="utf-8"))
        # ホールドアウト集計の起点。--train-trials 付きで推定されていれば
        # そこから先が真の外挿になる。指定が無ければ後半の集計にすぎない。
        cut = holdout_from if holdout_from is not None else cfg.get("train_trials")
        cut = None if cut is None else int(cut) + (1 if cfg.get("train_trials") == cut else 0)
        summ = ev.evaluate(latent, tidy, holdout_from=cut)
        o = summ["overall"]
        model_name = cfg.get("model") or label.split(":")[-1]
        n_param = cfg.get("n_params")
        if n_param is None:
            try:
                n_param = len(build_model(model_name).param_names)
            except ValueError:
                n_param = None
        row = {
            "fit": label,
            "method": cfg.get("method", "?"),
            "n_params": n_param,
            "observation_space": "binary" if model_name in BINARY_MODELS else "angle",
            "agreement": o["agreement"],
            "mean_p_observed": o["mean_p_observed"],
            "choice_log_lik": o["choice_log_lik"],
            "mcfadden_r2": o["mcfadden_r2"],
            "predicted_correct_rate": o["predicted_correct_rate"],
            "n_below_chance": summ["per_subject"]["n_below_chance"],
        }
        if "holdout" in summ:
            oos = cfg.get("train_trials") is not None
            row["holdout_is_out_of_sample"] = oos
            row["agreement_holdout"] = summ["holdout"]["agreement"]
            row["choice_log_lik_holdout"] = summ["holdout"]["choice_log_lik"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("choice_log_lik", ascending=False)
    return df.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="モデル比較と事前チェック")
    ap.add_argument("--precheck", action="store_true", help="2 峰性のモデル非依存チェック")
    ap.add_argument("--models", type=str, help="カンマ区切りのモデル名")
    ap.add_argument("--fit-root", type=Path, default=Path("outputs/pomdp/fit"))
    ap.add_argument(
        "--common", type=str, default=None,
        help="共通表を作る。'ラベル=ディレクトリ' をカンマ区切りで指定する"
             "（ラベルの末尾はモデル名にする。例 'map:M1z=outputs/pomdp/fit_map/M1z'）",
    )
    ap.add_argument("--holdout-from", type=int, default=None)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/comparison"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.common:
        if not args.input:
            ap.error("--common には --input が必要")
        fits = {}
        for item in args.common.split(","):
            label, _, path = item.strip().partition("=")
            fits[label] = Path(path)
        df = common_table(fits, pdata.load_tidy(args.input), holdout_from=args.holdout_from)
        df.to_csv(args.out / "common.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n[saved] {args.out}/common.csv")
        print("[note] observation_space が binary と angle の行の間では、"
              "角度込みの尤度・ELPD は比較できない。上の列は選択のみの共通指標。")
        if "holdout_is_out_of_sample" in df.columns and not df["holdout_is_out_of_sample"].all():
            print("[warn] holdout_is_out_of_sample=False の行は、全試行で推定した"
                  "モデルの後半だけを集計したもので、外挿性能ではない。"
                  "推定法どうしを比べるには --train-trials 付きで推定し直すこと。")
        return

    if args.precheck:
        if args.input:
            tidy = pdata.load_tidy(args.input)
        elif args.data_dir:
            tidy = pdata.load_from_raw(args.data_dir)
        else:
            ap.error("--precheck には --input か --data-dir が必要")
        rep = bimodality_check(tidy, args.out / "precheck_bimodality.png")
        (args.out / "precheck_bimodality.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8"
        )
        print("[precheck]", json.dumps(rep, indent=2))
        if rep["frac_intermediate_45_to_90"] > 0.10:
            print(
                "[warn] 中間帯（45-90 度）に反応の 10% 超が存在する。"
                "2 峰混合方策の仮定が疑わしい（AC-2）"
            )
        return

    if not args.models:
        ap.error("--models か --precheck のどちらかが必要")

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    paths = {n: args.fit_root / n / "trace.nc" for n in names}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"trace.nc が見つからない: {missing}")

    df, _ = loo_table(paths)
    df.to_csv(args.out / "loo.csv", index=False)
    print(df.to_string(index=False))
    bad = df[df["pareto_k_gt_0.7_frac"] > 0.01]
    if len(bad):
        print(f"[warn] AC-7 未達（pareto_k > 0.7 が 1% 超）: {list(bad['model'])}")


if __name__ == "__main__":
    main()
