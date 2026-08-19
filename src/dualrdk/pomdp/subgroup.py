"""参加者をモデル適合で層別し、層ごとに Delta alpha を推定する（副次解析）。

手続き:
    1. M0（学習なし）と M2z（学習あり）を同じデータに当てはめる
    2. 参加者ごとにどちらの当てはまりが良いかで 2 群に分ける
    3. 各群で M2z を推定し直し、alpha_out - alpha_in を比べる

**これは選択と推定に同じデータを使う手続きである（double dipping）。**
M2z が M0 に勝つかどうかは「その参加者が学習したか」をほぼそのまま測っており
（実データで M2z 優位度と正解率の相関 0.826）、学習量は alpha と直結する。
したがって learner 群の alpha 分布は上に切り詰められ、Delta alpha の推定も
偏りうる。**主解析にしてはならない。副次解析として、選択に使った統計量と
推定対象が依存関係にあることを明記して報告すること。**

偏りの目安は `--random-split` で得られる。同じ群サイズで参加者をランダムに
分け、同じ推定を走らせた結果を対照に置く。層別に意味があるなら、実際の層別で
得た群差はランダム分割の群差より大きくなるはずである。1 本のランダム分割は
検定にはならないが、「群差がサイズ由来の推定誤差で説明できる程度か」の目安に
はなる。

出力は既存の結果を上書きしないよう、独立したツリーに書く:

    outputs/pomdp/subgroup/<tag>/
        assignment.csv          参加者ごとの指標と割り当て
        learner_subjects.txt    群ごとの subject_id
        nonlearner_subjects.txt
        fit/learner/            群ごとの推定結果（fit.py と同じ形）
        fit/nonlearner/
        comparison.json         群間の Delta alpha 比較
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dualrdk.pomdp import data as pdata
from dualrdk.pomdp import evaluate as ev

METRICS = ("choice_log_lik", "agreement")
# 群を分ける最小差。指標がほぼ同点の参加者を「学習した」側に入れないための余裕。
# choice_log_lik は nats、agreement は割合なので既定値を分けている。
DEFAULT_MARGIN = {"choice_log_lik": 0.0, "agreement": 0.0}


def per_subject_metrics(fit_dir: Path, tidy: pd.DataFrame) -> pd.DataFrame:
    latent = pd.read_csv(Path(fit_dir) / "latent.csv")
    df = ev.merge_latent(latent, tidy)
    return ev.per_subject(df).set_index("subject_id")


def classify(
    baseline_dir: Path,
    learning_dir: Path,
    tidy: pd.DataFrame,
    *,
    metric: str = "choice_log_lik",
    margin: float | None = None,
) -> pd.DataFrame:
    """参加者ごとに、学習モデルがベースラインを上回るかで層別する。

    metric="choice_log_lik" を既定にしている。agreement（argmax の一致率）は
    予測の確信度を捨てるため、僅差の参加者で不安定になりやすい。
    """
    if metric not in METRICS:
        raise ValueError(f"metric は {METRICS} のいずれか: {metric!r}")
    if margin is None:
        margin = DEFAULT_MARGIN[metric]

    base = per_subject_metrics(baseline_dir, tidy)
    learn = per_subject_metrics(learning_dir, tidy)
    common = [s for s in learn.index if s in base.index]

    out = pd.DataFrame(
        {
            "subject_id": common,
            "n_trials": learn.loc[common, "n_trials"].to_numpy(),
            "correct_rate": learn.loc[common, "correct_rate"].to_numpy(),
            f"baseline_{metric}": base.loc[common, metric].to_numpy(),
            f"learning_{metric}": learn.loc[common, metric].to_numpy(),
        }
    )
    out["advantage"] = out[f"learning_{metric}"] - out[f"baseline_{metric}"]
    out["group"] = np.where(out["advantage"] > margin, "learner", "nonlearner")
    return out.sort_values("advantage", ascending=False).reset_index(drop=True)


def random_split(assignment: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """群サイズを保ったままランダムに割り当て直す（対照条件）。

    層別が意味を持つなら、実際の層別で得た群差はこの対照より大きくなるはず。
    小標本での推定誤差がどれくらいの見かけの群差を生むかの目安になる。
    """
    out = assignment.copy()
    rng = np.random.default_rng(seed)
    labels = out["group"].to_numpy().copy()
    rng.shuffle(labels)
    out["group"] = labels
    return out


def write_subject_lists(assignment: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for group, sub in assignment.groupby("group"):
        p = out_dir / f"{group}_subjects.txt"
        p.write_text("\n".join(sub["subject_id"].astype(str)) + "\n", encoding="utf-8")
        paths[group] = p
    return paths


def run_group_fits(
    subject_lists: dict[str, Path],
    *,
    model: str,
    input_path: Path,
    out_dir: Path,
    method: str,
    chains: int,
    draws: int,
    warmup: int,
    fix_perceptual: Path | None,
    seed: int,
) -> dict[str, Path]:
    """群ごとに fit.py をサブプロセスで走らせる。

    同一プロセスで NumPyro を繰り返し呼ぶとホストデバイス設定やコンパイル
    キャッシュが絡むため、独立プロセスにして再現性を優先する。

    片方の群が収束基準（FR-5.5）を落としても中断しない。群は独立に推定して
    いるので、落ちた方だけを報告すればよく、成功した方の結果まで失う理由が
    無いため。ただし **成功扱いにはせず**、diagnostics の passes で下流に
    伝える（compare_groups が読む）。
    """
    fits = {}
    for group, list_path in subject_lists.items():
        target = Path(out_dir) / "fit" / group
        cmd = [
            sys.executable, "-m", "dualrdk.pomdp.fit",
            "--model", model,
            "--method", method,
            "--input", str(input_path),
            "--subjects-file", str(list_path),
            "--out", str(target),
            "--seed", str(seed),
            "--contrast-out", str(target / "contrast.json"),
        ]
        if method == "nuts":
            cmd += ["--chains", str(chains), "--draws", str(draws), "--warmup", str(warmup)]
        if fix_perceptual:
            cmd += ["--fix-perceptual", str(fix_perceptual)]
        print(f"[subgroup] {group}: {' '.join(cmd[2:])}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            if not (target / "latent.csv").exists():
                raise SystemExit(f"[subgroup] {group} の推定が結果を残さずに失敗した（rc={rc}）")
            print(f"[warn] {group}: fit.py が非ゼロ終了（rc={rc}）。"
                  "収束基準未達の可能性がある。comparison.json の diagnostics を確認すること")
        fits[group] = target
    return fits


def _contrast(fit_dir: Path) -> dict | None:
    p = Path(fit_dir) / "contrast.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    for c in payload.get("contrasts", []):
        if c["parameter"] == "contrast_alpha":
            return c
    return None


def compare_groups(fits: dict[str, Path], assignment: pd.DataFrame, tidy: pd.DataFrame) -> dict:
    """群ごとの Delta alpha と、群間の差の事後分布を出す。"""
    import arviz as az

    out = {"groups": {}}
    draws = {}
    for group, path in fits.items():
        sub = assignment[assignment["group"] == group]
        entry = {
            "n_subjects": int(len(sub)),
            "median_correct_rate": float(sub["correct_rate"].median()),
            "median_advantage": float(sub["advantage"].median()),
            "contrast_alpha": _contrast(path),
        }
        trace = Path(path) / "trace.nc"
        if trace.exists():
            idata = az.from_netcdf(trace)
            if "contrast_alpha" in idata["posterior"].data_vars:
                draws[group] = np.asarray(idata["posterior"]["contrast_alpha"]).ravel()
            entry["diagnostics"] = json.loads(
                (Path(path) / "diagnostics.json").read_text(encoding="utf-8")
            )
        out["groups"][group] = entry

    failed = [g for g, e in out["groups"].items()
              if not e.get("diagnostics", {}).get("passes", True)]
    if failed:
        out["diagnostics_failed"] = failed

    if len(draws) == 2:
        a, b = "learner", "nonlearner"
        if a in draws and b in draws:
            # 群は独立に推定しているので、事後の差は draw をランダムに対応づけて作る
            n = min(len(draws[a]), len(draws[b]))
            rng = np.random.default_rng(0)
            d = rng.permutation(draws[a])[:n] - rng.permutation(draws[b])[:n]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["group_difference"] = {
                "definition": "contrast_alpha(learner) - contrast_alpha(nonlearner)",
                "mean": float(d.mean()),
                "sd": float(d.std(ddof=1)),
                "hdi_2.5%": float(lo),
                "hdi_97.5%": float(hi),
                "P(diff>0)": float(np.mean(d > 0)),
            }
    out["caveat"] = (
        "層別に使った統計量（M2z - M0 の当てはまり差）は alpha と依存関係にある。"
        "double dipping であり、主解析ではなく副次解析として報告すること。"
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="モデル適合による参加者層別と、層ごとの Delta alpha 比較（副次解析）"
    )
    ap.add_argument("--baseline-fit", type=Path, required=True, help="M0 の推定結果ディレクトリ")
    ap.add_argument("--learning-fit", type=Path, required=True, help="M2z の推定結果ディレクトリ")
    ap.add_argument("--input", type=Path, required=True, help="tidy テーブル")
    ap.add_argument("--out", type=Path, required=True, help="出力先（既存結果とは別のツリー）")
    ap.add_argument("--metric", default="choice_log_lik", choices=METRICS)
    ap.add_argument("--margin", type=float, default=None, help="層別の余裕。既定は 0")
    ap.add_argument("--model", default="M2z", help="群ごとに推定し直すモデル")
    ap.add_argument("--method", default="nuts", choices=("nuts", "map"))
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--fix-perceptual", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classify-only", action="store_true", help="層別だけして推定はしない")
    ap.add_argument(
        "--random-split", type=int, default=None, metavar="SEED",
        help="対照条件。群サイズを保ったままランダムに割り当てて同じ推定を走らせる",
    )
    args = ap.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"[warn] {args.out} は空でない。同名ファイルは上書きされる")
    args.out.mkdir(parents=True, exist_ok=True)

    tidy = pdata.load_tidy(args.input)
    assignment = classify(
        args.baseline_fit, args.learning_fit, tidy,
        metric=args.metric, margin=args.margin,
    )
    if args.random_split is not None:
        assignment = random_split(assignment, seed=args.random_split)
        print(f"[subgroup] 対照条件：ランダム分割（seed={args.random_split}）")
    assignment.to_csv(args.out / "assignment.csv", index=False)
    lists = write_subject_lists(assignment, args.out)

    counts = assignment["group"].value_counts().to_dict()
    print(f"[subgroup] metric={args.metric}  {counts}")
    print(
        assignment.groupby("group")
        .agg(n=("subject_id", "size"), correct_rate=("correct_rate", "median"),
             advantage=("advantage", "median"))
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    (args.out / "config.json").write_text(
        json.dumps(
            {
                "baseline_fit": str(args.baseline_fit),
                "learning_fit": str(args.learning_fit),
                "metric": args.metric,
                "margin": args.margin,
                "model": args.model,
                "method": args.method,
                "counts": counts,
                "random_split_seed": args.random_split,
                "is_control": args.random_split is not None,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.classify_only:
        print(f"[saved] {args.out}/assignment.csv, *_subjects.txt")
        return assignment

    if len(lists) < 2:
        raise SystemExit(f"片方の群が空なので比較できない: {counts}")

    fits = run_group_fits(
        lists, model=args.model, input_path=args.input, out_dir=args.out,
        method=args.method, chains=args.chains, draws=args.draws, warmup=args.warmup,
        fix_perceptual=args.fix_perceptual, seed=args.seed,
    )
    summary = compare_groups(fits, assignment, tidy)
    (args.out / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out}/comparison.json")
    if summary.get("diagnostics_failed"):
        print(f"[warn] 収束基準（FR-5.5）未達の群: {summary['diagnostics_failed']}。"
              "draws を増やして再実行すること")
    return summary


if __name__ == "__main__":
    main()
