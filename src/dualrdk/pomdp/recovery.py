"""パラメータリカバリ（§5.6）。

事前分布からパラメータをサンプル -> 48 試行 x N 名のデータを生成 -> 推定 ->
真値と比較する。刺激と注意状態は実データから借りる（design-matched）。

最優先の検証項目は群レベルの `alpha_out - alpha_in`（M2z）の回復である。
真値-推定値の相関と、真値が事後 95% 区間に入る割合（カバレッジ）を報告する。
**ここが通らなければ以降の解析は成立しない**（FR-5.9 / AC-3）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dualrdk.pomdp import data as pdata
from dualrdk.pomdp.fit import CONTRAST_BASES, run_nuts, to_inference_data
from dualrdk.pomdp.likelihood import precompute_trial_features
from dualrdk.pomdp.models import _TRANSFORMS, MODEL_NAMES, build_model
from dualrdk.pomdp.simulate import simulate_batch, stim_arrays

# シミュレーション時の個人差の大きさ。事前分布 HalfNormal(1) をそのまま使うと
# 非現実的に大きな個人差が出るため、控えめな値を使う。
SIM_SIGMA_SCALE = 0.5

# AC-3 の合否を出すのに必要な最小反復数。少数の反復では相関が乱数と区別できず、
# 「通った」という誤った緑信号を出してしまうため。
MIN_SIM_FOR_VERDICT = 20


def draw_true_params(model, rng, n_subj):
    """事前分布から群レベル・個人レベルのパラメータを引く。"""
    mu_unc, sigma, unc = {}, {}, {}
    for ps in model.params:
        mu_unc[ps.name] = rng.normal(ps.prior_loc, ps.prior_scale)
        sigma[ps.name] = abs(rng.normal(0.0, SIM_SIGMA_SCALE))
        unc[ps.name] = jnp.asarray(
            mu_unc[ps.name] + sigma[ps.name] * rng.normal(0.0, 1.0, size=n_subj)
        )
    params = model.constrain(unc)
    params = {k: jnp.broadcast_to(jnp.asarray(v), (n_subj,)) for k, v in params.items()}

    pop = {ps.name: float(_TRANSFORMS[ps.transform](jnp.asarray(mu_unc[ps.name]))) for ps in model.params}
    contrasts = {}
    for base in CONTRAST_BASES:
        if f"{base}_in" in pop and f"{base}_out" in pop:
            contrasts[f"contrast_{base}"] = pop[f"{base}_out"] - pop[f"{base}_in"]
    return params, pop, contrasts


def one_replicate(model, stim, n_subj, seed, *, chains, draws, warmup):
    rng = np.random.default_rng(seed)
    params, pop, true_contrasts = draw_true_params(model, rng, n_subj)
    trials = simulate_batch(model, params, stim, jax.random.PRNGKey(seed))
    # シミュレーションでは a_t を実行時にサンプルするため事前計算できない。
    # 推定に入る前にここで付与する（勾配評価のたびの再計算を避ける）。
    trials = precompute_trial_features(trials)

    mcmc = run_nuts(
        model, trials, n_subj, chains=chains, draws=draws, warmup=warmup, seed=seed
    )
    idata = to_inference_data(mcmc, [f"s{i}" for i in range(n_subj)])
    post = idata["posterior"]

    rec = {"seed": seed}
    for name, true_val in true_contrasts.items():
        x = np.asarray(post[name]).ravel()
        lo, hi = np.percentile(x, [2.5, 97.5])
        rec[f"true_{name}"] = float(true_val)
        rec[f"est_{name}"] = float(x.mean())
        rec[f"lo_{name}"] = float(lo)
        rec[f"hi_{name}"] = float(hi)
        rec[f"cover_{name}"] = bool(lo <= true_val <= hi)
    for ps in model.params:
        x = np.asarray(post[f"pop_{ps.name}"]).ravel()
        rec[f"true_pop_{ps.name}"] = float(pop[ps.name])
        rec[f"est_pop_{ps.name}"] = float(x.mean())
    return rec


def _records_path(out_dir: Path) -> Path:
    return Path(out_dir) / "records.csv"


def _run_config(args, model, n_subj: int) -> dict:
    """レプリケートの互換性を判定するための設定。

    draws などを変えて再実行したときに、条件の異なる古い行を再開対象として
    拾ってしまわないよう、各行に設定を埋め込んで照合する。
    """
    return {
        "model": model.name,
        "agent_reward": args.agent_reward,
        "chains": args.chains,
        "draws": args.draws,
        "warmup": args.warmup,
        "n_subj": n_subj,
    }


def append_record(out_dir: Path, rec: dict) -> None:
    """1 レプリケート終わるごとに追記する。

    数時間かかる実行で、途中で落ちても結果が失われないようにするため。
    """
    import pandas as pd

    path = _records_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec]).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_records(out_dir: Path, config: dict) -> list[dict]:
    """既存の records.csv から、設定が一致する行だけを読む（再開用）。"""
    import pandas as pd

    path = _records_path(out_dir)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for key, value in config.items():
        if key in df.columns:
            df = df[df[key] == value]
    return df.to_dict("records")


def summarize(records, model):
    df_keys = records[0].keys()
    out = {"n_sim": len(records), "parameters": {}}
    for key in df_keys:
        if not key.startswith("true_"):
            continue
        base = key[len("true_"):]
        est_key = f"est_{base}"
        if est_key not in df_keys:
            continue
        t = np.array([r[key] for r in records])
        e = np.array([r[est_key] for r in records])
        entry = {"corr": float(np.corrcoef(t, e)[0, 1]) if len(t) > 1 else float("nan")}
        cov_key = f"cover_{base}"
        if cov_key in df_keys:
            entry["coverage_95"] = float(np.mean([r[cov_key] for r in records]))
        out["parameters"][base] = entry

    # AC-3: 群レベル contrast の回復
    primary = [k for k in out["parameters"] if k.startswith("contrast_")]
    if len(records) < MIN_SIM_FOR_VERDICT:
        out["ac3_pass"] = None
        out["note"] = (
            f"n_sim={len(records)} は判定に不足（{MIN_SIM_FOR_VERDICT} 以上必要）。"
            "相関が乱数と区別できないため合否を出さない。"
        )
    else:
        out["ac3_pass"] = bool(
            primary
            and all(
                out["parameters"][k]["corr"] > 0.7
                and out["parameters"][k].get("coverage_95", 0.0) > 0.9
                for k in primary
            )
        )
    return out


def plot_scatter(records, out_path: Path):
    """相関係数だけで判断しないため散布図を出す（FR-5.8）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k[len("true_"):] for k in records[0] if k.startswith("true_") and f"est_{k[5:]}" in records[0]]
    n = len(keys)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        t = [r[f"true_{key}"] for r in records]
        e = [r[f"est_{key}"] for r in records]
        ax.scatter(t, e, s=18, alpha=0.8)
        lims = [min(min(t), min(e)), max(max(t), max(e))]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("true")
        ax.set_ylabel("recovered")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="パラメータリカバリ（FR-5.8, FR-5.9）")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--input", type=Path, help="刺激と zone を借りる tidy テーブル")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--agent-reward", default="ordinal", choices=("ordinal", "smooth"))
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--subjects", type=int, default=None, help="先頭 N 名だけ使う（動作確認用）")
    ap.add_argument("--fresh", action="store_true",
                    help="既存の records.csv を捨てて最初からやり直す（既定は再開）")
    args = ap.parse_args(argv)

    if args.input:
        tidy = pdata.load_tidy(args.input)
    elif args.data_dir:
        tidy = pdata.load_from_raw(args.data_dir)
    else:
        ap.error("--input か --data-dir のどちらかが必要")

    if args.subjects:
        keep = sorted(tidy["subject_id"].unique())[: args.subjects]
        tidy = tidy[tidy["subject_id"].isin(keep)].reset_index(drop=True)

    s = pdata.stimuli_from_tidy(tidy)
    n_subj = len(s["subjects"])
    stim = stim_arrays(
        s["theta_white_deg"], s["theta_black_deg"], s["target_deg"], s["zone"], s["valid"]
    )
    model = build_model(args.model, agent_reward_variant=args.agent_reward)
    out_dir = args.out or Path("outputs/pomdp/recovery") / model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _run_config(args, model, n_subj)
    if args.fresh and _records_path(out_dir).exists():
        _records_path(out_dir).unlink()
        print("[fresh] 既存の records.csv を削除した")

    records = load_records(out_dir, config)
    done_seeds = {int(r["seed"]) for r in records}
    if done_seeds:
        print(f"[resume] 設定が一致する完了済み {len(done_seeds)} レプリケートをスキップする")

    for i in range(args.n_sim):
        seed = args.seed + i
        if seed in done_seeds:
            continue
        t0 = time.perf_counter()
        print(f"[recovery] replicate {i + 1}/{args.n_sim} (seed={seed})", flush=True)
        rec = one_replicate(
            model, stim, n_subj, seed,
            chains=args.chains, draws=args.draws, warmup=args.warmup,
        )
        rec.update(config)
        append_record(out_dir, rec)  # 落ちても失わないよう毎回追記する
        records.append(rec)
        elapsed = time.perf_counter() - t0
        remaining = args.n_sim - (i + 1)
        print(
            f"[recovery] seed={seed} 完了 {elapsed / 60:.1f} 分"
            f"（残り {remaining} 本、推定 {elapsed * remaining / 3600:.1f} 時間）",
            flush=True,
        )
        # 途中経過の要約も毎回更新しておく
        (out_dir / "summary.json").write_text(
            json.dumps(summarize(records, model), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not records:
        print("[warn] レプリケートが 1 本も無い")
        return

    summary = summarize(records, model)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_scatter(records, out_dir / "scatter.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["ac3_pass"] is None:
        print(f"[warn] {summary['note']}")
    elif not summary["ac3_pass"]:
        print("[warn] AC-3 未達（contrast の相関 > 0.7 かつカバレッジ > 0.9 が必要）")


if __name__ == "__main__":
    main()
