"""階層ベイズ推定（§5.4）。NumPyro + NUTS。

48 試行・最大 6 パラメータでは個人ごとの独立推定は成立しないため、推定は
階層ベイズで行う（FR-5.3）。無制約空間で非中心化パラメータ化を用いる。

    theta~_{i,k} = mu_k + sigma_k eta_{i,k},  eta ~ N(0,1)
    mu_k ~ N(m_k, s_k^2),  sigma_k ~ HalfNormal(1)

`alpha_out - alpha_in` は個人ごとに推定しない。群レベルの平均差を部分
プーリングで推定し、その事後分布を報告する（FR-5.4）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from dualrdk.pomdp import data as pdata
from dualrdk.pomdp.likelihood import batched_latents, batched_log_lik, subject_log_lik
from dualrdk.pomdp.models import _TRANSFORMS, MODEL_NAMES, build_model

CONTRAST_BASES = ("alpha", "eps")


def hierarchical_model(model, trials, n_subj):
    """NumPyro のモデル関数。"""
    unc = {}
    mu = {}
    for ps in model.params:
        mu[ps.name] = numpyro.sample(f"mu_{ps.name}", dist.Normal(ps.prior_loc, ps.prior_scale))
        sigma = numpyro.sample(f"sigma_{ps.name}", dist.HalfNormal(1.0))
        with numpyro.plate(f"subject_{ps.name}", n_subj):
            eta = numpyro.sample(f"eta_{ps.name}", dist.Normal(0.0, 1.0))
        unc[ps.name] = mu[ps.name] + sigma * eta

    params = model.constrain(unc)
    params = {k: jnp.broadcast_to(jnp.asarray(v), (n_subj,)) for k, v in params.items()}

    ll = batched_log_lik(model, params, trials)
    numpyro.deterministic("log_lik", ll)
    numpyro.factor("obs", jnp.sum(ll))

    # 群レベルの制約付きスケールの量（解釈可能な単位で報告する）
    tf = {p.name: _TRANSFORMS[p.transform] for p in model.params}
    pop = {}
    for ps in model.params:
        pop[ps.name] = numpyro.deterministic(f"pop_{ps.name}", tf[ps.name](mu[ps.name]))
    for base in CONTRAST_BASES:
        if f"{base}_in" in pop and f"{base}_out" in pop:
            numpyro.deterministic(f"contrast_{base}", pop[f"{base}_out"] - pop[f"{base}_in"])


def run_nuts(model, trials, n_subj, *, chains=4, draws=2000, warmup=1000, seed=0,
             target_accept=0.9):
    # chain の並列実行にはデバイス数 >= chain 数が必要（dualrdk.pomdp.__init__ で
    # ホストデバイス数を設定している）。足りない場合は逐次に落とす。
    chain_method = "parallel" if jax.device_count() >= chains else "sequential"
    if chain_method == "sequential" and chains > 1:
        print(
            f"[warn] devices={jax.device_count()} < chains={chains} なので逐次実行にする。"
            "環境変数 DUALRDK_HOST_DEVICES を増やすと並列になる。"
        )
    kernel = NUTS(hierarchical_model, target_accept_prob=target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), model, trials, n_subj, extra_fields=("diverging",))
    return mcmc


# --------------------------------------------------------------------------
# MAP 推定（参加者ごとに独立）
# --------------------------------------------------------------------------
# 階層ベイズと同じモデル・同じ事前分布のまま、推定法だけを差し替えるための経路。
# 既存の 2 値実装（models/q_learning_ooz_map.py）が参加者ごとの MAP だったので、
# それと揃えて参加者ごとに独立最適化する（部分プーリングをしない）。
#
# 最適化は無制約空間で行う。階層モデルが事前分布を無制約スケールで置いている
# ため、同じ事前を同じスケールで使うにはこちらに合わせる必要がある（制約空間で
# 最頻値を取るとヤコビアンの分だけ別の量になる）。
def read_subject_list(path: Path) -> list[str]:
    """subject_id の一覧を読む。1行1名のテキストでも subject_id 列の CSV でもよい。"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "subject_id" not in df.columns:
            raise ValueError(f"{path}: subject_id 列が無い")
        return df["subject_id"].astype(str).tolist()
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _slice_fixed(fixed: dict, i: int) -> dict:
    """固定値が参加者ごとの配列なら 1 名分を取り出す（スカラーはそのまま）。"""
    return {k: (v[i] if jnp.ndim(v) > 0 else v) for k, v in fixed.items()}


def _map_objective(model):
    """-log p(y_i | theta_i) - log p(theta_i)。1 回だけ jit する。

    trials_i と固定値は引数で渡す（クロージャに閉じ込めると参加者ごとに
    再コンパイルされてしまう）。
    """
    names = model.param_names
    locs = jnp.asarray([p.prior_loc for p in model.params])
    scales = jnp.asarray([p.prior_scale for p in model.params])
    free = tuple(p.name for p in model.params)

    def obj(vec, trials_i, fixed_i):
        unc = {n: vec[k] for k, n in enumerate(names)}
        params = {n: _TRANSFORMS[p.transform](unc[n]) for n, p in zip(free, model.params)}
        params.update(fixed_i)
        ll = subject_log_lik(model, params, trials_i)
        log_prior = jnp.sum(dist.Normal(locs, scales).log_prob(vec))
        return -(ll + log_prior)

    return jax.jit(jax.value_and_grad(obj)), obj


def run_map(model, trials, n_subj, *, n_starts=8, seed=0, verbose=True):
    """参加者ごとに独立に MAP 推定する（部分プーリングなし）。

    多点スタートで局所解を避ける。1 点目は事前分布の中心、以降は事前分布から
    引いた乱数を初期値にする。

    Returns
    -------
    params : 制約付きスケール。各値が (n_subj,)
    info   : DataFrame（subject_index, 各パラメータ, log_lik, log_posterior, 収束）
    """
    from scipy.optimize import minimize

    names = model.param_names
    rng = np.random.default_rng(seed)
    locs = np.array([p.prior_loc for p in model.params])
    scales = np.array([p.prior_scale for p in model.params])
    value_and_grad, _ = _map_objective(model)

    best_vecs, rows = [], []
    for i in range(n_subj):
        trials_i = jax.tree_util.tree_map(lambda x: x[i], trials)
        fixed_i = _slice_fixed(model.fixed, i)

        def fun(v):
            val, grad = value_and_grad(jnp.asarray(v), trials_i, fixed_i)
            return float(val), np.asarray(grad, dtype=float)

        best = None
        for s in range(n_starts):
            x0 = locs if s == 0 else locs + scales * rng.normal(size=len(locs))
            res = minimize(fun, x0=np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

        vec = jnp.asarray(best.x)
        params_i = {
            n: _TRANSFORMS[p.transform](vec[k]) for k, (n, p) in enumerate(zip(names, model.params))
        }
        params_i.update(fixed_i)
        best_vecs.append(best.x)
        rows.append(
            {
                "subject_index": i,
                **{n: float(np.asarray(params_i[n])) for n in params_i},
                "log_posterior": float(-best.fun),
                "log_lik": float(subject_log_lik(model, params_i, trials_i)),
                "converged": bool(best.success),
            }
        )
        if verbose and (i + 1) % 10 == 0:
            print(f"[map] {i + 1}/{n_subj} 名", flush=True)

    vecs = jnp.asarray(np.array(best_vecs))  # (n_subj, n_param)
    params = model.constrain({n: vecs[:, k] for k, n in enumerate(names)})
    params = {k: jnp.broadcast_to(jnp.asarray(v), (n_subj,)) for k, v in params.items()}
    return params, pd.DataFrame(rows)


def to_inference_data(mcmc, subjects):
    """arviz の DataTree を組む。log_likelihood の観測単位は参加者（FR-5.7）。"""
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    log_lik = post.pop("log_lik")
    extra = mcmc.get_extra_fields(group_by_chain=True)
    groups = {
        "posterior": post,
        "log_likelihood": {"subject": log_lik},
        "sample_stats": {"diverging": np.asarray(extra["diverging"])},
    }
    # 次元名は変数名（"subject"）と別にする。同名にすると arviz が chain/draw を
    # 観測次元として畳み込んでしまい、n_data_points が chains*draws*n_subj になる。
    return az.from_dict(
        groups,
        coords={"subject_id": list(subjects)},
        dims={"subject": ["subject_id"]},
    )


def diagnostics(idata) -> dict:
    """収束診断（FR-5.5）。基準を満たさない場合は呼び出し側でエラーにする。"""
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(max(float(np.nanmax(np.asarray(v))) for v in rhat.data_vars.values()))
    min_ess = float(min(float(np.nanmin(np.asarray(v))) for v in ess.data_vars.values()))
    div = np.asarray(idata["sample_stats"]["diverging"])
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "n_divergent": int(div.sum()),
        "divergent_frac": float(div.mean()),
        "passes": bool(max_rhat < 1.01 and min_ess > 400 and float(div.mean()) < 0.001),
    }


def posterior_mean_params(model, mcmc, n_subj):
    """参加者ごとの制約付きパラメータの事後平均（潜在変数の軌跡計算用）。"""
    s = mcmc.get_samples()
    unc = {}
    for ps in model.params:
        mu = np.asarray(s[f"mu_{ps.name}"])[:, None]
        sig = np.asarray(s[f"sigma_{ps.name}"])[:, None]
        eta = np.asarray(s[f"eta_{ps.name}"])
        unc[ps.name] = jnp.asarray(np.mean(mu + sig * eta, axis=0))
    params = model.constrain(unc)
    return {k: jnp.broadcast_to(jnp.asarray(v), (n_subj,)) for k, v in params.items()}


def contrast_summary(idata, base: str) -> dict | None:
    name = f"contrast_{base}"
    if name not in idata["posterior"].data_vars:
        return None
    x = np.asarray(idata["posterior"][name]).ravel()
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {
        "parameter": name,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "hdi_2.5%": float(lo),
        "hdi_97.5%": float(hi),
        "P(delta>0)": float(np.mean(x > 0)),
    }


def save_latent(out_dir: Path, model, params, subjects, trials) -> pd.DataFrame:
    """潜在変数の軌跡を latent.csv に書く（FR-6.1）。

    NUTS でも MAP でも同じ形で書くことが重要である。evaluate.py はこのファイル
    だけを見るので、推定法によらず同じ評価コードが使える。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = batched_latents(model, params, trials)
    n_subj, t_max = np.asarray(lat["d"]).shape
    rows = pd.DataFrame(
        {
            "subject_id": np.repeat(subjects, t_max),
            "trial": np.tile(np.arange(1, t_max + 1), n_subj),
            "d": np.asarray(lat["d"]).ravel(),
            "p_w": np.asarray(lat["p_w"]).ravel(),
            "q": np.asarray(lat["q"]).ravel(),
            "log_evidence": np.asarray(lat["log_evidence"]).ravel(),
            "log_pi": np.asarray(lat["log_pi"]).ravel(),
            "zone": np.asarray(trials["zone"]).ravel(),
            "valid": np.asarray(trials["valid"]).ravel(),
        }
    )
    rows.to_csv(out_dir / "latent.csv", index=False)
    return rows


def save_outputs(out_dir: Path, model, mcmc, idata, subjects, trials, contrast_out: Path | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(out_dir / "trace.nc")

    diag = diagnostics(idata)
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # 潜在変数の軌跡（FR-6.1）
    params = posterior_mean_params(model, mcmc, len(subjects))
    lat = batched_latents(model, params, trials)

    n_subj, t_max = np.asarray(lat["d"]).shape
    rows = pd.DataFrame(
        {
            "subject_id": np.repeat(subjects, t_max),
            "trial": np.tile(np.arange(1, t_max + 1), n_subj),
            "d": np.asarray(lat["d"]).ravel(),
            "p_w": np.asarray(lat["p_w"]).ravel(),
            "q": np.asarray(lat["q"]).ravel(),
            "log_evidence": np.asarray(lat["log_evidence"]).ravel(),
            "log_pi": np.asarray(lat["log_pi"]).ravel(),
            "zone": np.asarray(trials["zone"]).ravel(),
            "valid": np.asarray(trials["valid"]).ravel(),
        }
    )
    rows.to_csv(out_dir / "latent.csv", index=False)

    # 群レベル差の事後要約（FR-6.2）。本解析の主結果
    contrasts = [c for c in (contrast_summary(idata, b) for b in CONTRAST_BASES) if c]
    if contrasts:
        payload = {"model": model.name, "contrasts": contrasts}
        text = json.dumps(payload, indent=2)
        (out_dir / "contrast.json").write_text(text, encoding="utf-8")
        if contrast_out is not None:
            contrast_out = Path(contrast_out)
            contrast_out.parent.mkdir(parents=True, exist_ok=True)
            contrast_out.write_text(text, encoding="utf-8")
    return diag, contrasts


def main(argv=None):
    ap = argparse.ArgumentParser(description="dual-RDK POMDP/Q学習モデルの階層ベイズ推定")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--input", type=Path, help="tidy テーブル（.csv / .parquet）")
    ap.add_argument("--data-dir", type=Path, help="生データディレクトリ（--input の代わり）")
    ap.add_argument("--agent-reward", default="ordinal", choices=("ordinal", "smooth"))
    ap.add_argument(
        "--method", default="nuts", choices=("nuts", "map"),
        help="nuts=階層ベイズ（部分プーリング）, map=参加者ごとに独立な MAP",
    )
    ap.add_argument("--n-starts", type=int, default=8, help="MAP の多点スタート数")
    ap.add_argument(
        "--fix-perceptual", type=Path, default=None,
        help="perceptual.py の出力 CSV。kappa / lam を推定対象から外して固定する",
    )
    ap.add_argument(
        "--train-trials", type=int, default=None,
        help="先頭 N 試行だけで推定する（残りはホールドアウト評価用）",
    )
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strict-validate", action="store_true")
    ap.add_argument(
        "--contrast-out",
        type=Path,
        default=Path("outputs/pomdp/alpha_contrast.json"),
        help="群レベル差の事後要約の書き出し先（FR-6.2）",
    )
    ap.add_argument("--subjects", type=int, default=None, help="先頭 N 名だけ使う（動作確認用）")
    ap.add_argument(
        "--subjects-file", type=Path, default=None,
        help="subject_id を列挙したファイル（1行1名、または subject_id 列を持つ CSV）。"
             "部分集団解析で使う",
    )
    args = ap.parse_args(argv)

    if args.input:
        tidy = pdata.load_tidy(args.input)
    elif args.data_dir:
        tidy = pdata.load_from_raw(args.data_dir)
    else:
        ap.error("--input か --data-dir のどちらかが必要")

    if args.subjects_file:
        keep = read_subject_list(args.subjects_file)
        missing = sorted(set(keep) - set(tidy["subject_id"]))
        if missing:
            raise SystemExit(f"{args.subjects_file}: tidy に無い参加者 {missing[:5]}")
        tidy = tidy[tidy["subject_id"].isin(keep)].reset_index(drop=True)
        print(f"[fit] 参加者を {len(keep)} 名に限定（{args.subjects_file}）")
    if args.subjects:
        keep = sorted(tidy["subject_id"].unique())[: args.subjects]
        tidy = tidy[tidy["subject_id"].isin(keep)].reset_index(drop=True)

    report = pdata.validate_tidy(tidy, strict=args.strict_validate)
    print("[validate]", json.dumps(report, ensure_ascii=False))

    subjects, trials = pdata.arrays_from_tidy(tidy)

    # 2 段階推定（FR-5.10）。kappa / lam は応答角の残差だけで決まり、学習について
    # 何の情報も持たない（リカバリ相関 0.99）。段階1で先に測って固定すれば、
    # 階層モデルの可変量が 2 x (2 + n_subj) 個減る。
    fixed = None
    if args.fix_perceptual:
        from dualrdk.pomdp.perceptual import load_fixed

        fixed = load_fixed(args.fix_perceptual, subjects)
        print(f"[fit] 固定: {sorted(fixed)}（{args.fix_perceptual}）")

    model = build_model(args.model, agent_reward_variant=args.agent_reward, fixed=fixed)
    print(f"[fit] model={model.name} method={args.method} "
          f"params={model.param_names} n_subj={len(subjects)}")

    # 学習用に前半だけ切り出す場合も、評価は全試行の latent で行う（後半は
    # 推定に使われていないので、そのまま真のホールドアウトになる）。
    trials_fit = trials
    if args.train_trials:
        mask = np.arange(np.asarray(trials["valid"]).shape[1]) < args.train_trials
        trials_fit = dict(trials)
        trials_fit["valid"] = jnp.asarray(np.asarray(trials["valid"]) & mask)
        print(f"[fit] 推定は先頭 {args.train_trials} 試行のみ（残りはホールドアウト）")

    args.out.mkdir(parents=True, exist_ok=True)
    # 推定条件を残す。--train-trials 無しの「後半だけの一致率」は外挿性能では
    # ないので、下流（compare.py）がそれを見分けられるようにしておく。
    (args.out / "fit_config.json").write_text(
        json.dumps(
            {
                "model": model.name,
                "method": args.method,
                "n_params": len(model.param_names),
                "param_names": list(model.param_names),
                "fixed": sorted(model.fixed),
                "train_trials": args.train_trials,
                "n_subjects": len(subjects),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.method == "map":
        params, info = run_map(
            model, trials_fit, len(subjects), n_starts=args.n_starts, seed=args.seed
        )
        info.insert(0, "subject_id", [subjects[i] for i in info["subject_index"]])
        info.to_csv(args.out / "params.csv", index=False)
        save_latent(args.out, model, params, subjects, trials)
        n_bad = int((~info["converged"]).sum())
        print(f"[map] 完了。収束しなかった参加者 {n_bad}/{len(subjects)}")
        print(f"[map] log_lik 合計 {info['log_lik'].sum():.1f}")
        print(f"[saved] {args.out}/params.csv, latent.csv")
        return

    mcmc = run_nuts(
        model, trials_fit, len(subjects),
        chains=args.chains, draws=args.draws, warmup=args.warmup, seed=args.seed,
    )
    idata = to_inference_data(mcmc, subjects)
    diag, contrasts = save_outputs(
        args.out, model, mcmc, idata, subjects, trials, contrast_out=args.contrast_out
    )

    print("[diagnostics]", json.dumps(diag, indent=2))
    for c in contrasts:
        print("[contrast]", json.dumps(c, indent=2))
    if not diag["passes"]:
        raise SystemExit(
            "収束診断が基準を満たさない（FR-5.5: R_hat<1.01, ESS>400, divergences<0.1%）"
        )


if __name__ == "__main__":
    main()
