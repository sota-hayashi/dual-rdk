"""段階1：知覚・運動パラメータ（kappa, lam）の事前推定（FR-5.10）。

kappa（応答角の精度）と lam（ラプス率）は、参加者が**どちらの雲を選んだか**と
**その方向をどれだけ正確に指したか**だけで決まる。学習については何の情報も
持たない。リカバリでの復元相関がそれぞれ 0.987 / 0.990 と、他のどのパラメータ
よりも高いのはそのためである。

したがってこの 2 つは学習モデルと切り離して先に測れる。段階2で固定すれば、
階層モデルから 2 x (mu + sigma + eta x n_subj) 個の可変量が消える。

段階1のモデル（学習を一切含まない）:

    a_t ~ (1 - lam) [ w vM(a; th_W, kappa) + (1 - w) vM(a; th_B, kappa) ]
          + lam / (2 pi)

w は「その参加者が白を選ぶ周辺確率」で、ここでは純粋な nuisance である。試行
ごとの変動（学習）を無視して 1 つの定数に潰しているが、kappa は各成分の幅、
lam は裾の重さから決まるので、w の値にはほとんど影響されない。

段階1の不確実性を段階2に伝えない点が理論的な弱点である。復元相関 0.99 の量に
ついては実害が無いと判断しているが、感度分析として `--fix-perceptual` を外した
推定と結果を比べること。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from dualrdk.pomdp import data as pdata
from dualrdk.pomdp.circular import TWO_PI, log_vonmises_from_cos

# 事前分布（無制約スケール）。学習モデルの ParamSpec と同じものを使う。
PRIORS = {
    "kappa": ("log", float(np.log(15.0)), 0.7),
    "lam": ("logit", -3.2, 1.2),
    "w": ("logit", 0.0, 1.5),  # 白選択の周辺確率。nuisance なので弱い事前
}
PARAM_ORDER = ("kappa", "lam", "w")

FIXABLE = ("kappa", "lam")


def _neg_log_posterior(vec, cos_aw, cos_ab, valid):
    log_kappa, logit_lam, logit_w = vec[0], vec[1], vec[2]
    kappa = jnp.exp(log_kappa)
    lam = jax.nn.sigmoid(logit_lam)

    lw = jax.nn.log_sigmoid(logit_w) + log_vonmises_from_cos(cos_aw, kappa)
    lb = jax.nn.log_sigmoid(-logit_w) + log_vonmises_from_cos(cos_ab, kappa)
    log_mix = jnp.logaddexp(lw, lb)
    log_pi = jnp.logaddexp(
        jnp.log1p(-lam) + log_mix, jnp.log(lam) - jnp.log(TWO_PI)
    )
    ll = jnp.sum(jnp.where(valid, log_pi, 0.0))

    locs = jnp.asarray([PRIORS[k][1] for k in PARAM_ORDER])
    scales = jnp.asarray([PRIORS[k][2] for k in PARAM_ORDER])
    log_prior = jnp.sum(-0.5 * ((vec - locs) / scales) ** 2 - jnp.log(scales))
    return -(ll + log_prior)


_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_neg_log_posterior))


def fit_perceptual(tidy: pd.DataFrame, *, n_starts: int = 4, seed: int = 0) -> pd.DataFrame:
    """参加者ごとに kappa, lam, w を MAP 推定する。"""
    from scipy.optimize import minimize

    subjects, trials = pdata.arrays_from_tidy(tidy)
    cos_aw = np.asarray(trials["cos_aw"])
    cos_ab = np.asarray(trials["cos_ab"])
    valid = np.asarray(trials["valid"])
    rng = np.random.default_rng(seed)
    locs = np.array([PRIORS[k][1] for k in PARAM_ORDER])
    scales = np.array([PRIORS[k][2] for k in PARAM_ORDER])

    rows = []
    for i, subj in enumerate(subjects):
        args = (jnp.asarray(cos_aw[i]), jnp.asarray(cos_ab[i]), jnp.asarray(valid[i]))

        def fun(v):
            val, grad = _VALUE_AND_GRAD(jnp.asarray(v), *args)
            return float(val), np.asarray(grad, dtype=float)

        best = None
        for s in range(n_starts):
            x0 = locs if s == 0 else locs + scales * rng.normal(size=3)
            res = minimize(fun, x0=np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

        kappa = float(np.exp(best.x[0]))
        rows.append(
            {
                "subject_id": subj,
                "kappa": kappa,
                "lam": float(1.0 / (1.0 + np.exp(-best.x[1]))),
                "w": float(1.0 / (1.0 + np.exp(-best.x[2]))),
                "circ_sd_deg": float(np.rad2deg(np.sqrt(1.0 / kappa))),
                "n_valid": int(valid[i].sum()),
                "log_posterior": float(-best.fun),
                "converged": bool(best.success),
            }
        )
    return pd.DataFrame(rows)


def load_fixed(path: Path, subjects: list[str], params=FIXABLE) -> dict:
    """段階1の出力を、build_model(fixed=...) に渡せる形にする。

    参加者順は subjects に合わせる。1 名でも欠けていればエラーにする
    （黙って群平均で埋めると、その参加者だけ別モデルになってしまう）。
    """
    df = pd.read_csv(path).set_index("subject_id")
    missing = [s for s in subjects if s not in df.index]
    if missing:
        raise ValueError(f"{path}: 段階1の結果に無い参加者がいる: {missing[:5]}")
    return {p: jnp.asarray(df.loc[subjects, p].to_numpy(dtype=float)) for p in params}


def main(argv=None):
    ap = argparse.ArgumentParser(description="段階1：kappa / lam の事前推定（FR-5.10）")
    ap.add_argument("--input", type=Path, required=True, help="tidy テーブル")
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/perceptual.csv"))
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tidy = pdata.load_tidy(args.input)
    df = fit_perceptual(tidy, n_starts=args.n_starts, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"[perceptual] {len(df)} 名")
    for col in ("kappa", "circ_sd_deg", "lam"):
        v = df[col]
        print(f"  {col:12s} 中央値 {v.median():7.3f}  範囲 [{v.min():.3f}, {v.max():.3f}]")
    n_bad = int((~df["converged"]).sum())
    if n_bad:
        print(f"[warn] 収束しなかった参加者 {n_bad} 名")
    print(f"[saved] {args.out}")
    return df


if __name__ == "__main__":
    main()
