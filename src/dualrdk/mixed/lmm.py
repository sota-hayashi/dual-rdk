"""線形混合モデル（statsmodels MixedLM）のフィッティング。

statsmodels の罠を 2 つ、ここで吸収する。

1. `MixedLMResults.cov_re` は **すでにデータのスケール**で返る。
   `params` 側は scale で割った値なので、両者を取り違えて cov_re に
   もう一度 scale を掛けると分散を 1/scale 倍に潰してしまう。
2. 最適化手法によって収束先が変わる。bfgs / cg は本データで logLik を
   230 も取り違えるので、既定を lbfgs にし、毎回 check_optimizers で
   他手法との一致を確認する。
"""
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dualrdk.mixed.config import (
    CHECK_OPTIMIZERS,
    DEFAULT_OPTIMIZER,
    DEFAULT_REML,
)
from dualrdk.mixed.results import MixedResult
from dualrdk.mixed.specs import ModelSpec


def fit_lmm(
    spec: ModelSpec,
    df: pd.DataFrame,
    reml: bool = DEFAULT_REML,
    optimizer: str = DEFAULT_OPTIMIZER,
    check_optimizers: bool = True,
) -> MixedResult:
    """ModelSpec を statsmodels MixedLM で推定し、MixedResult に正規化する。"""
    if spec.is_glmm:
        raise ValueError(f"{spec.name} は GLMM 仕様です。dualrdk.mixed.glmm を使ってください。")

    md = smf.mixedlm(
        spec.formula, df, groups=df[spec.groups], re_formula=spec.re_formula
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = md.fit(reml=reml, method=optimizer)

    # --- 固定効果 ---------------------------------------------------------
    fe_names = list(res.fe_params.index)
    ci = res.conf_int()
    fixed = pd.DataFrame(
        {
            "estimate": [res.params[n] for n in fe_names],
            "se": [res.bse[n] for n in fe_names],
            "z": [res.tvalues[n] for n in fe_names],
            "p": [res.pvalues[n] for n in fe_names],
            "ci_low": [ci.loc[n, 0] for n in fe_names],
            "ci_high": [ci.loc[n, 1] for n in fe_names],
        },
        index=fe_names,
    )

    # --- ランダム効果 -----------------------------------------------------
    # cov_re はデータのスケール。scale を掛けないこと（冒頭の注記参照）。
    C = res.cov_re.copy()
    labels = ["Intercept" if n == "Group" else n for n in C.index]
    C.index = labels
    C.columns = labels

    rows = {n: {"variance": C.loc[n, n], "sd": np.sqrt(max(C.loc[n, n], 0.0))} for n in labels}
    rows["Residual"] = {"variance": res.scale, "sd": np.sqrt(res.scale)}
    varcomp = pd.DataFrame(rows).T[["variance", "sd"]]

    corr = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            denom = np.sqrt(C.loc[a, a] * C.loc[b, b])
            corr[f"{a}~{b}"] = float(C.loc[a, b] / denom) if denom > 0 else np.nan

    # --- BLUP -------------------------------------------------------------
    re_dict = res.random_effects
    blup = pd.DataFrame(
        [{"subject": k, **{("Intercept" if kk == "Group" else kk): vv for kk, vv in v.items()}}
         for k, v in re_dict.items()]
    ).set_index("subject")
    for n in blup.columns:
        if n in fixed.index:
            blup[f"{n}_total"] = fixed.loc[n, "estimate"] + blup[n]

    out = MixedResult(
        spec=spec,
        fixed=fixed,
        varcomp=varcomp,
        corr=corr,
        cov_re=C,
        scale=float(res.scale),
        llf=float(res.llf),
        n_obs=int(res.nobs),
        n_groups=int(md.n_groups),
        converged=bool(res.converged),
        reml=reml,
        optimizer=optimizer,
        random_effects=blup,
        raw=res,
    )
    if check_optimizers:
        out.convergence_check = check_optimizer_agreement(spec, df, reml=reml)
    return out


def check_optimizer_agreement(
    spec: ModelSpec,
    df: pd.DataFrame,
    reml: bool = DEFAULT_REML,
    methods: tuple = CHECK_OPTIMIZERS,
) -> pd.DataFrame:
    """複数の最適化手法で推定し、同じ最尤解に到達しているか確認する。

    logLik が最良値から 0.1 以上離れた手法は別の解に落ちたとみなす。
    0.1 未満は最適化の収束許容誤差の範囲。
    """
    tol = 0.1
    rows = []
    for m in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = smf.mixedlm(
                    spec.formula, df, groups=df[spec.groups], re_formula=spec.re_formula
                ).fit(reml=reml, method=m)
            sds = {
                f"sd[{'Intercept' if n == 'Group' else n}]": np.sqrt(max(r.cov_re.iloc[i, i], 0))
                for i, n in enumerate(r.cov_re.index)
            }
            rows.append(
                {
                    "method": m,
                    "converged": bool(r.converged),
                    "logLik": float(r.llf),
                    **sds,
                    **{f"beta[{n}]": float(r.fe_params[n]) for n in r.fe_params.index},
                }
            )
        except Exception as e:  # pragma: no cover - 手法によっては例外で落ちる
            rows.append({"method": m, "converged": False, "logLik": np.nan, "error": type(e).__name__})

    out = pd.DataFrame(rows)
    if out["logLik"].notna().any():
        best = out["logLik"].max()
        out["dLogLik"] = out["logLik"] - best
        out["agrees"] = out["dLogLik"].abs() < tol
    return out


def fit_all(
    specs: List[ModelSpec],
    df: pd.DataFrame,
    reml: bool = DEFAULT_REML,
    optimizer: str = DEFAULT_OPTIMIZER,
) -> List[MixedResult]:
    return [fit_lmm(s, df, reml=reml, optimizer=optimizer) for s in specs]


def median_regression_check(
    df: pd.DataFrame, formula: str, term: str
) -> dict:
    """試行を削らない中央値回帰で、固定効果の傾きを独立に検証する。

    トリムが傾き推定を膨らませていないかを見るための対照。クラスタ構造を
    無視するので SE と p は比較に使えない。点推定だけを見ること。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = smf.quantreg(formula, df).fit(q=0.5)
    return {"term": term, "estimate": float(r.params[term]), "note": "クラスタ非考慮。点推定のみ参照"}
