"""一般化線形混合モデル。選択行動（二値）の解析用。

現状は statsmodels の BinomialBayesMixedGLM を薄く包んだ実装のみ。
lmm.fit_lmm と同じ MixedResult を返すので、report / viz はそのまま使える。

statsmodels を使う際の制約（設計判断が要るので明記しておく）
------------------------------------------------------------
1. `BinomialBayesMixedGLM` は分散成分を**独立**と仮定し、ランダム切片と
   ランダム傾きの**相関 tau01 を推定しない**。「開始水準が低い人ほど
   立ち上がりが急か」を問うなら、この相関そのものが争点になるので
   statsmodels では答えられない。Bambi / PyMC か lme4 が要る。
2. `fit_vb()` は変分ベイズで、事後 SD を**過小評価**する。z 値と p 値を
   額面通りに受け取らないこと。`fit_map()` の方が保守的だが CI が出ない。
3. 尤度がないので LRT も AIC も使えない。モデル比較は WAIC / LOO が要り、
   それには完全ベイズ実装が必要。

したがって、確定的な数値を出す段階では PyMC / Bambi への移行を推奨する。
ここでの実装は探索用と位置づける。
"""
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from dualrdk.mixed.results import MixedResult
from dualrdk.mixed.specs import ModelSpec


def _vc_formulas(spec: ModelSpec) -> dict:
    """ModelSpec のランダム効果部分を BinomialBayesMixedGLM の vc_formulas に変換。

    (1 | subject)            -> {"a": "0 + C(subject)"}
    (1 + trial_s | subject)  -> {"a": "0 + C(subject)", "b": "0 + C(subject):trial_s"}
    """
    g = spec.groups
    re = (spec.re_formula or "~1").lstrip("~").strip()
    terms = [t.strip() for t in re.split("+") if t.strip()]
    has_intercept = "0" not in terms
    slopes = [t for t in terms if t not in ("0", "1")]

    vc = {}
    if has_intercept:
        vc["re_intercept"] = f"0 + C({g})"
    for s in slopes:
        vc[f"re_{s}"] = f"0 + C({g}):{s}"
    return vc


def fit_glmm(
    spec: ModelSpec,
    df: pd.DataFrame,
    method: str = "vb",
) -> MixedResult:
    """二項 GLMM を推定する。

    Parameters
    ----------
    method : {"vb", "map"}
        "vb" は変分ベイズ（事後 SD は過小評価）。"map" は最大事後確率点推定。
    """
    if not spec.is_glmm:
        raise ValueError(f"{spec.name} は LMM 仕様です。dualrdk.mixed.lmm を使ってください。")
    if spec.family != "binomial":
        raise NotImplementedError(f"family='{spec.family}' は未実装です（binomial のみ対応）")

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    data = df.dropna(subset=[spec.formula.split("~")[0].strip()]).copy()
    vc = _vc_formulas(spec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = BinomialBayesMixedGLM.from_formula(spec.formula, vc, data)
        res = md.fit_vb() if method == "vb" else md.fit_map()

    n_fe = len(md.exog_names)
    fe_names = list(md.exog_names)
    est = np.asarray(res.params)[:n_fe]
    se = np.asarray(res.cov_params())[:n_fe] if np.ndim(res.cov_params()) == 1 else np.sqrt(
        np.diag(np.atleast_2d(res.cov_params()))[:n_fe]
    )
    z = est / se
    from scipy import stats as _st

    fixed = pd.DataFrame(
        {
            "estimate": est,
            "se": se,
            "z": z,
            "p": 2 * _st.norm.sf(np.abs(z)),
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
        },
        index=fe_names,
    )

    # vcp_mean は log(SD) スケールで返る
    vc_names = list(vc.keys())
    log_sd = np.asarray(res.vcp_mean) if hasattr(res, "vcp_mean") else np.array([])
    sds = np.exp(log_sd) if len(log_sd) else np.array([])
    varcomp = pd.DataFrame(
        {"variance": sds ** 2, "sd": sds}, index=vc_names[: len(sds)]
    )

    cov_re = pd.DataFrame(
        np.diag(sds ** 2) if len(sds) else np.zeros((0, 0)),
        index=varcomp.index,
        columns=varcomp.index,
    )

    return MixedResult(
        spec=spec,
        fixed=fixed,
        varcomp=varcomp,
        corr={},  # statsmodels は相関を推定しない（冒頭の注記 1 を参照）
        cov_re=cov_re,
        scale=None,  # 二項分布に残差分散パラメータはない
        llf=np.nan,  # 変分下界であって尤度ではないので比較に使わない
        n_obs=len(data),
        n_groups=int(data[spec.groups].nunique()),
        converged=True,
        reml=False,
        optimizer=method,
        random_effects=pd.DataFrame(),
        raw=res,
    )
