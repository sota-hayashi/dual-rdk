"""LMM / GLMM に共通の結果コンテナ。

statsmodels の MixedLM と BinomialBayesMixedGLM は API が揃っていないので、
フィッティング側でこの形に正規化しておく。レポートと可視化はこの型だけを
知っていればよい。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dualrdk.mixed.specs import ModelSpec


@dataclass
class MixedResult:
    """1 モデルの推定結果。

    Attributes
    ----------
    fixed : DataFrame
        index=項名, 列 = [estimate, se, z, p, ci_low, ci_high]
    varcomp : DataFrame
        index=成分名, 列 = [variance, sd]。共分散・相関は corr に入れる。
    corr : dict
        {(項A, 項B): 相関} 形式。ランダム切片のみなら空。
    scale : float or None
        残差分散 sigma^2。GLMM では None。
    """

    spec: ModelSpec
    fixed: pd.DataFrame
    varcomp: pd.DataFrame
    corr: Dict[str, float]
    cov_re: pd.DataFrame
    scale: Optional[float]
    llf: float
    n_obs: int
    n_groups: int
    converged: bool
    reml: bool
    optimizer: str
    random_effects: pd.DataFrame
    raw: Any = field(repr=False, default=None)
    convergence_check: Optional[pd.DataFrame] = None
    bootstrap: Optional[pd.DataFrame] = None

    # -- 情報量規準 ---------------------------------------------------------
    @property
    def n_params(self) -> int:
        """情報量規準に数えるパラメータ数。

        REML の尤度は固定効果を積分消去した後のものなので、共分散
        パラメータのみを数える。ML なら固定効果も数える。
        statsmodels は REML 時に aic/bic を nan で返すため自前で計算する。
        """
        n_cov = self.spec.n_re_params + (0 if self.scale is None else 1)
        return n_cov if self.reml else n_cov + len(self.fixed)

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.llf + np.log(self.n_obs) * self.n_params

    # -- 分散の分解 ---------------------------------------------------------
    def icc(self) -> Optional[float]:
        """級内相関。ランダム切片分散 / (それ + 残差分散)。

        ランダム傾きがある場合、これは trial_c = 0（＝1 試行目）での値。
        """
        if self.scale is None or self.cov_re.empty:
            return None
        return float(self.cov_re.iloc[0, 0] / (self.cov_re.iloc[0, 0] + self.scale))

    def r2(self, df: pd.DataFrame) -> Dict[str, float]:
        """Nakagawa & Schielzeth の周辺 / 条件付き R^2。

        ランダム傾きがある場合、ランダム効果の分散は説明変数の分布上で
        平均する: tau0^2 + 2*tau01*E[x] + tau1^2*E[x^2]
        """
        if self.scale is None:
            return {}
        terms = [t for t in self.fixed.index if t != "Intercept"]
        pred = np.full(len(df), self.fixed.loc["Intercept", "estimate"])
        for t in terms:
            if t in df.columns:
                pred = pred + self.fixed.loc[t, "estimate"] * df[t].values
        var_f = float(np.var(pred, ddof=0))

        C = self.cov_re.values
        if C.shape[0] == 1:
            var_r = float(C[0, 0])
        else:
            x = df[self.cov_re.index[1]].values if self.cov_re.index[1] in df else None
            if x is None:
                var_r = float(C[0, 0])
            else:
                var_r = float(
                    C[0, 0] + 2 * C[0, 1] * x.mean() + C[1, 1] * (x ** 2).mean()
                )
        tot = var_f + var_r + self.scale
        return {
            "var_fixed": var_f,
            "var_random": var_r,
            "var_residual": float(self.scale),
            "r2_marginal": var_f / tot,
            "r2_conditional": (var_f + var_r) / tot,
        }

    # -- 表示 ---------------------------------------------------------------
    def summary_rows(self) -> pd.DataFrame:
        """1 行 1 モデルの要約（モデル比較表に積むため）。"""
        row = {
            "model": self.spec.name,
            "lme4": self.spec.lme4,
            "n_obs": self.n_obs,
            "n_groups": self.n_groups,
            "logLik": self.llf,
            "k": self.n_params,
            "AIC": self.aic,
            "BIC": self.bic,
            "converged": self.converged,
        }
        for t in self.fixed.index:
            row[f"beta[{t}]"] = self.fixed.loc[t, "estimate"]
            row[f"p[{t}]"] = self.fixed.loc[t, "p"]
        for c in self.varcomp.index:
            row[f"sd[{c}]"] = self.varcomp.loc[c, "sd"]
        return pd.DataFrame([row])
