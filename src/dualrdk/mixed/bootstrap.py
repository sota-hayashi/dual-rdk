"""参加者単位のノンパラメトリック・ブートストラップ。

分散パラメータの Wald 型 SE は境界近傍で信用できないので、CI は
リサンプリングで出す。参加者をまとめて（試行ではなく）リサンプルするので、
クラスタ構造が保たれる。
"""
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dualrdk.mixed.config import BOOTSTRAP_SEED, DEFAULT_OPTIMIZER, DEFAULT_REML, N_BOOTSTRAP
from dualrdk.mixed.results import MixedResult
from dualrdk.mixed.specs import ModelSpec


def bootstrap_lmm(
    spec: ModelSpec,
    df: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    reml: bool = DEFAULT_REML,
    optimizer: str = DEFAULT_OPTIMIZER,
    progress: bool = False,
) -> pd.DataFrame:
    """参加者を復元抽出して n_boot 回フィットし、推定値の分布を返す。"""
    rng = np.random.default_rng(seed)
    subjects = df[spec.groups].unique()
    draws = []

    for b in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        # 同じ参加者が複数回選ばれても別クラスタとして扱うため ID を振り直す
        boot = pd.concat(
            [
                df[df[spec.groups] == s].assign(**{spec.groups: f"{s}__{j}"})
                for j, s in enumerate(pick)
            ],
            ignore_index=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = smf.mixedlm(
                    spec.formula, boot, groups=boot[spec.groups], re_formula=spec.re_formula
                ).fit(reml=reml, method=optimizer)
            row = {f"beta[{n}]": float(r.fe_params[n]) for n in r.fe_params.index}
            C = r.cov_re
            labels = ["Intercept" if n == "Group" else n for n in C.index]
            for i, n in enumerate(labels):
                row[f"sd[{n}]"] = float(np.sqrt(max(C.iloc[i, i], 0.0)))
            for i, a in enumerate(labels):
                for j, bb in enumerate(labels):
                    if j > i:
                        d = np.sqrt(max(C.iloc[i, i] * C.iloc[j, j], 0.0))
                        row[f"corr[{a}~{bb}]"] = float(C.iloc[i, j] / d) if d > 0 else np.nan
            row["sd[Residual]"] = float(np.sqrt(r.scale))
            draws.append(row)
        except Exception:
            continue
        if progress and (b + 1) % 50 == 0:
            print(f"    bootstrap {b + 1}/{n_boot}", flush=True)

    return pd.DataFrame(draws)


def bootstrap_ci(
    draws: pd.DataFrame, point: Optional[dict] = None, alpha: float = 0.05
) -> pd.DataFrame:
    """パーセンタイル CI の表を作る。"""
    rows = []
    for c in draws.columns:
        v = draws[c].dropna()
        if v.empty:
            continue
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append(
            {
                "parameter": c,
                "point": (point or {}).get(c, np.nan),
                "boot_median": float(v.median()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_draws": int(len(v)),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)


def point_dict(res: MixedResult) -> dict:
    """MixedResult から、ブートストラップ表と同じキーの点推定を作る。"""
    d = {f"beta[{n}]": float(res.fixed.loc[n, "estimate"]) for n in res.fixed.index}
    for n in res.varcomp.index:
        d[f"sd[{n}]"] = float(res.varcomp.loc[n, "sd"])
    for k, v in res.corr.items():
        d[f"corr[{k}]"] = float(v)
    return d
