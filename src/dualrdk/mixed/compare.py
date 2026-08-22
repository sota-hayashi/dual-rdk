"""モデル比較。境界上の検定に対応した尤度比検定を含む。

分散パラメータの検定 (tau^2 = 0) は帰無仮説がパラメータ空間の**境界**に
あるため、素朴な chi2(df) を参照分布に使うと保守的すぎる。
Self & Liang (1987) / Stram & Lee (1994) に従い混合分布を使う。

    ランダム切片のみ vs ランダム切片+傾き（tau1^2 と tau01 の 2 個を追加）
        -> 0.5 * chi2(1) + 0.5 * chi2(2)
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from dualrdk.mixed.results import MixedResult


def lrt(
    reduced: MixedResult,
    full: MixedResult,
    boundary: bool = True,
) -> dict:
    """尤度比検定。

    Parameters
    ----------
    boundary : bool
        True なら境界補正した混合 chi2 を使う（分散成分の検定）。
        固定効果の検定なら False にして素朴な chi2 を使うこと。その場合は
        REML ではなく ML で推定したモデルを渡す必要がある。
    """
    if reduced.reml != full.reml:
        raise ValueError("REML/ML が揃っていないモデルは比較できません")
    if reduced.reml and reduced.fixed.index.tolist() != full.fixed.index.tolist():
        raise ValueError(
            "REML 尤度は固定効果が異なるモデル間で比較できません。ML で推定し直してください"
        )

    chi2 = 2 * (full.llf - reduced.llf)
    df = full.n_params - reduced.n_params
    p_naive = float(stats.chi2.sf(chi2, df))
    if boundary and df >= 1:
        # 追加パラメータのうち 1 つが境界（分散 >= 0）にある場合の混合分布
        p = 0.5 * stats.chi2.sf(chi2, df - 1) + 0.5 * stats.chi2.sf(chi2, df)
    else:
        p = p_naive

    return {
        "reduced": reduced.spec.name,
        "full": full.spec.name,
        "chi2": float(chi2),
        "df": int(df),
        "p_boundary": float(p),
        "p_naive": p_naive,
        "dAIC": float(full.aic - reduced.aic),
        "dBIC": float(full.bic - reduced.bic),
        "reference": f"0.5*chi2({df-1}) + 0.5*chi2({df})" if boundary else f"chi2({df})",
    }


def comparison_table(results: List[MixedResult]) -> pd.DataFrame:
    """全モデルを 1 表にまとめ、AIC 差と Akaike 重みを付ける。"""
    tab = pd.concat([r.summary_rows() for r in results], ignore_index=True)
    tab["dAIC"] = tab["AIC"] - tab["AIC"].min()
    w = np.exp(-0.5 * tab["dAIC"])
    tab["akaike_weight"] = w / w.sum()
    tab["selected"] = tab["dAIC"] == 0
    return tab


def selection_note(cmp: dict) -> str:
    """比較結果を日本語 1 行で言い切る。"""
    better = cmp["full"] if cmp["dAIC"] < 0 else cmp["reduced"]
    sig = "有意" if cmp["p_boundary"] < 0.05 else "有意でない"
    return (
        f"{cmp['full']} vs {cmp['reduced']}: chi2({cmp['df']}) = {cmp['chi2']:.3f}, "
        f"p = {cmp['p_boundary']:.4f}（{cmp['reference']}）で{sig}、"
        f"dAIC = {cmp['dAIC']:+.2f} → {better} を採択"
    )
