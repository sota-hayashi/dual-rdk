"""混合モデル（LMM / GLMM）解析。

構成
----
    config.py       除外規則・推定設定。値を変えたら run_config.json に残る
    data.py         試行水準データフレームの構築（LMM/GLMM 共通）
    specs.py        ModelSpec。lme4 記法との対応もここ
    results.py      MixedResult。LMM/GLMM 共通の結果コンテナ
    lmm.py          statsmodels MixedLM でのフィッティング
    glmm.py         二項 GLMM（探索用。制約は glmm.py の docstring 参照）
    compare.py      境界補正付き尤度比検定、AIC/BIC 比較
    bootstrap.py    参加者単位のブートストラップ CI
    diagnostics.py  残差診断、等分散性、除外の影響
    report.py       整形と保存
    viz.py          サマリ図

実行スクリプト
--------------
    run_rt_lmm.py   反応時間の LMM（ランダム切片 vs 切片+傾き）

使い方
------
    python -m dualrdk.mixed.run_rt_lmm

新しい解析を足すときは ModelSpec を specs.py に定義し、run_*.py を
1 本増やす。fit / compare / report / viz はそのまま再利用できる。
"""
from dualrdk.mixed.specs import ModelSpec  # noqa: F401

__all__ = ["ModelSpec"]
