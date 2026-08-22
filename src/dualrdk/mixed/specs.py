"""モデル仕様。LMM と GLMM で共通のインターフェースを持たせる。

lme4 記法との対応
-----------------
    logrt ~ trial_c + (1 | subject)
        -> ModelSpec(formula="logrt ~ trial_c", re_formula=None,   groups="subject")
    logrt ~ trial_c + (1 + trial_c | subject)
        -> ModelSpec(formula="logrt ~ trial_c", re_formula="~trial_c", groups="subject")

`|` の左側が re_formula、右側が groups。re_formula は暗黙に切片を含むので
"~trial_c" は "~1 + trial_c" と同じ。切片を外すなら "~0 + trial_c"。

family を指定すると GLMM 扱いになる（dualrdk.mixed.glmm が処理する）。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    """1 つの混合モデルの仕様。

    Attributes
    ----------
    name : str
        出力に使う短い識別名。
    formula : str
        固定効果部分（patsy 記法）。`~` の右側が固定効果デザイン行列 X。
    re_formula : str or None
        ランダム効果部分。lme4 の `|` の左側。None は "~1"（ランダム切片のみ）。
    groups : str
        グループ化変数の列名。lme4 の `|` の右側。
    family : str or None
        None なら LMM。"binomial" 等を指定すると GLMM。
    label : str
        図表に出す説明。
    """

    name: str
    formula: str
    re_formula: Optional[str] = None
    groups: str = "subject"
    family: Optional[str] = None
    label: str = ""

    @property
    def is_glmm(self) -> bool:
        return self.family is not None

    @property
    def lme4(self) -> str:
        """lme4 記法での表現。論文に書くときはこれを使う。"""
        re = self.re_formula or "~1"
        terms = re.lstrip("~").strip()
        if terms in ("", "1"):
            inner = "1"
        elif terms.startswith(("0", "1")):
            inner = terms
        else:
            inner = f"1 + {terms}"
        return f"{self.formula} + ({inner} | {self.groups})"

    @property
    def n_re_params(self) -> int:
        """ランダム効果の分散共分散パラメータ数。

        ランダム項が q 個なら q(q+1)/2 個（分散 q 個 + 共分散 q(q-1)/2 個）。
        LMM ではこれに残差分散 1 個が加わる。
        """
        re = (self.re_formula or "~1").lstrip("~").strip()
        if re in ("", "1"):
            q = 1
        else:
            terms = [t.strip() for t in re.split("+") if t.strip()]
            has_intercept = "0" not in terms
            terms = [t for t in terms if t not in ("0", "1")]
            q = len(terms) + (1 if has_intercept else 0)
        return q * (q + 1) // 2


# ---------------------------------------------------------------------------
# 既定のモデル群
# ---------------------------------------------------------------------------

# 反応時間の LMM（今回の主解析）
RT_INTERCEPT = ModelSpec(
    name="M1_intercept",
    formula="logrt ~ trial_c",
    re_formula=None,
    label="ランダム切片のみ",
)

RT_SLOPE = ModelSpec(
    name="M2_slope",
    formula="logrt ~ trial_c",
    re_formula="~trial_c",
    label="ランダム切片 + ランダム傾き",
)

RT_MODELS = [RT_INTERCEPT, RT_SLOPE]


# 選択の GLMM（未実行。dualrdk.mixed.glmm 実装後に使う）
CHOICE_INTERCEPT = ModelSpec(
    name="G1_intercept",
    formula="chose_target ~ trial_s",
    re_formula=None,
    family="binomial",
    label="学習効果（ランダム切片）",
)

CHOICE_SLOPE = ModelSpec(
    name="G2_slope",
    formula="chose_target ~ trial_s",
    re_formula="~trial_s",
    family="binomial",
    label="学習効果（ランダム切片 + 傾き）",
)

CHOICE_MODELS = [CHOICE_INTERCEPT, CHOICE_SLOPE]
