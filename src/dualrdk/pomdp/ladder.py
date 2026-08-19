"""既存 MAP 実装と pomdp M1z の推定値の差を、4 段の梯子で切り分ける。

`models/q_learning_ooz_map.py`（制約空間・Beta/Gamma 事前・chosen_color 由来の
選択）と `pomdp` の M1z（無制約空間・logit/log-normal 事前・応答角由来の選択）は
同じ Q 学習モデルなのに推定される alpha が大きく違う。両者は「座標系」「事前分布」
「データ整形」の 3 点で同時に異なるので、一度に入れ替えると原因が特定できない。

そこで既存実装から出発し、**一度に一つだけ**差し替える:

    R1  legacy_constrained  : 既存実装そのもの
                              制約空間で最適化 / Beta(2,2), Gamma(2,3), N(0,2)
                              / chosen_color・target_item 由来のデータ
    R2  legacy_unconstrained: 座標系だけ pomdp に合わせる
                              無制約空間 (logit alpha, log beta) で最適化。事前
                              分布は R1 と**同じ分布**を座標変換して書き直す
                              （対数ヤコビアンを加える）。データは R1 のまま。
    R3  pomdp_prior         : その上で事前分布だけ差し替える
                              logit alpha ~ N(-1.0, 1.2), log beta ~ N(log 5, 1.0),
                              c_black ~ N(0, 1)。データは R1 のまま。
    R4  pomdp_data          : その上でデータ整形も差し替える
                              選択を応答角から導き（最近傍の雲）、状態を
                              target_deg == theta_white から導き、無効試行は
                              除去ではなく valid マスクで飛ばす。

R1 -> R2 の差が座標系の寄与、R2 -> R3 が事前分布の寄与、R3 -> R4 がデータ整形の
寄与になる。尤度・モデル・最適化手順は 4 段すべてで共通なので、差分は上の 3 因子
以外から生じない。

MAP は再パラメータ化で不変ではない（同じ分布でも最頻値を取る空間が変われば別の
点が出る）ので、R2 を挟まずに R1 と R3 を比べても、事前の違いなのか座標の違いなの
かが区別できない。R2 はそのための踏み台である。

推定は全段で「参加者ごとに独立な MAP」。階層ベイズのシュリンケージはこの梯子には
含まれない（それは 4 段目のさらに先にある別の因子）。

    python -m dualrdk.pomdp.ladder --data-dir data/raw/online \\
        --out outputs/pomdp/ladder
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PARAM_NAMES = ("alpha_in", "alpha_out", "beta", "c_black")

# 探索の初期値（制約スケール）。無制約段では logit / log で写して使うので、
# 4 段はまったく同じ点から探索を始める。
GRID_STARTS = tuple(
    itertools.product((0.2, 0.7), (0.2, 0.7), (2.0, 8.0), (-1.5, 0.0, 1.5))
)

_EPS = 1e-6
CONSTRAINED_BOUNDS = ((_EPS, 1 - _EPS), (_EPS, 1 - _EPS), (_EPS, None), (None, None))

_LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------
# 対数尤度（4 段で共通）
# --------------------------------------------------------------------------
# 符号規約は pomdp（models._m1_step）に合わせる:
#     d = Q[s, white] - Q[s, black],  z = beta * d - c_black,  P(white) = sigmoid(z)
# 既存実装は dq = beta * (Q[s, black] - Q[s, white]) + c_black で z = -dq。
# P(white) = 1/(1 + exp(dq)) = sigmoid(z) なので同一の式である（c_black > 0 が
# 黒選好、という向きも一致する）。


def _log_sigmoid(x: float) -> float:
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def log_lik(alpha_in: float, alpha_out: float, beta: float, c_black: float, arr: dict) -> float:
    """M1z の対数尤度（2 値選択の確率質量）。

    arr は chose_white / s_state / r_norm / zone / valid を持つ辞書。無効試行は
    尤度に寄与せず、Q も更新しない（モデル化しない試行が内部状態を動かすのは
    筋が通らないため）。
    """
    cw = arr["chose_white"]
    ss = arr["s_state"]
    rr = arr["r_norm"]
    zz = arr["zone"]
    ok = arr["valid"]

    q = [[0.5, 0.5], [0.5, 0.5]]  # [状態][0=白, 1=黒]
    total = 0.0
    for t in range(len(cw)):
        if not ok[t]:
            continue
        row = q[ss[t]]
        z = beta * (row[0] - row[1]) - c_black
        total += _log_sigmoid(z) if cw[t] else _log_sigmoid(-z)

        alpha = alpha_out if zz[t] == 1 else alpha_in
        j = 0 if cw[t] else 1
        row[j] += alpha * (rr[t] - row[j])
    return total


# --------------------------------------------------------------------------
# 事前分布
# --------------------------------------------------------------------------
# scipy.stats を内側ループで呼ぶと呼び出し overhead が支配的になるので閉じた形で書く。
_LOG_BETA22 = math.log(6.0)  # Beta(2,2) の正規化定数 1/B(2,2) = 6
_LOG_GAMMA23 = math.log(9.0)  # Gamma(k=2, scale=3) の 3^2 * Gamma(2) = 9


def _log_normal_pdf(x: float, loc: float, scale: float) -> float:
    zz = (x - loc) / scale
    return -0.5 * zz * zz - math.log(scale) - 0.5 * _LOG_2PI


def log_prior_legacy_constrained(theta) -> float:
    """既存実装の事前分布を、制約スケールの密度として評価する。

        alpha ~ Beta(2, 2),  beta ~ Gamma(shape=2, scale=3),  c_black ~ N(0, 2)
    """
    a_in, a_out, b, c = theta
    if not (0.0 < a_in < 1.0 and 0.0 < a_out < 1.0 and b > 0.0):
        return -math.inf
    return (
        _LOG_BETA22 + math.log(a_in) + math.log1p(-a_in)
        + _LOG_BETA22 + math.log(a_out) + math.log1p(-a_out)
        + math.log(b) - b / 3.0 - _LOG_GAMMA23
        + _log_normal_pdf(c, 0.0, 2.0)
    )


def log_prior_legacy_unconstrained(vec) -> float:
    """**同じ分布**を無制約スケールの密度として書き直したもの。

    分布そのものは R1 と一字一句同じで、座標が変わった分の対数ヤコビアン

        log |d alpha / d alpha~|  = log alpha + log(1 - alpha)      （logit）
        log |d beta  / d beta~ |  = log beta  = beta~               （log）

    を足すだけ。これを足さないと「別の分布」を置いたことになってしまう。
    """
    a_in, a_out, b, c = _to_constrained(vec)
    log_jac = (
        math.log(a_in) + math.log1p(-a_in)
        + math.log(a_out) + math.log1p(-a_out)
        + vec[2]
    )
    return log_prior_legacy_constrained((a_in, a_out, b, c)) + log_jac


def log_prior_pomdp_unconstrained(vec) -> float:
    """pomdp の事前分布（models.py の ParamSpec）。

        logit alpha ~ N(-1.0, 1.2)   _alpha_spec
        log beta    ~ N(log 5, 1.0)  _BETA
        c_black     ~ N(0, 1)        _C_BLACK

    こちらは**もともと無制約スケールで定義された密度**なので、ヤコビアンを
    足す必要はない（足したら別の分布になる）。R2 との非対称はここにある。
    """
    return (
        _log_normal_pdf(vec[0], -1.0, 1.2)
        + _log_normal_pdf(vec[1], -1.0, 1.2)
        + _log_normal_pdf(vec[2], math.log(5.0), 1.0)
        + _log_normal_pdf(vec[3], 0.0, 1.0)
    )


def _to_constrained(vec):
    return (_sigmoid(vec[0]), _sigmoid(vec[1]), math.exp(vec[2]), vec[3])


def _to_unconstrained(theta):
    a_in, a_out, b, c = theta
    return (math.log(a_in / (1 - a_in)), math.log(a_out / (1 - a_out)), math.log(b), c)


# --------------------------------------------------------------------------
# 梯子の定義
# --------------------------------------------------------------------------
RUNGS = (
    # name, 座標系, 事前, データ, 説明
    ("R1_legacy_constrained", "constrained", "legacy", "legacy", "既存実装そのもの"),
    ("R2_legacy_unconstrained", "unconstrained", "legacy", "legacy", "座標系だけ差し替え"),
    ("R3_pomdp_prior", "unconstrained", "pomdp", "legacy", "＋事前分布を差し替え"),
    ("R4_pomdp_data", "unconstrained", "pomdp", "pomdp", "＋データ整形を差し替え"),
)
RUNG_NAMES = tuple(r[0] for r in RUNGS)


# --------------------------------------------------------------------------
# データ整形（2 通り）
# --------------------------------------------------------------------------
def legacy_arrays(concat_list) -> dict[str, dict]:
    """既存実装のデータ整形。

    - 選択  : `chosen_color`（io.load.infer_choice）。応答が**両方の雲から 45 度
              より遠い**試行は chosen_item = -1 となり NaN、dropna で落ちる。
    - 状態  : `target_item`（0 = 白がターゲット / 1 = 黒）。
    - 報酬  : `reward_points / 10`。
    - 無効  : rt / chosen_color / ooz の NaN 行を **除去**する（行数が参加者ごと
              に変わる）。
    """
    required = {"rt", "chosen_color", "reward_points", "target_item", "ooz"}
    out = {}
    for subj_id, df in concat_list:
        if not required.issubset(df.columns):
            continue
        work = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
        if work.empty:
            continue
        color = work["chosen_color"].astype(str).to_numpy()
        if not np.isin(color, ("white", "black")).all():
            raise ValueError(f"{subj_id}: chosen_color に white/black 以外がある")
        out[str(subj_id)] = {
            "chose_white": (color == "white").tolist(),
            "s_state": work["target_item"].to_numpy(dtype=int).tolist(),
            "r_norm": (work["reward_points"].to_numpy(dtype=float) / 10.0).tolist(),
            "zone": work["ooz"].to_numpy(dtype=int).tolist(),
            "valid": [True] * len(work),
        }
    return out


def pomdp_arrays(tidy: pd.DataFrame) -> dict[str, dict]:
    """pomdp のデータ整形（data.arrays_from_tidy / likelihood.derive_choice_fields）。

    - 選択  : 応答角がどちらの雲の**格子インデックスに近いか**。45 度の閾値を
              持たないので、既存実装が捨てた「どちらからも遠い」試行にも選択が
              割り当てられる。
    - 状態  : target_deg の格子インデックスが白の雲と一致するか。
    - 報酬  : `reward / REWARD_MAX`（= /10。既存と同じ）。
    - 無効  : 48 行すべて残し、`valid` マスクで尤度と更新から飛ばす。
    """
    from dualrdk.pomdp.data import arrays_from_tidy

    subjects, trials = arrays_from_tidy(tidy)
    out = {}
    for i, s in enumerate(subjects):
        out[str(s)] = {
            "chose_white": np.asarray(trials["chose_white"][i]).tolist(),
            "s_state": np.asarray(trials["s_state"][i]).astype(int).tolist(),
            "r_norm": np.asarray(trials["r_norm"][i]).astype(float).tolist(),
            "zone": np.asarray(trials["zone"][i]).astype(int).tolist(),
            "valid": np.asarray(trials["valid"][i]).astype(bool).tolist(),
        }
    return out


# --------------------------------------------------------------------------
# 推定
# --------------------------------------------------------------------------
def _make_objective(space: str, prior: str, arr: dict):
    """-log p(y | theta) - log p(theta) を返す。theta は space の座標で受ける。"""
    if space == "constrained":
        if prior != "legacy":
            raise ValueError("制約空間で pomdp の事前は定義できない（無制約スケールの密度）")

        def obj(v):
            theta = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            lp = log_prior_legacy_constrained(theta)
            if not math.isfinite(lp):
                return 1e12
            return -(log_lik(*theta, arr) + lp)

        return obj

    log_prior = log_prior_legacy_unconstrained if prior == "legacy" else log_prior_pomdp_unconstrained

    def obj(v):
        vec = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        lp = log_prior(vec)
        if not math.isfinite(lp):
            return 1e12
        return -(log_lik(*_to_constrained(vec), arr) + lp)

    return obj


def fit_subject(arr: dict, *, space: str, prior: str, starts=GRID_STARTS):
    """1 参加者を多点スタートで MAP 推定する。制約スケールの推定値を返す。"""
    obj = _make_objective(space, prior, arr)
    unconstrained = space == "unconstrained"
    bounds = None if unconstrained else list(CONSTRAINED_BOUNDS)

    best = None
    for st in starts:
        x0 = np.asarray(_to_unconstrained(st) if unconstrained else st, dtype=float)
        res = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res

    theta = _to_constrained(best.x) if unconstrained else tuple(float(x) for x in best.x)
    return {
        **dict(zip(PARAM_NAMES, (float(v) for v in theta))),
        "delta_alpha": float(theta[1] - theta[0]),
        "log_lik": float(log_lik(*theta, arr)),
        "log_posterior": float(-best.fun),
        "n_trials": int(sum(arr["valid"])),
        "converged": bool(best.success),
    }


def fit_rung(name: str, space: str, prior: str, data: dict, *, verbose=True) -> pd.DataFrame:
    rows = []
    subjects = sorted(data)
    for i, s in enumerate(subjects):
        rows.append({"rung": name, "subject_id": s, **fit_subject(data[s], space=space, prior=prior)})
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{name}] {i + 1}/{len(subjects)} 名", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 検証: この梯子の尤度が両実装の尤度と一致することを確かめる
# --------------------------------------------------------------------------
def verify(legacy: dict, pomdp: dict, *, n_draws=5, seed=0) -> dict:
    """R1 が既存実装と、R4 が pomdp の M1z と、同じ尤度を計算しているか確認する。

    ここが一致していなければ梯子の 4 段は「同じモデルの 4 通りの推定」ではなく
    なるので、結果を読む前にこれを見ること。
    """
    rng = np.random.default_rng(seed)
    draws = [
        (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)),
         float(rng.uniform(0.5, 12.0)), float(rng.normal(0.0, 1.5)))
        for _ in range(n_draws)
    ]
    report = {"n_draws": n_draws}

    # --- R1 vs models/q_learning_ooz_map.py ---
    from dualrdk.models.q_learning_ooz_map import _compute_log_likelihood

    diffs = []
    for s in sorted(legacy)[:3]:
        arr = legacy[s]
        choices = np.array([0 if c else 1 for c in arr["chose_white"]])  # 1=black, 0=white
        rewards = np.asarray(arr["r_norm"], dtype=float)
        states = np.asarray(arr["s_state"], dtype=int)
        ooz = np.asarray(arr["zone"], dtype=int)
        for th in draws:
            mine = log_lik(*th, arr)
            theirs = _compute_log_likelihood(th[0], th[1], th[2], th[3], choices, rewards, states, ooz)
            diffs.append(abs(mine - theirs))
    report["max_abs_diff_vs_legacy"] = float(max(diffs)) if diffs else None

    # --- R4 vs pomdp の M1z ---
    import jax.numpy as jnp

    from dualrdk.pomdp.likelihood import subject_log_lik
    from dualrdk.pomdp.models import build_model

    model = build_model("M1z")
    diffs = []
    for s in sorted(pomdp)[:3]:
        arr = pomdp[s]
        trials_i = {
            "chose_white": jnp.asarray(arr["chose_white"]),
            "s_state": jnp.asarray(arr["s_state"]),
            "r_norm": jnp.asarray(arr["r_norm"], dtype=float),
            "zone": jnp.asarray(arr["zone"]),
            "valid": jnp.asarray(arr["valid"]),
        }
        for th in draws:
            mine = log_lik(*th, arr)
            params = {
                "alpha_in": jnp.asarray(th[0]), "alpha_out": jnp.asarray(th[1]),
                "beta": jnp.asarray(th[2]), "c_black": jnp.asarray(th[3]),
            }
            theirs = float(subject_log_lik(model, params, trials_i))
            diffs.append(abs(mine - theirs))
    report["max_abs_diff_vs_pomdp_M1z"] = float(max(diffs)) if diffs else None
    report["passes"] = bool(
        (report["max_abs_diff_vs_legacy"] or 0) < 1e-8
        and (report["max_abs_diff_vs_pomdp_M1z"] or 0) < 1e-6
    )
    return report


# --------------------------------------------------------------------------
# 要約
# --------------------------------------------------------------------------
def summarize(per_subject: pd.DataFrame) -> pd.DataFrame:
    """段ごとの群レベル要約と、R1 および直前段との一致度。"""
    wide = {r: d.set_index("subject_id") for r, d in per_subject.groupby("rung")}
    ref = wide[RUNG_NAMES[0]]

    rows = []
    for k, name in enumerate(RUNG_NAMES):
        if name not in wide:
            continue
        d = wide[name]
        alpha = pd.concat([d["alpha_in"], d["alpha_out"]])
        row = {
            "rung": name,
            "n_subjects": len(d),
            "mean_alpha": float(alpha.mean()),
            "median_alpha": float(alpha.median()),
            "mean_alpha_in": float(d["alpha_in"].mean()),
            "mean_alpha_out": float(d["alpha_out"].mean()),
            "mean_delta_alpha": float(d["delta_alpha"].mean()),
            "mean_beta": float(d["beta"].mean()),
            "mean_c_black": float(d["c_black"].mean()),
            "alpha_at_boundary_frac": float(
                ((alpha < 0.01) | (alpha > 0.99)).mean()
            ),
            "mean_log_lik_per_trial": float((d["log_lik"] / d["n_trials"]).mean()),
            "n_not_converged": int((~d["converged"]).sum()),
        }
        prev = wide.get(RUNG_NAMES[k - 1]) if k else None
        for tag, other in (("vs_R1", ref), ("vs_prev", prev)):
            if other is None or name == RUNG_NAMES[0]:
                continue
            common = d.index.intersection(other.index)
            row[f"n_common_{tag}"] = len(common)
            for p in ("alpha_in", "alpha_out", "beta", "c_black"):
                x, y = other.loc[common, p], d.loc[common, p]
                row[f"r_{p}_{tag}"] = float(np.corrcoef(x, y)[0, 1]) if len(common) > 2 else np.nan
            row[f"d_mean_alpha_{tag}"] = float(
                pd.concat([d.loc[common, "alpha_in"], d.loc[common, "alpha_out"]]).mean()
                - pd.concat([other.loc[common, "alpha_in"], other.loc[common, "alpha_out"]]).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _print_report(summary: pd.DataFrame, verification: dict) -> None:
    print("\n=== 検証（梯子の尤度が両実装と一致するか）===")
    print(f"  R1 vs models/q_learning_ooz_map : 最大絶対差 {verification['max_abs_diff_vs_legacy']:.3e}")
    print(f"  R4 vs pomdp M1z                 : 最大絶対差 {verification['max_abs_diff_vs_pomdp_M1z']:.3e}")
    print(f"  判定: {'一致' if verification['passes'] else '不一致（結果を読む前に原因を調べること）'}")

    print("\n=== 段ごとの群レベル要約 ===")
    cols = ["rung", "n_subjects", "mean_alpha", "mean_alpha_in", "mean_alpha_out",
            "mean_delta_alpha", "mean_beta", "mean_c_black", "alpha_at_boundary_frac",
            "mean_log_lik_per_trial"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== 直前の段からの変化（＝その因子だけの寄与）===")
    labels = dict(zip(RUNG_NAMES, ("—", "座標系", "事前分布", "データ整形")))
    prev_alpha = None
    for _, r in summary.iterrows():
        cur = r["mean_alpha"]
        delta = "" if prev_alpha is None else f"{cur - prev_alpha:+.4f}"
        print(f"  {r['rung']:<24s} 因子={labels[r['rung']]:<10s} mean_alpha={cur:.4f}  変化={delta}")
        prev_alpha = cur

    print("\n=== 参加者ごとの推定値の相関（R1 との一致度）===")
    rcols = [c for c in summary.columns if c.startswith("r_") and c.endswith("_vs_R1")]
    if rcols:
        print(summary[["rung", *rcols]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="既存 MAP と pomdp M1z の差を 4 段で切り分ける")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/online"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/ladder"))
    ap.add_argument("--limit-subjects", type=int, default=None,
                    help="先頭 N 名だけで走らせる（動作確認用）")
    ap.add_argument("--rungs", default=",".join(RUNG_NAMES),
                    help=f"実行する段をカンマ区切りで指定（既定は全段）: {RUNG_NAMES}")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from dualrdk.io.load import load_all_concatenated
    from dualrdk.pomdp.data import tidy_from_concat_list

    print(f"[load] {args.data_dir}")
    _, learning, _ = load_all_concatenated(args.data_dir)
    if args.limit_subjects:
        learning = learning[: args.limit_subjects]

    legacy = legacy_arrays(learning)
    pomdp = pomdp_arrays(tidy_from_concat_list(learning))
    print(f"[load] legacy 整形 {len(legacy)} 名 / pomdp 整形 {len(pomdp)} 名")

    n_leg = sum(len(v["chose_white"]) for v in legacy.values())
    n_pom = sum(int(np.sum(v["valid"])) for v in pomdp.values())
    print(f"[load] 有効試行数  legacy {n_leg} / pomdp {n_pom}（差 {n_pom - n_leg:+d}）")

    verification = {} if args.skip_verify else verify(legacy, pomdp)

    wanted = [r.strip() for r in args.rungs.split(",") if r.strip()]
    frames = []
    for name, space, prior, data_kind, desc in RUNGS:
        if name not in wanted:
            continue
        print(f"\n[fit] {name}  ({desc})  空間={space} 事前={prior} データ={data_kind}")
        data = legacy if data_kind == "legacy" else pomdp
        frames.append(fit_rung(name, space, prior, data, verbose=not args.quiet))

    per_subject = pd.concat(frames, ignore_index=True)
    summary = summarize(per_subject)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(out / "per_subject.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    meta = {
        "rungs": [dict(zip(("name", "space", "prior", "data", "desc"), r)) for r in RUNGS],
        "n_valid_trials": {"legacy": n_leg, "pomdp": n_pom},
        "n_starts": len(GRID_STARTS),
        "verification": verification,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if verification:
        _print_report(summary, verification)
    else:
        print(summary.to_string(index=False))
    print(f"\n[out] {out}/per_subject.csv, summary.csv, meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
