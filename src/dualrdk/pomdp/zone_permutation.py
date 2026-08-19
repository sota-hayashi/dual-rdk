"""ゾーンラベルの循環置換による帰無分布（Delta alpha の人工物検証）。

M1z を参加者ごとの MAP で推定すると Delta alpha = alpha_out - alpha_in が
+0.041, 40/54 が正（符号検定 p = 0.0005）になる。ところがこのデータでは

    OOZ 試行数     中央値 12（範囲 2-19）
    非OOZ 試行数   中央値 36（範囲 28-46）
    n_out < n_in の参加者  54 / 54

であり、alpha_out は alpha_in の 1/3 の試行数で推定されている。データが少ない
側は事前分布の中心に強く引かれるので、**真の差がゼロでも** alpha_out だけが
上に持ち上がり Delta alpha > 0 が出る。実際、事前の中心が違う 2 つの段で

    R1 (Beta(2,2)):  alpha_in 0.4315 < alpha_out 0.4768 < 中心 0.5
    R4 (logit-N)  :  alpha_in 0.2053 < alpha_out 0.2462 < 中心 0.269

と、どちらも alpha_out が alpha_in と事前中心のちょうど間に落ちている。

「OOZ で学習率が高い」と「OOZ は試行が少ないのでシュリンケージが強い」は、
54 人全員が同じ向きの非対称を持つため**観測データ上は完全に交絡**していて、
回帰的な統制では分離できない。分離するには片方の原因を消したデータが要る。

そこで各参加者の zone ラベル列を**円環状にずらす**:

    実際   0 0 1 1 1 0 0 0 0 1 1 0 ...   （48 試行、うち out が 12）
    ずらし 1 0 0 0 0 1 1 0 0 0 0 1 ...   （48 試行、うち out が 12）

    保たれる: 実際の選択・報酬・刺激、各参加者の真の学習率、
              **試行数の非対称（12 対 36）**、OOZ のラン長構造、推定手続き
    壊れる  : どの試行が実際に OOZ だったか

つまり「非対称である」条件は温存したまま「OOZ 試行が本当に OOZ である」情報
だけを消す。置換下でも Delta alpha ≈ +0.04 が普通に出るなら非対称由来の人工物、
出ないなら非対称では説明できない。

i.i.d. のシャッフルではなく**円環シフト**を使うのは、OOZ ラベルが HMM 由来で
まとまって出るため。シャッフルするとラン長構造まで壊れ、「非対称だけを残す」
という条件が崩れる。

なおシフトが小さい引きでは置換後のラベルが実際のラベルとよく似るので、帰無
分布はわずかに観測側に寄る（＝検定は保守的になる）。結論を甘くする方向の
偏りなので既定では全ての非恒等シフトを許す。`--min-shift` で変更できる。

**この検定が消せるのは対抗仮説 1 つだけである。** 棄却できても「真の学習率差が
ある」の証明にはならない。OOZ 試行で beta や lam も違っていてモデルがそれを
alpha に押し込んでいる、という可能性は別に潰す必要がある。

    python -m dualrdk.pomdp.zone_permutation --n-perm 200 --n-jobs 8 \\
        --out outputs/pomdp/zone_permutation
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dualrdk.pomdp.ladder import GRID_STARTS, RUNGS, fit_subject, legacy_arrays, pomdp_arrays

STAT_NAMES = ("mean_delta", "median_delta", "sd_delta", "n_positive", "t",
              "mean_alpha_in", "mean_alpha_out")
DEFAULT_RUNG = "R4_pomdp_data"


# --------------------------------------------------------------------------
# 置換
# --------------------------------------------------------------------------
def shift_zone(arr: dict, shift: int) -> dict:
    """有効試行の zone ラベルだけを円環シフトする。

    無効試行を跨いでずらすと有効試行中の OOZ 個数が変わってしまい、保存したい
    「試行数の非対称」が崩れる。そこで有効試行の並びの中でだけ回す。
    """
    z = np.asarray(arr["zone"], dtype=int)
    ok = np.asarray(arr["valid"], dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return arr
    out = z.copy()
    out[idx] = np.roll(z[idx], shift % len(idx))
    return {**arr, "zone": out.tolist()}


def _statistics(rows: list[dict]) -> dict:
    d = np.array([r["delta_alpha"] for r in rows], dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    return {
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "sd_delta": sd,
        "n_positive": int((d > 0).sum()),
        "t": float(d.mean() / (sd / np.sqrt(n))) if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "mean_alpha_in": float(np.mean([r["alpha_in"] for r in rows])),
        "mean_alpha_out": float(np.mean([r["alpha_out"] for r in rows])),
    }


def _fit_all(data: dict, space: str, prior: str, starts) -> list[dict]:
    return [fit_subject(data[s], space=space, prior=prior, starts=starts) for s in sorted(data)]


def _one_rep(args):
    """1 回分の置換。ProcessPoolExecutor に渡すのでトップレベル関数にする。"""
    rep, seed, data, space, prior, starts, min_shift = args
    rng = np.random.default_rng(seed)
    permuted, shifts = {}, []
    for s in sorted(data):
        arr = data[s]
        n_ok = int(np.sum(arr["valid"]))
        hi = max(n_ok, min_shift + 1)
        sh = int(rng.integers(min_shift, hi)) if n_ok > 1 else 0
        shifts.append(sh)
        permuted[s] = shift_zone(arr, sh)
    stats = _statistics(_fit_all(permuted, space, prior, starts))
    return {"rep": rep, **stats, "mean_shift": float(np.mean(shifts))}


# --------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------
def pick_starts(n: int):
    """GRID_STARTS から n 点を均等に間引く（速度と局所解回避の折り合い）。"""
    if n >= len(GRID_STARTS):
        return GRID_STARTS
    idx = np.linspace(0, len(GRID_STARTS) - 1, n).round().astype(int)
    return tuple(GRID_STARTS[i] for i in dict.fromkeys(idx))


def run(data: dict, *, space: str, prior: str, n_perm=200, n_starts=4, seed=0,
        n_jobs=1, min_shift=1, verbose=True) -> tuple[dict, pd.DataFrame]:
    """観測統計量と、置換帰無分布を返す。

    観測側も**同じ n_starts で推定し直す**。多点スタートの数を変えると最適化の
    精度が変わるので、帰無分布と違う設定で得た観測値を突き合わせてはならない。
    """
    starts = pick_starts(n_starts)
    if verbose:
        print(f"[obs] 観測データで推定（starts={len(starts)}）")
    t0 = time.time()
    observed = _statistics(_fit_all(data, space, prior, starts))
    per_rep = time.time() - t0
    if verbose:
        print(f"[obs] {observed['mean_delta']:+.4f}  正 {observed['n_positive']}/{len(data)}"
              f"  （1 反復 {per_rep:.1f} 秒）")
        eta = per_rep * n_perm / max(n_jobs, 1) / 60
        print(f"[perm] {n_perm} 回 × {n_jobs} 並列  推定所要 {eta:.1f} 分")

    ss = np.random.SeedSequence(seed).spawn(n_perm)
    jobs = [
        (i, ss[i], data, space, prior, starts, min_shift) for i in range(n_perm)
    ]

    rows = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for k, r in enumerate(ex.map(_one_rep, jobs), 1):
                rows.append(r)
                if verbose and k % 10 == 0:
                    print(f"  [perm] {k}/{n_perm}", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            rows.append(_one_rep(j))
            if verbose and k % 10 == 0:
                print(f"  [perm] {k}/{n_perm}", flush=True)

    return observed, pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)


def check_counts(data: dict) -> pd.DataFrame:
    """置換が保存すべき量（有効試行中の OOZ 個数）を参加者ごとに記録する。"""
    rows = []
    for s in sorted(data):
        z = np.asarray(data[s]["zone"], dtype=int)
        ok = np.asarray(data[s]["valid"], dtype=bool)
        rows.append({"subject_id": s, "n_valid": int(ok.sum()),
                     "n_out": int(z[ok].sum()), "n_in": int((z[ok] == 0).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="zone ラベルの円環置換で Delta alpha の帰無分布を作る")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/online"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/zone_permutation"))
    ap.add_argument("--rung", default=DEFAULT_RUNG,
                    choices=[r[0] for r in RUNGS],
                    help="どの推定設定で回すか（既定は pomdp の事前・データ）")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-starts", type=int, default=4,
                    help="多点スタートの数。ladder.py の既定は 24 だが置換では速度優先")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--min-shift", type=int, default=1,
                    help="円環シフトの最小値。1 は恒等シフトのみ除く")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-subjects", type=int, default=None)
    args = ap.parse_args(argv)

    from dualrdk.io.load import load_all_concatenated
    from dualrdk.pomdp.data import tidy_from_concat_list

    name, space, prior, data_kind, desc = next(r for r in RUNGS if r[0] == args.rung)
    print(f"[load] {args.data_dir}  設定={name}（空間={space} 事前={prior} データ={data_kind}）")
    _, learning, _ = load_all_concatenated(args.data_dir)
    if args.limit_subjects:
        learning = learning[: args.limit_subjects]
    data = legacy_arrays(learning) if data_kind == "legacy" else pomdp_arrays(
        tidy_from_concat_list(learning)
    )

    counts = check_counts(data)
    print(f"[load] {len(data)} 名  OOZ 中央値 {counts.n_out.median():.0f}"
          f"（{counts.n_out.min()}-{counts.n_out.max()}）/ 非OOZ 中央値 {counts.n_in.median():.0f}"
          f"  n_out<n_in: {(counts.n_out < counts.n_in).sum()}/{len(counts)}")

    observed, null = run(
        data, space=space, prior=prior, n_perm=args.n_perm, n_starts=args.n_starts,
        seed=args.seed, n_jobs=args.n_jobs, min_shift=args.min_shift,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    null.to_csv(out / "null_draws.csv", index=False)
    pd.DataFrame([observed]).to_csv(out / "observed.csv", index=False)
    counts.to_csv(out / "trial_counts.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "rung": name, "space": space, "prior": prior, "data": data_kind, "desc": desc,
        "n_subjects": len(data), "n_perm": args.n_perm,
        "n_starts": len(pick_starts(args.n_starts)), "min_shift": args.min_shift,
        "seed": args.seed,
        "permutation": "有効試行の zone ラベルを円環シフト（個数とラン長を保存）",
        "caveat": "消せる対抗仮説は「試行数の非対称に由来する推定量のバイアス」1 つだけ。"
                  "棄却できても真の学習率差の証明にはならない。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[out] {out}/null_draws.csv, observed.csv, trial_counts.csv, meta.json")
    print("検定は次で: python -m dualrdk.pomdp.zone_permutation_stats "
          f"--input {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
