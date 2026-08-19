"""`zone_permutation.py` の帰無分布に対する検定。

置換検定の p 値は「帰無分布のどこに観測が落ちるか」であって、正規性も推定量の
不偏性も仮定しない。ここで問うのは 1 点だけ:

    OOZ 試行が少ないという非対称だけで、観測された Delta alpha が出るか。

出力の読み方:

    null_mean(mean_delta) が **非対称由来のバイアスの大きさ**そのもの。
    これが観測の +0.041 に近ければ、Delta alpha は人工物で説明がつく。
    0 に近ければ、非対称では説明できない。

    bias_corrected = observed - null_mean が、人工物を差し引いた推定値。

p 値は (1 + #{帰無 >= 観測}) / (B + 1) の形にする（+1 は観測自身を帰無分布の
一員と数えるため。B が有限のとき p = 0 を出さない標準的な扱い）。したがって
到達可能な最小 p は 1/(B+1) であり、B=200 なら 0.005 より小さくは出ない。

    python -m dualrdk.pomdp.zone_permutation_stats \\
        --input outputs/pomdp/zone_permutation \\
        --out   outputs/pomdp/zone_permutation/stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 帰無から見て「大きいほど効果あり」の向き。両側も併記する。
DIRECTED = {"mean_delta": "greater", "median_delta": "greater",
            "n_positive": "greater", "t": "greater"}
REPORT_STATS = ("mean_delta", "median_delta", "n_positive", "t")


def permutation_tests(observed: dict, null: pd.DataFrame,
                      stats=REPORT_STATS) -> pd.DataFrame:
    rows = []
    B = len(null)
    for name in stats:
        if name not in null.columns or name not in observed:
            continue
        v = null[name].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        obs = float(observed[name])
        mu, sd = float(v.mean()), float(v.std(ddof=1))

        # 片側（観測が帰無より大きい向き）と、帰無平均を中心にした両側
        p_greater = (1 + int((v >= obs).sum())) / (len(v) + 1)
        p_less = (1 + int((v <= obs).sum())) / (len(v) + 1)
        p_two = (1 + int((np.abs(v - mu) >= abs(obs - mu)).sum())) / (len(v) + 1)

        rows.append({
            "statistic": name,
            "observed": obs,
            "null_mean": mu,
            "null_sd": sd,
            "null_q2.5": float(np.percentile(v, 2.5)),
            "null_q50": float(np.percentile(v, 50)),
            "null_q97.5": float(np.percentile(v, 97.5)),
            "bias_corrected": obs - mu,
            "z_vs_null": (obs - mu) / sd if sd > 0 else np.nan,
            "p_greater": p_greater,
            "p_less": p_less,
            "p_two_sided": p_two,
            "n_valid_draws": len(v),
            "p_floor": 1.0 / (len(v) + 1),
        })
    return pd.DataFrame(rows)


def _sparkline(v: np.ndarray, obs: float, width=48) -> str:
    """帰無分布と観測位置の粗いヒストグラム。"""
    lo, hi = min(v.min(), obs), max(v.max(), obs)
    if hi <= lo:
        return ""
    edges = np.linspace(lo, hi, width + 1)
    cnt, _ = np.histogram(v, bins=edges)
    blocks = " ▁▂▃▄▅▆▇█"
    scaled = (cnt / cnt.max() * (len(blocks) - 1)).round().astype(int)
    bar = "".join(blocks[s] for s in scaled)
    pos = int(np.clip(np.searchsorted(edges, obs) - 1, 0, width - 1))
    marker = " " * pos + "^"
    return f"    {lo:+.3f} |{bar}| {hi:+.3f}\n             {marker} 観測 {obs:+.3f}"


def print_report(res: pd.DataFrame, null: pd.DataFrame, observed: dict, meta: dict) -> None:
    print(f"\n設定 {meta.get('rung')}  参加者 {meta.get('n_subjects')} 名  "
          f"置換 {meta.get('n_perm')} 回  多点スタート {meta.get('n_starts')}")

    print("\n=== 帰無分布（zone ラベルを円環シフト＝真の差ゼロ、非対称は保存）===")
    cols = ["statistic", "observed", "null_mean", "null_sd", "null_q2.5", "null_q97.5",
            "bias_corrected", "z_vs_null", "p_greater", "p_two_sided"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  到達可能な最小 p = {res['p_floor'].iloc[0]:.4f}（置換回数で決まる下限）")

    for name in ("mean_delta", "n_positive"):
        if name in null.columns and name in observed:
            print(f"\n  {name}:")
            print(_sparkline(null[name].to_numpy(dtype=float), float(observed[name])))

    md = res[res.statistic == "mean_delta"]
    if len(md):
        r = md.iloc[0]
        print("\n=== 解釈 ===")
        print(f"  非対称由来のバイアス（帰無平均）      : {r['null_mean']:+.4f}")
        print(f"  観測                                  : {r['observed']:+.4f}")
        print(f"  バイアス補正後                        : {r['bias_corrected']:+.4f}")
        if r["observed"] and np.sign(r["null_mean"]) == np.sign(r["observed"]):
            print(f"  観測のうち非対称で説明される割合      : {r['null_mean'] / r['observed']:.1%}")
        else:
            print("  観測のうち非対称で説明される割合      : —（帰無と観測の符号が逆）")
        if r["p_greater"] > 0.05:
            print("  -> 置換帰無を棄却できない。観測された Delta alpha は試行数の非対称"
                  "だけで説明がつく範囲にある。")
        else:
            print("  -> 置換帰無を棄却する。試行数の非対称では説明できない。"
                  "ただし消せた対抗仮説はこれ 1 つだけで、真の学習率差の証明ではない"
                  "（beta / lam のゾーン依存などは別に潰す必要がある）。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="円環置換の帰無分布に対する検定")
    ap.add_argument("--input", type=Path, default=Path("outputs/pomdp/zone_permutation"),
                    help="zone_permutation.py の出力ディレクトリ")
    ap.add_argument("--out", type=Path, default=None,
                    help="既定は <input>/stats")
    args = ap.parse_args(argv)

    src = Path(args.input)
    null = pd.read_csv(src / "null_draws.csv")
    observed = pd.read_csv(src / "observed.csv").iloc[0].to_dict()
    meta_path = src / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    res = permutation_tests(observed, null)

    out = Path(args.out) if args.out else src / "stats"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "permutation_tests.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "input": str(src),
        "source_meta": meta,
        "p_definition": "(1 + #{null >= observed}) / (B + 1)",
        "caveat": meta.get("caveat"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(res, null, observed, meta)
    print(f"\n[out] {out}/permutation_tests.csv, meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
