"""反応時間の線形混合モデル。

    M1:  logrt ~ trial_c + (1 | subject)
    M2:  logrt ~ trial_c + (1 + trial_c | subject)

試行を重ねるごとに反応時間がどう変わるか、その変化に個人差があるかを
調べる。M1 と M2 を境界補正した尤度比検定と AIC で比較し、採択された
モデルを詳細に評価する。

    python -m dualrdk.mixed.run_rt_lmm
    python -m dualrdk.mixed.run_rt_lmm --no-bootstrap --n-boot 100
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

from dualrdk.mixed import bootstrap as bs
from dualrdk.mixed import compare, diagnostics, report, viz
from dualrdk.mixed.config import (
    BOOTSTRAP_SEED,
    DEFAULT_OPTIMIZER,
    DEFAULT_REML,
    N_BOOTSTRAP,
    RT_CEILING_MS,
    RT_FLOOR_MS,
    run_dir,
)
from dualrdk.mixed.data import build_trial_frame, subject_level_dispersion
from dualrdk.mixed.lmm import fit_lmm, median_regression_check
from dualrdk.mixed.specs import RT_INTERCEPT, RT_MODELS, RT_SLOPE

ANALYSIS = "rt_lmm"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rt-ceiling", type=float, default=RT_CEILING_MS,
                    help="この ms を超える試行を除外（既定 15000）")
    ap.add_argument("--rt-floor", type=float, default=RT_FLOOR_MS,
                    help="この ms 未満を除外（既定 なし）")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    args = ap.parse_args(argv)

    out = run_dir(ANALYSIS)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(report.header("反応時間の線形混合モデル  logrt ~ trial + (…| subject)"))
    say(f"  出力先: {out}")

    # ================= 1. データ =========================================
    say(report.header("1.  データと試行除外", "-"))
    td = build_trial_frame(rt_ceiling_ms=args.rt_ceiling, rt_floor_ms=args.rt_floor)
    df = td.df
    say(td.exclusion_summary().to_string(index=False))

    n = td.trials_per_subject
    say(f"\n  参加者あたり試行数: min={n.min()}  Q1={n.quantile(.25):.0f}  "
        f"median={n.median():.0f}  max={n.max()}")
    for th in (45, 40, 35, 30):
        k = int((n < th).sum())
        say(f"    {th} 試行未満の参加者: {k} 名" + ("  ← 要確認" if th <= 30 and k else ""))
    if n.min() < 30:
        say("  ! 30 試行を切る参加者がいます。傾き推定が不安定になるので除外を検討してください")

    say(f"\n  log RT: mean={df.logrt.mean():.4f}  sd={df.logrt.std():.4f}  "
        f"skew={df.logrt.skew():+.3f}")
    say(f"  RT (ms): median={df.rt.median():.0f}  "
        f"[{df.rt.min():.0f}, {df.rt.max():.0f}]")

    # 除外の偏り
    say("\n  除外の系統性:")
    ti = diagnostics.trim_impact(df, td.removed)
    say(report.fmt_dict(ti, indent="    "))
    if ti.get("position_biased"):
        say("    ! 除外が試行位置に偏っています。trial の傾きが機械的に急になる恐れ")

    # rt_cv の保存
    raw_td = build_trial_frame(rt_ceiling_ms=None, rt_floor_ms=None)
    dp = diagnostics.dispersion_preservation(raw_td.df, df)
    say("\n  参加者ごとの散布度（rt_cv の材料）が除外で歪んでいないか:")
    say(report.fmt_dict(dp, indent="    "))

    # ================= 2. 推定 ===========================================
    say(report.header("2.  モデルの推定", "-"))
    results = {}
    for spec in RT_MODELS:
        r = fit_lmm(spec, df, reml=DEFAULT_REML, optimizer=args.optimizer)
        results[spec.name] = r
        say(f"\n  {spec.name}:  {spec.lme4}")
        say(f"  （{spec.label}）")
        say(f"    n_obs={r.n_obs}  n_groups={r.n_groups}  converged={r.converged}  "
            f"{'REML' if r.reml else 'ML'}  optimizer={r.optimizer}")
        say(f"    logLik={r.llf:.4f}  k={r.n_params}  AIC={r.aic:.3f}  BIC={r.bic:.3f}")
        say("\n" + report.fmt_fixed(r))
        say("\n" + report.fmt_varcomp(r))

    m1, m2 = results[RT_INTERCEPT.name], results[RT_SLOPE.name]

    # ================= 3. モデル比較 =====================================
    say(report.header("3.  モデル比較", "-"))
    cmp = compare.lrt(m1, m2, boundary=True)
    say(f"  {'':>4}{'model':<14}{'logLik':>12}{'k':>4}{'AIC':>11}{'BIC':>11}")
    for r in (m1, m2):
        say(f"  {'':>4}{r.spec.name:<14}{r.llf:>12.4f}{r.n_params:>4}{r.aic:>11.3f}{r.bic:>11.3f}")
    say(f"\n  尤度比検定  chi2({cmp['df']}) = {cmp['chi2']:.4f}")
    say(f"    p（境界補正 {cmp['reference']}）= {cmp['p_boundary']:.4f}   ← こちらが正しい参照分布")
    say(f"    p（素朴 chi2({cmp['df']})）      = {cmp['p_naive']:.4f}")
    say(f"  dAIC = {cmp['dAIC']:+.3f}   dBIC = {cmp['dBIC']:+.3f}")
    say(f"\n  → {compare.selection_note(cmp)}")
    say("\n  注: 分散が 0 という帰無仮説はパラメータ空間の境界にあるため、"
        "素朴な chi2 は保守的すぎる")

    cmp_table = compare.comparison_table([m1, m2])
    best = results[cmp_table.loc[cmp_table["selected"], "model"].iloc[0]]

    # ================= 4. 採択モデルの詳細 ================================
    say(report.header(f"4.  採択モデル {best.spec.name} の詳細評価", "-"))
    say(f"  {best.spec.lme4}\n")
    say(report.fmt_fixed(best))
    say("\n" + report.fmt_varcomp(best))

    r2 = best.r2(df)
    say("\n  分散説明率:")
    say(f"    R2_marginal    = {r2['r2_marginal']:.4f}   固定効果のみ")
    say(f"    R2_conditional = {r2['r2_conditional']:.4f}   固定 + ランダム")
    say(f"    ICC            = {best.icc():.4f}   参加者間分散の比（trial_c=0 時点）")

    say("\n  最適化の頑健性:")
    say(report.fmt_convergence(best))

    # 傾きの独立検証（トリムの影響）
    mr = median_regression_check(raw_td.df, "logrt ~ trial_c", "trial_c")
    b1 = best.fixed.loc["trial_c", "estimate"]
    say(f"\n  傾きの独立検証（試行を削らない中央値回帰）:")
    say(f"    中央値回帰 beta1 = {mr['estimate']:+.6f}（全 {len(raw_td.df)} 試行）")
    say(f"    LMM        beta1 = {b1:+.6f}   比 = {b1 / mr['estimate']:.2f}")
    say(f"    ※ {mr['note']}。除外により傾きが膨らむため、大きさは幅で報告すること")

    # 予測軌跡
    b0 = best.fixed.loc["Intercept", "estimate"]
    tmax = df.trial_c.max()
    say(f"\n  予測される軌跡:")
    for t in [0, tmax // 4, tmax // 2, 3 * tmax // 4, tmax]:
        say(f"    trial {int(t)+1:2d}:  log RT {b0 + b1*t:.4f}  ->  {np.exp(b0 + b1*t):7.1f} ms")
    say(f"    全体変化 {100*(np.exp(b1*tmax)-1):+.2f}%  ({np.exp(b0+b1*tmax)-np.exp(b0):+.0f} ms)")

    if "trial_c" in best.random_effects.columns:
        sl = (b1 + best.random_effects["trial_c"]).values
        tau1 = best.varcomp.loc["trial_c", "sd"]
        say(f"\n  個人差の大きさ:")
        say(f"    BLUP 傾き  mean={sl.mean():+.6f}  sd={sl.std(ddof=1):.6f}  "
            f"[{sl.min():+.6f}, {sl.max():+.6f}]")
        say(f"    {int(tmax)+1} 試行での RT 変化率  中央値 {100*(np.exp(np.median(sl)*tmax)-1):+.1f}%  "
            f"最速化 {100*(np.exp(sl.min()*tmax)-1):+.1f}%  最遅化 {100*(np.exp(sl.max()*tmax)-1):+.1f}%")
        say(f"    tau1 の ±1SD 帯: {100*(np.exp((b1-tau1)*tmax)-1):+.1f}% .. "
            f"{100*(np.exp((b1+tau1)*tmax)-1):+.1f}%")
        say(f"    傾きが正（遅くなる）の参加者: {int((sl > 0).sum())} / {len(sl)}")

    # ================= 5. ブートストラップ ================================
    boot_ci = None
    if not args.no_bootstrap:
        say(report.header(f"5.  参加者ブートストラップ（{args.n_boot} 反復）", "-"))
        t0 = time.time()
        draws = bs.bootstrap_lmm(best.spec, df, n_boot=args.n_boot,
                                 seed=BOOTSTRAP_SEED, optimizer=args.optimizer)
        boot_ci = bs.bootstrap_ci(draws, point=bs.point_dict(best))
        best.bootstrap = draws
        say(f"  {len(draws)}/{args.n_boot} 回成功（{time.time()-t0:.0f} 秒）\n")
        say(report.fmt_bootstrap(boot_ci))
        say("\n  ※ 分散パラメータの Wald 型 SE は境界近傍で信用できないため、"
            "CI はリサンプリングで出している")

    # ================= 6. 診断 ============================================
    say(report.header("6.  残差診断と仮定の検証", "-"))
    diag = diagnostics.residual_diagnostics(best, df)
    say(report.fmt_dict(diag, indent="  "))
    if not diag["homoscedastic"]:
        say("\n  ! 等分散仮定は棄却されました。")
        say("    MixedLM の残差分散 sigma^2 には参加者の添字がなく、"
            "「全員の試行間ばらつきは等しい」と仮定している。")
        say(f"    実際には参加者ごとの残差 SD が {diag['resid_sd_min']:.3f}–"
            f"{diag['resid_sd_max']:.3f}（{diag['resid_sd_ratio']:.2f} 倍）に散らばる。")
        say("    この散らばりこそが rt_cv（MW 指標）であり、"
            "本モデルは仮定によってそれを消している。")
        say("    → 次の一手は sigma に添字 i を与えるモデル（location-scale）。")

    # ================= 7. 保存 ============================================
    say(report.header("7.  保存", "-"))
    disp = subject_level_dispersion(td)
    per_sd = diagnostics.per_subject_residual_sd(best, df)
    tables = {
        "exclusion_summary": td.exclusion_summary(),
        "model_comparison": cmp_table,
        "fixed_effects_M1": m1.fixed,
        "fixed_effects_M2": m2.fixed,
        "varcomp_M1": m1.varcomp,
        "varcomp_M2": m2.varcomp,
        f"random_effects_{best.spec.name}": best.random_effects,
        "convergence_check": best.convergence_check,
        "subject_dispersion": disp,
        "per_subject_residual_sd": per_sd,
        "bootstrap_ci": boot_ci,
        "bootstrap_draws": best.bootstrap,
    }
    for p in report.save_tables(out, tables):
        say(f"  {p.relative_to(out.parent.parent)}")

    payload = {
        "analysis": ANALYSIS,
        "config": {
            "rt_ceiling_ms": args.rt_ceiling,
            "rt_floor_ms": args.rt_floor,
            "reml": DEFAULT_REML,
            "optimizer": args.optimizer,
            "n_bootstrap": 0 if args.no_bootstrap else args.n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "data": {
            "n_trials": td.n_trials, "n_subjects": td.n_subjects,
            "n_removed_rt_ceiling": td.n_rt_ceiling,
            "trials_per_subject_min": int(n.min()),
            "trials_per_subject_median": float(n.median()),
        },
        "models": {r.spec.name: {
            "lme4": r.spec.lme4, "logLik": r.llf, "AIC": r.aic, "BIC": r.bic,
            "converged": r.converged,
            "fixed": r.fixed.to_dict(orient="index"),
            "varcomp": r.varcomp.to_dict(orient="index"),
            "corr": r.corr,
        } for r in (m1, m2)},
        "selected": best.spec.name,
        "lrt": cmp,
        "r2": r2,
        "icc": best.icc(),
        "diagnostics": diag,
        "trim_impact": ti,
        "dispersion_preservation": dp,
        "median_regression_check": mr,
    }
    say(f"  {report.save_json(out, 'results', payload).relative_to(out.parent.parent)}")

    # ================= 8. 図 ==============================================
    f1 = viz.plot_lmm_summary(
        best, df, out / "figures" / f"{ANALYSIS}_summary.pdf",
        boot_ci=boot_ci, diag=diag,
        title=f"{best.spec.lme4}    (RT <= {args.rt_ceiling:.0f} ms, "
              f"n={td.n_trials} trials / {td.n_subjects} subjects)")
    f2 = viz.plot_model_comparison(
        cmp_table, out / "figures" / f"{ANALYSIS}_model_comparison.pdf",
        title="ランダム効果構造の比較")
    for f in (f1, f2):
        say(f"  {f.relative_to(out.parent.parent)}")

    report.save_text(out, "summary", "\n".join(lines))
    print(f"\n  summary.txt / results.json を {out} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
