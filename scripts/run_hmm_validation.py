"""HMM 妥当性検証（パラメータ回復・外的妥当性）のエントリポイント。"""
import sys
from pathlib import Path

# pip install -e . への移行までは src/ を直接パスに追加して dualrdk を解決する
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualrdk.models.hmm_validation import (
    run_recovery_analysis, evaluate_recovery_results,
    run_external_validation, print_validation_summary
)

recovery = run_recovery_analysis(n_trials=48, n_simulations=100)
recovery_eval = evaluate_recovery_results(recovery)

validation = run_external_validation(hmm_results, df_original)
print_validation_summary(recovery_eval, validation)
