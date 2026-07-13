from stats.hmm_validation import (                                                                                                            
    run_recovery_analysis, evaluate_recovery_results,                                                                                         
    run_external_validation, print_validation_summary                                                                                         
)                                                                                                                                             
                                                                                                                                            
recovery = run_recovery_analysis(n_trials=48, n_simulations=100)                                                                              
recovery_eval = evaluate_recovery_results(recovery)                                                                                           
                                                                                                                                            
validation = run_external_validation(hmm_results, df_original)                                                                                
print_validation_summary(recovery_eval, validation)  