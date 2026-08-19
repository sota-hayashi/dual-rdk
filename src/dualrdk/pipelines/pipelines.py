from dualrdk.config import ONLINE_DATA_DIR
from dualrdk.io.load import load_all_concatenated
from dualrdk.pipelines.pipelines_behavior import run_behavior, get_subjects_by_behavior_data
from dualrdk.pipelines.pipelines_hmm import run_hmm



def run_default():
    # 1) データを一度だけロード
    all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
        ONLINE_DATA_DIR,
        subjects_include=None
    )

    # 2) 行動指標の計算（除外は 1) のロード時点で適用済み）
    #
    # 以前はここで rt_cv 上位10%を分離し、3) で subjects_include を指定して
    # 再ロードしていた。その除外は config.EXCLUDED_SUBJECTS に凍結したため、
    # 再ロードは同じ被験者集合を読み直すだけの無駄になったので削除した。
    # label_if_ooz の群閾値も同一集合から計算されるため、結果は変わらない。
    subjects_behavior_on, subjects_behavior_off, behavioral_df = get_subjects_by_behavior_data(all_data_learning, threshold=1)

    # 4) HMM解析
    hmm_df = run_hmm(
        all_data_learning=all_data_learning,
        train=False,
        input_type="angular_error",
    )

    # 5) 行動解析
    run_behavior(
        all_data_learning, all_data_awareness,
        subjects_behavior_on, subjects_behavior_off,
        hmm_df=hmm_df, behavioral_df=behavioral_df,
    )
