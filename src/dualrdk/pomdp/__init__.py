"""dual-RDK：POMDP / Q 学習モデル（要件定義書 v3.0 の実装）。

docs/specs/dual_rdk_pomdp_q_learning_spec.md に対応する。
learning ステージ48試行のみをモデル化する。

ここで JAX の設定を2つ行う。どちらも最初の JAX 配列が作られる前に実行する
必要があるため、パッケージ初期化時に置いている。

1. 倍精度（FR-6.0）。信念は 360 点グリッド上で 48 回逐次更新され、受け入れ
   基準（AC-1, AC-5）が 1e-10 / 1e-8 の許容誤差を要求するため float32 では
   不足する。
2. ホストデバイス数。JAX は既定で CPU を 1 デバイスとして扱うため、NUTS の
   chain が逐次実行になる。論理コアを分割して chain を並列に回す。
   環境変数 DUALRDK_HOST_DEVICES で上書きできる。

サブモジュールはここで先読みしない。`python -m dualrdk.pomdp.data` のように
サブモジュールを直接実行したときに二重初期化の RuntimeWarning が出るため。
"""
import os

import numpyro

_DEFAULT_HOST_DEVICES = 4
HOST_DEVICE_COUNT = int(os.environ.get("DUALRDK_HOST_DEVICES", _DEFAULT_HOST_DEVICES))

# set_host_device_count は XLA_FLAGS を書き換えるため、jax のバックエンド初期化前に呼ぶ
numpyro.set_host_device_count(HOST_DEVICE_COUNT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
