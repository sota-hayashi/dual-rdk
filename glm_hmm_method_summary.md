# GLM-HMM Method Summary

## 1. 概要

GLM-HMM（Generalized Linear Model Hidden Markov Model）は、  
標準的な Hidden Markov Model (HMM) を拡張したモデルである。

通常のHMMでは **状態ごとに固定の観測分布** を持つが、  
GLM-HMMでは **状態ごとにGLM（Generalized Linear Model）を持つ**。

つまり

- HMM：状態 → 観測分布
- GLM-HMM：状態 → **GLMモデル**

となる。

---

# 2. Standard HMM

潜在状態

$$
z_t \in \{1,\dots,K\}
$$

状態遷移

$$
P(z_t = k | z_{t-1} = j) = A_{jk}
$$

観測生成

$$
P(y_t | z_t = k) = f_k(y_t)
$$

ここでは **観測は状態のみに依存する**。

---

# 3. GLM-HMM

GLM-HMMでは観測は

- 状態
- 説明変数

の両方に依存する。

$$
P(y_t | z_t = k, x_t, w_k)
$$

説明変数

$$
x_t \in \mathbb{R}^M
$$

GLM重み

$$
w_k \in \mathbb{R}^M
$$

---

## 二値選択モデル

$$
P(y_t = 1 | z_t = k, x_t, w_k)
=
\sigma(w_k^\top x_t)
$$

シグモイド関数

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

したがって

$$
P(y_t | z_t=k,x_t,w_k)
=
\text{Bernoulli}(\sigma(w_k^\top x_t))
$$

つまり

**状態ごとに異なる意思決定モデルが存在する**

---

# 4. モデルパラメータ

状態遷移行列

$$
A \in \mathbb{R}^{K \times K}
$$

$$
A_{jk} = P(z_t = k | z_{t-1} = j)
$$

初期状態分布

$$
\pi_k = P(z_1 = k)
$$

GLM重み

$$
w_k \in \mathbb{R}^M
$$

全パラメータ

$$
\Theta = \{A, \pi, w_1,\dots,w_K\}
$$

---

# 5. 尤度

データ

$$
D = \{x_t,y_t\}_{t=1}^T
$$

尤度

$$
p(D|\Theta)
=
\sum_{z_1\dots z_T}
p(y,z|\Theta)
$$

同時分布

$$
p(y,z|\Theta)
=
p(z_1)p(y_1|z_1,x_1)
\prod_{t=2}^T
p(z_t|z_{t-1})p(y_t|z_t,x_t)
$$

---

# 6. EMアルゴリズム

GLM-HMMは **EMアルゴリズム** で推定する。

各反復

```
E-step
M-step
```

---

# 7. E-step

Forward-backwardアルゴリズムで  
状態の事後確率を求める。

状態事後確率

$$
\gamma_{t,k}
=
P(z_t=k | D,\Theta)
$$

遷移事後確率

$$
\xi_{t,j,k}
=
P(z_t=j,z_{t+1}=k|D,\Theta)
$$

---

## Forward

$$
\alpha_{t,k}
=
P(y_{1:t},z_t=k)
$$

---

## Backward

$$
\beta_{t,k}
=
P(y_{t+1:T}|z_t=k)
$$

---

## Posterior

$$
\gamma_{t,k}
=
\frac{\alpha_{t,k}\beta_{t,k}}
{\sum_j \alpha_{t,j}\beta_{t,j}}
$$

---

# 8. Expected Complete Log Likelihood

EMでは以下を最大化する。

$$
ECLL(\Theta)
=
\sum_k \gamma_{1,k}\log\pi_k
+
\sum_{t=1}^{T-1}
\sum_{j,k}
\xi_{t,j,k}\log A_{jk}
+
\sum_t
\sum_k
\gamma_{t,k}
\log P(y_t|z_t=k,x_t,w_k)
$$

---

# 9. M-step

## 初期状態分布

$$
\pi_k
=
\gamma_{1,k}
$$

---

## 遷移行列

$$
A_{jk}
=
\frac{
\sum_t \xi_{t,j,k}
}{
\sum_t \sum_k \xi_{t,j,k}
}
$$

---

## GLM重み

次の目的関数を最大化する。

$$
\sum_t \gamma_{t,k}
\log P(y_t|z_t=k,x_t,w_k)
$$

閉形式解はないため

**BFGSで最適化**

---

# 10. 事前分布

GLM重み

$$
w_k \sim \mathcal{N}(0,\sigma^2 I)
$$

遷移行列

$$
A_j \sim Dir(\alpha)
$$

---

# 11. 初期化

### Step1

単一GLMを推定

---

### Step2

GLM重みにノイズ

$$
w_k = w_{GLM} + \epsilon
$$

$$
\epsilon \sim N(0,0.2^2)
$$

---

### 遷移行列

$$
A = 0.95I + N(0,0.05I)
$$

行を正規化

---

# 12. 学習アルゴリズム

```
INPUT
    y_t
    x_t
    K

STEP1
    fit single GLM

STEP2
    initialize parameters

STEP3
    repeat EM

        E-step
            forward-backward
            compute gamma
            compute xi

        M-step
            update pi
            update A
            optimize w_k

STEP4
    repeat with multiple initializations

OUTPUT
    {pi, A, w_k}
```

---

# 13. 解釈

各状態は

**異なる意思決定戦略**

を表す。

- latent state → 行動モード
- GLM weights → 判断ルール
- transition → 戦略スイッチ