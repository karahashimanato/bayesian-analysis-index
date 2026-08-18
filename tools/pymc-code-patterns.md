# PyMC/ArviZコーディングパターン辞書

自分のベイズ分析プロジェクト群(12リポジトリ)の実コードを横断調査し、繰り返し書いているPyMC/ArviZのコードパターンをまとめたカンニングペーパー。`tools/`の他のエントリが概念・定義の辞書であるのに対し、こちらは実際に書くコードそのものの辞書。

各エントリのコード例はすべて実プロジェクトのnotebookから抜粋したもの(このドキュメント用に新規に書き起こしたコードではない)。3リポジトリ以上で確認できたパターンを本編、2リポジトリのみで確認できたパターンは「参考」として分けている。さらに、頻出はしないがゼロから書くと地味に大変で控えておきたいもの(Euler法・forward algorithm・ICAR/BYM2のスケーリング係数など)を「低頻度だが見返す価値のあるパターン」として末尾に置いている。

---

## 3リポジトリ以上で確認したパターン

### 1. 標準的な `pm.sample()` の呼び出し形

- **用途**: サンプリング実行時の引数の型。`draws=1000〜2000, tune=1000〜2000, chains=4, target_accept=0.9〜0.95, random_seed=42`が基本形で、`nuts_sampler="numpyro"`・`progressbar=False`を付けることが多い。
- **コード例**:
  ```python
  trace_reward_all = pm.sample(
      draws=1000, tune=1500, chains=4, cores=4,
      nuts_sampler="numpyro", target_accept=0.9,
      progressbar=False, random_seed=42,
  )
  ```
- **出典**: [Multi-Armed-Bandit/notebooks/hier_dm_shrinkage.ipynb](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/notebooks/hier_dm_shrinkage.ipynb)
- **登場プロジェクト**: 上記に加え bayesian-A-B-testing / bayesian-causal-inference / bayesian-epidemiological-models / bayesian-gaussian-process / bayesian-hazard-models / bayesian-modeling-lab / bayesian-spatial-models / bitcoin-utxo-survival(コーパス全体で`pm.sample(`が113件)

---

### 2. 非中心化(non-centered)パラメータ化の定型

- **用途**: `raw = Normal(0,1)` を引いてから `mu + sigma * raw` を `pm.Deterministic` で組み立てる、階層モデルの標準形。`tools/prior-distributions.md`の事前分布選択とセットで使う実装パターン。
- **コード例**:
  ```python
  with pm.Model() as model_reward_all:
      mu_logit2 = pm.Normal("mu_logit", mu=np.log(overall_ctr_all/(1-overall_ctr_all)), sigma=1.5)
      sigma_arm2 = pm.HalfNormal("sigma_arm", sigma=1.0)
      offset_raw2 = pm.Normal("offset_raw", mu=0, sigma=1, shape=n_arms_all2)
      logit_theta2 = pm.Deterministic("logit_theta", mu_logit2 + sigma_arm2 * offset_raw2)
      theta2 = pm.Deterministic("theta", pm.math.sigmoid(logit_theta2))
      obs2 = pm.Binomial("obs", n=trials_all2, p=theta2, observed=successes_all2)
  ```
- **出典**: [Multi-Armed-Bandit/notebooks/hier_dm_shrinkage.ipynb](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/notebooks/hier_dm_shrinkage.ipynb)
- **登場プロジェクト**: 上記に加え bayesian-A-B-testing / bayesian-hazard-models / bayesian-modeling-lab / bayesian-spatial-models / bitcoin-utxo-survival(`_raw`/`_offset`命名で検出した中では最も頻出)

---

### 3. Prior predictive checkの定型コード

- **用途**: サンプリング前に`pm.sample_prior_predictive`で生成した値を実データと重ねてヒストグラム表示し、パーセンタイルを確認する。`techniques/prior-predictive-check.md`の実装版。
- **コード例**:
  ```python
  with shark_model:
      prior = pm.sample_prior_predictive(draws=500, random_seed=42)

  obs_prior = prior.prior_predictive["obs"].values.flatten()
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  axes[0].hist(obs_prior, bins=range(0, 60), density=True, alpha=0.6, label="prior predictive")
  axes[0].hist(attacks, bins=range(0, 60), density=True, alpha=0.6, label="実データ")
  print(f"  99パーセンタイル: {np.percentile(obs_prior, 99):.1f}")
  ```
- **出典**: [bayesian-modeling-lab/notebooks/shark_attack.ipynb](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/notebooks/shark_attack.ipynb)
- **登場プロジェクト**: 上記に加え bayesian-A-B-testing / bayesian-causal-inference / bayesian-epidemiological-models / bayesian-gaussian-process / bayesian-hazard-models / bayesian-optimization / bitcoin-utxo-survival(`sample_prior_predictive`がコーパス全体で44件)

---

### 4. Divergence + `az.summary` の診断チェック

- **用途**: サンプリング直後に divergence 件数を出し、`az.summary`でr_hat/ESSを一望する。`tools/mcmc-diagnostics.md`の実装版。
- **コード例**:
  ```python
  print(f"divergences: {trace_reward_all.sample_stats.diverging.sum().item()}")
  az.summary(trace_reward_all, var_names=["mu_logit", "sigma_arm"])
  ```
- **出典**: [Multi-Armed-Bandit/notebooks/ope_full_data.ipynb](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/notebooks/ope_full_data.ipynb)
- **登場プロジェクト**: `az.summary(`はbayesian-deep-learningを除く11リポジトリ全てで計95件、`.sample_stats.diverging.sum()`は20件以上

---

### 5. LOO/WAICモデル比較(`pm.compute_log_likelihood` → `az.compare`)

- **用途**: 複数モデルの当てはまりを`az.compare`で横並び比較する前段として、各`idata`に対して`pm.compute_log_likelihood`を実行しておく。`tools/evaluation-metrics.md`のLOOの実装版。
- **コード例**:
  ```python
  with model:
      pm.compute_log_likelihood(idata_type1)

  comparison = az.compare({"Type I (非構造)": idata_type1, "Type IV (構造化)": idata_type4})
  ```
- **出典**: [bayesian-spatial-models/notebooks/spatiotemporal_bym.ipynb](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/notebooks/spatiotemporal_bym.ipynb)
- **登場プロジェクト**: bayesian-A-B-testing(`cross_random_effects.ipynb`・`spline.ipynb`) / bayesian-hazard-models(`validation.ipynb`)

---

### 6. GP周辺尤度(marginal likelihood)のセットアップ

- **用途**: RBFカーネルを`eta**2 * pm.gp.cov.ExpQuad(...)`で組み、`pm.gp.Marginal`で周辺尤度を取る定型。`tools/inference-methods.md`のGP推論の実装版。
- **コード例**:
  ```python
  with pm.Model() as model:
      ell = pm.Gamma("ell", alpha=5, beta=1)
      eta = pm.HalfNormal("eta", sigma=1.0)
      sigma = pm.HalfNormal("sigma", sigma=0.2)
      cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
      gp = pm.gp.Marginal(cov_func=cov_func)
      y_obs = gp.marginal_likelihood("y_obs", X=X, y=y, sigma=sigma)
  ```
- **出典**: [bayesian-gaussian-process/notebooks/gp_rbf_temperature.ipynb](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/notebooks/gp_rbf_temperature.ipynb)
- **登場プロジェクト**: bayesian-optimization(`04_partb_xgboost_hyperparameter_tuning.ipynb`のGP代理モデル) / bayesian-spatial-models(`lgcp_noto.ipynb`)

---

### 7. `pm.model_to_graphviz(model)` をモデル定義直後に置く

- **用途**: モデル定義のすぐ後(サンプリング前)にグラフを可視化し、依存関係を目視確認する習慣。
- **コード例**:
  ```python
  y_obs = gp.marginal_likelihood("y_obs", X=X, y=y, sigma=sigma)
  pm.model_to_graphviz(model)
  ```
- **出典**: [bayesian-gaussian-process/notebooks/gp_rbf_temperature.ipynb](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/notebooks/gp_rbf_temperature.ipynb)
- **登場プロジェクト**: bayesian-A-B-testing / bayesian-epidemiological-models / bayesian-hazard-models / bayesian-modeling-lab / bitcoin-utxo-survival(コーパス全体で35件)

---

### 8. シード固定の規約(`random_seed=42`)

- **用途**: `np.random.default_rng(42)` / `np.random.seed(SEED)` / `pm.sample(..., random_seed=42)`など、乱数シードを固定する箇所すべてで`42`をデフォルト値として使う規約。
- **コード例**:
  ```python
  rng = np.random.default_rng(42)
  idata = pm.sample(..., random_seed=42)
  ```
- **登場プロジェクト**: `np.random.default_rng`/`np.random.seed`が計57件、`random_seed=`が計165件。PyMCを使うほぼ全リポジトリ(8リポジトリ以上)で`default_rng(42)`形式を確認

---

### 9. Posterior predictive checkを観測値と重ねてプロット

- **用途**: `pm.sample_posterior_predictive(idata)`の結果を`az.plot_ppc`または手動ヒストグラムで観測データと重ねて確認する。
- **登場プロジェクト**: bayesian-gaussian-process(`gp_composite_co2.ipynb` / `gp_poisson_wildfires.ipynb` / `gp_rbf_temperature.ipynb` / `gp_sparse_nyc_temperature.ipynb`) / bayesian-modeling-lab(`lynx_sol.ipynb` / `nikkei_sol.ipynb` / `shark_attack.ipynb`) / bayesian-spatial-models

---

### 10. 階層モデルのインデックスは`dims=`/`coords`より生のnumpyインデックス配列が主流

- **用途**: グループ効果を`pm.Model(coords=...)`の`dims=`ラベルではなく、`day_idx`のような生のインデックス配列で参照する書き方が、非中心化パラメータ化(パターン2)と組み合わせて最も多く使われている。
- **コード例**:
  ```python
  logit_p = (
      beta0 + day_effect[day_idx_sub] + hour_effect[hour_idx_sub]
      + beta1 * group_idx_sub + beta2 * log_ads_std_sub + beta3 * group_idx_sub * log_ads_std_sub
  )
  ```
- **出典**: [bayesian-A-B-testing/notebooks/cross_random_effects.ipynb](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/notebooks/cross_random_effects.ipynb)
- **注記**: 明示的な`pm.Model(coords=coords)` + `dims=`によるxarrayラベリングは bayesian-causal-inference / bayesian-spatial-models の2リポジトリのみで確認。生インデックス配列の方がより広く使われている主流パターン。

---

## 参考(2リポジトリのみで確認、頻出と言い切るには材料不足)

### r_hat閾値による明示的なゲート処理

```python
n_div = int(idata.sample_stats["diverging"].sum())
rhat_ds = az.rhat(idata, var_names=["alpha", "beta_gdp", "beta_urban", "sigma"])
rhat_max = max(float(rhat_ds[v]) for v in rhat_ds.data_vars)
if n_div > 0 or rhat_max > 1.01:
    print(f"  [警告] divergences={n_div}, max_rhat={rhat_max:.3f}")
```

出典: [bayesian-missing-data/notebooks/01_mcar_mar_single_variable.ipynb](https://github.com/karahashimanato/bayesian-missing-data/blob/main/notebooks/01_mcar_mar_single_variable.ipynb)。ほぼ同型のコードが bayesian-causal-inference(`mean_reverting_extension.ipynb`・`secondary_add_to_cart.ipynb`)にも存在。`summary["r_hat"].max()`という別の書き方でbayesian-modeling-lab(`sunspot_sol.ipynb`)にも近い意図のコードあり。

### Beta(mu*kappa, (1-mu)*kappa)の明示的な実装

```python
with pm.Model() as bb_model:
    mu = pm.Beta("mu", alpha=4, beta=12)
    kappa = pm.Exponential("kappa", lam=0.01)
    alpha = pm.Deterministic("alpha", mu * kappa)
    beta = pm.Deterministic("beta", (1 - mu) * kappa)
    p = pm.Beta("p", alpha=alpha, beta=beta, shape=n_players)
```

出典: [bayesian-modeling-lab/notebooks/batting_average.ipynb](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/notebooks/batting_average.ipynb)。近い形が Multi-Armed-Bandit(`ts_replay_hierarchical_local.ipynb`)にも存在。`tools/prior-distributions.md`の[確率・割合エントリ](prior-distributions.md#確率割合01に制約された値)で紹介している再パラメータ化の実装版。

---

## 低頻度だが見返す価値のあるパターン(1〜2プロジェクトのみ)

頻出はしないが、ゼロから導出・実装するのが地味に大変で、コピペ元として持っておくと嬉しいもの。上の2区分とは異なり「何度も書いている」ことは要求せず、1プロジェクトにしか無くても価値があれば掲載する。

### Leapfrog積分器(HMC/NUTSの内部で使われる数値積分)

- **用途**: HMC/NUTSがステップサイズ超過で発散(divergence)する様子を、実際のleapfrog積分で再現する。`tools/mcmc-diagnostics.md`の[Divergenceエントリ](mcmc-diagnostics.md#divergence発散)の図はこのコードで生成した。
- **コード例**:
  ```python
  def leapfrog(x0, p0, eps, n_steps):
      x, p = x0, p0
      xs, ps = [x], [p]
      for _ in range(n_steps):
          p = p - 0.5 * eps * grad_U(x)
          x = x + eps * p
          p = p - 0.5 * eps * grad_U(x)
          xs.append(x)
          ps.append(p)
      return np.array(xs), np.array(ps)
  ```
- **出典**: このリポジトリ自身の [scripts/generate_mcmc_diagnostics_plots.py](../scripts/generate_mcmc_diagnostics_plots.py)(外部プロジェクトではなく、本indexのために自作したnumpy実装)

### Euler法によるSIR/SEIRのODE積分(`pytensor.scan`)

- **用途**: PyMCに組み込みのSIR/SEIRソルバーは無いため、Euler法の1ステップを`pytensor.scan`に載せてNUTSが軌道全体を通して微分できるようにする。
- **コード例**:
  ```python
  def euler_step(S_prev, I_prev, beta, gamma, N, dt):
      dS = -beta * S_prev * I_prev / N * dt
      dI = (beta * S_prev * I_prev / N - gamma * I_prev) * dt
      S_new = S_prev + dS
      I_new = pt.maximum(I_prev + dI, 0.0)
      return S_new, I_new

  (S_path, I_path), _ = pytensor.scan(
      fn=euler_step,
      outputs_info=[S0, I0],
      non_sequences=[beta, gamma, N_pop, dt],
      n_steps=total_steps,
  )
  ```
- **出典**: [bayesian-epidemiological-models/notebooks/sir.ipynb](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/notebooks/sir.ipynb)(4コンパートメントに拡張した同型のコードが`seir.ipynb`にも、`sis.ipynb`/`sirs.ipynb`にも存在)

### HMM(2状態Markov-switching)のforward algorithm

- **用途**: 状態遷移確率`P`と観測尤度から、`pm.Potential`経由で周辺尤度を計算する再帰フィルタリング。正規化のタイミングや対数スケールの扱いを間違えやすい教科書アルゴリズム。
- **コード例**:
  ```python
  def forward_step(x_t, Gamma_prev, P, mu, sigma):
      Gamma_pred = pt.dot(Gamma_prev, P)
      log_emission = -0.5 * pt.log(2 * np.pi * sigma ** 2) - 0.5 * ((x_t - mu) / sigma) ** 2
      emission = pt.exp(log_emission) + 1e-12
      Gamma_next = Gamma_pred * emission
      return Gamma_next / pt.sum(Gamma_next)

  probabilities, _ = pytensor.scan(
      fn=forward_step, sequences=X, outputs_info=init_prob, non_sequences=[P, mu, sigma])

  pm.Potential('marginal_likelihood', pt.sum(pt.log(pt.sum(probabilities, axis=1))))
  ```
- **出典**: [bayesian-modeling-lab/notebooks/index_sol.ipynb](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/notebooks/index_sol.ipynb)

### ICAR/BYM2の隣接構造・スケーリング係数・厳密事前サンプリング

- **用途**: モデル本体は`pm.ICAR(sigma=1, W=W, ...)`という組み込みを使うが、その前段の隣接行列構築と、BYM2で`sigma_phi`を非構造化ノイズと同じスケールに揃えるためのスケーリング係数(グラフラプラシアンの疑似逆行列の対角の対数平均、Riebler et al. 2016)は完全に手書き。論文1本分の導出が1行の式に圧縮されている。
- **コード例**:
  ```python
  # Q = D - W (ICAR精度行列、特異行列)
  Q_laplacian = np.diag(degree) - W
  scaling_factor = np.exp(np.mean(np.log(np.diag(pinv(Q_laplacian)))))

  def sample_icar_prior(Q, n_samples, rng, zero_sum_stdev=0.001):
      """pm.ICAR(sigma=1)の事前分布から厳密にサンプリングする(閉形式の共分散を使用)"""
      n = Q.shape[0]
      precision = Q + np.ones((n, n)) / (zero_sum_stdev * n) ** 2
      cov = inv(precision)
      return rng.multivariate_normal(np.zeros(n), cov, size=n_samples)
  ```
- **出典**: [bayesian-spatial-models/notebooks/bym_evolution.ipynb](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/notebooks/bym_evolution.ipynb)(`spatiotemporal_bym.ipynb`にも同型あり)

### Weibullハザード対数尤度 + 手書きKaplan-Meier推定量

- **用途**: 打ち切り(censoring)ありの生存時間データで、イベント発生分のhazard項と打ち切り分のsurvival項を混ぜた対数尤度。打ち切りの扱いを間違えやすい。
- **コード例**:
  ```python
  log_hazard = pt.log(k) + (k - 1) * pt.log(t_obs) - k * pt.log(lam)
  log_survival = -pt.pow(t_obs / lam, k)
  log_lik = event_obs * log_hazard + log_survival
  pm.Potential("loglike", pt.sum(log_lik))
  ```
  ```python
  def kaplan_meier(t_obs, event_obs):
      order = np.argsort(t_obs)
      t_sorted = t_obs[order]; e_sorted = event_obs[order]
      unique_times = np.unique(t_sorted[e_sorted == 1])
      survival = 1.0; km_times = [0]; km_surv = [1.0]
      for t in unique_times:
          n_event = np.sum((t_sorted == t) & (e_sorted == 1))
          n_at_risk = np.sum(t_sorted >= t)
          survival *= (1 - n_event / n_at_risk)
          km_times.append(t); km_surv.append(survival)
      return np.array(km_times), np.array(km_surv)
  ```
- **出典**: [bayesian-hazard-models/notebooks/weibull_hazard.ipynb](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/notebooks/weibull_hazard.ipynb)

### Piecewise Exponentialハザード尤度(2バリエーション)

- **用途**: 時間区間ごとに一定のハザードを仮定するモデル。単純な区間ビニング版と、対象が複数区間をまたいで生存した場合の累積ハザードを正しく積み上げる版の2種類。後者は`bayesian-hazard-models`側のコードに`#[修正]`という自己バグ修正コメントが残っており、間違えやすい箇所であることの実例になっている。
- **コード例(単純ビニング版)**:
  ```python
  h = h0[interval_idx] * m[value_bin]
  log_lik = n_events * pt.log(h) - h * exposure_days
  pm.Potential("loglike", pt.sum(log_lik))
  ```
- **コード例(複数区間をまたぐ累積ハザード版)**:
  ```python
  def build_exposure_matrix(interval_idx, exposure, breaks):
      n_int = len(breaks) - 1
      widths = np.diff(breaks)
      mat = np.zeros((len(interval_idx), n_int))
      for j in range(n_int):
          mat[interval_idx > j, j] = widths[j]
          mat[interval_idx == j, j] = exposure[interval_idx == j]
      return mat
  # H_i = 通過した全区間の累積ハザード
  H_i = pt.dot(exposure_matrix, h0) * hazard_ratio
  log_lik = event_obs * pt.log(h_i) - H_i
  ```
- **出典**: [bitcoin-utxo-survival/notebooks/stage_a_hazard_model.ipynb](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/notebooks/stage_a_hazard_model.ipynb)(単純ビニング版) / [bayesian-hazard-models/notebooks/cox_hazard.ipynb](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/notebooks/cox_hazard.ipynb)(累積ハザード版)

### 変化点のシグモイド緩和(実装版)

- **用途**: `tools/prior-distributions.md`の[変化点位置エントリ](prior-distributions.md#変化点位置離散-vs-連続の時点パラメータ)・`tools/state-space-models.md`で概念として触れているシグモイド緩和の、実際に動くコード。
- **コード例**:
  ```python
  tau = pm.Uniform('tau', lower=0, upper=n-1)
  s = pm.HalfNormal('s', sigma=5)
  weight = pm.math.sigmoid((idx - tau) / s)
  mu = mu1 + (mu2 - mu1) * weight
  ```
- **出典**: [bayesian-modeling-lab/notebooks/nile_sol.ipynb](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/notebooks/nile_sol.ipynb)

### 獲得関数(PI/EI/UCB/GP-TS)のスクラッチ実装

- **用途**: `tools/acquisition-functions.md`で定義だけ紹介しているPI/EI/UCB/GP-TSの、ライブラリ非依存の実装。EIの`norm.cdf`/`norm.pdf`の組み合わせ方と`sd`がほぼ0のときのエッジケース処理が実務上のポイント。
- **コード例**:
  ```python
  def acquisition(kind, mu, sd, y_best, rng, cov=None, xi=0.01, kappa=2.0):
      if kind == "PI":
          z = (mu - y_best - xi) / sd
          return norm.cdf(z)
      if kind == "EI":
          z = (mu - y_best - xi) / sd
          val = (mu - y_best - xi) * norm.cdf(z) + sd * norm.pdf(z)
          return np.where(sd < 1e-9, 0.0, val)
      if kind == "UCB":
          return mu + kappa * sd
      if kind == "GP-TS":
          jitter = 1e-8 * np.eye(cov.shape[0])
          return rng.multivariate_normal(mu, cov + jitter)
      raise ValueError(kind)
  ```
- **出典**: [bayesian-optimization/notebooks/01_1d_benchmark_acquisition_functions.ipynb](https://github.com/karahashimanato/bayesian-optimization/blob/main/notebooks/01_1d_benchmark_acquisition_functions.ipynb)

### 調査したが見つからなかったもの

以下は候補として探したが、12プロジェクト中に実装が存在しなかった(ライブラリ呼び出しのみ、または該当なし)。将来自分で書いたら追記する。

- **Hawkes過程の強度関数・尤度**: 全プロジェクトのnotebookを`hawkes`でgrepしてヒットなし。`tools/observation-models.md`では概念として触れているが、実装は未着手。
- **IPCW(逆確率打ち切り重み付け)**: `bayesian-hazard-models`・`bitcoin-utxo-survival`に実装なし。
- **カスタムpytensor `Op`**: `pytensor.scan`の利用例はあるが、`Op`をサブクラス化した例は無し。
- **手書きC-index**: `bayesian-hazard-models/notebooks/validation.ipynb`で計算はしているが`lifelines.concordance_index`をそのまま呼んでいるだけで、スクラッチ実装ではない。
