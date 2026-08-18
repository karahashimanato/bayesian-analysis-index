# PyMC/ArviZコーディングパターン辞書

自分のベイズ分析プロジェクト群(12リポジトリ)の実コードを横断調査し、繰り返し書いているPyMC/ArviZのコードパターンをまとめたカンニングペーパー。`tools/`の他のエントリが概念・定義の辞書であるのに対し、こちらは実際に書くコードそのものの辞書。

各エントリのコード例はすべて実プロジェクトのnotebookから抜粋したもの(このドキュメント用に新規に書き起こしたコードではない)。3リポジトリ以上で確認できたパターンを本編、2リポジトリのみで確認できたパターンは「参考」として末尾に分けている。

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
