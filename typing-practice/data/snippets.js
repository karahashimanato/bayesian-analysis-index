// このファイルは scripts/generate_typing_snippets.py が自動生成する。直接編集しない。
window.SNIPPETS_DATA = {
  "chapters": [
    {
      "id": "prior-predictive-check",
      "title": "事前分布設計・prior predictive check",
      "source": "techniques/prior-predictive-check.md",
      "problems": [
        {
          "type": "line",
          "text": "Gamma(alpha>1, ...)"
        },
        {
          "type": "line",
          "text": "Gamma(shape>1)"
        },
        {
          "type": "line",
          "text": "beta0~N(0,5)"
        },
        {
          "type": "line",
          "text": "beta0~N(0,0.3)"
        }
      ]
    },
    {
      "id": "observation-model",
      "title": "尤度・観測モデル選択",
      "source": "techniques/observation-model.md",
      "problems": [
        {
          "type": "line",
          "text": "pm.Potential"
        },
        {
          "type": "line",
          "text": "pytensor.scan"
        },
        {
          "type": "line",
          "text": "pm.Normal"
        },
        {
          "type": "line",
          "text": "y ~ Poisson(exp(f(x)))"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "log(count+1)"
        },
        {
          "type": "line",
          "text": "pm.gp.Marginal"
        },
        {
          "type": "line",
          "text": "y ~ Normal(f(x), sigma)"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        }
      ]
    },
    {
      "id": "reparameterization",
      "title": "パラメータ化・非識別性対策",
      "source": "techniques/reparameterization.md",
      "problems": [
        {
          "type": "line",
          "text": "I0 = incidence_obs[0] / gamma"
        },
        {
          "type": "line",
          "text": "mu_logit + sigma_arm * offset_raw"
        },
        {
          "type": "line",
          "text": "offset_raw"
        },
        {
          "type": "line",
          "text": "sigma_arm"
        },
        {
          "type": "line",
          "text": "pt.sort()"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        },
        {
          "type": "line",
          "text": "log(mean(fires))"
        },
        {
          "type": "line",
          "text": "m=20"
        },
        {
          "type": "line",
          "text": "az.compare"
        },
        {
          "type": "line",
          "text": "pm.GaussianRandomWalk"
        },
        {
          "type": "line",
          "text": "time_raw - mean(time_raw)"
        }
      ]
    },
    {
      "id": "diagnostics",
      "title": "診断・収束判定",
      "source": "techniques/diagnostics.md",
      "problems": [
        {
          "type": "line",
          "text": "adapt_step_size=False"
        },
        {
          "type": "line",
          "text": "target_accept"
        },
        {
          "type": "line",
          "text": "target_accept=0.99"
        },
        {
          "type": "line",
          "text": "n_steps_per_week"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        }
      ]
    },
    {
      "id": "model-evaluation",
      "title": "モデル評価・比較",
      "source": "techniques/model-evaluation.md",
      "problems": [
        {
          "type": "line",
          "text": "az.compare"
        },
        {
          "type": "line",
          "text": "elpd_diff"
        },
        {
          "type": "line",
          "text": "p_LOO"
        },
        {
          "type": "line",
          "text": "az.loo"
        },
        {
          "type": "line",
          "text": "log(GDP per capita)"
        },
        {
          "type": "line",
          "text": "under5_mortality"
        },
        {
          "type": "line",
          "text": "beta_gdp"
        },
        {
          "type": "line",
          "text": "gp.conditional"
        }
      ]
    },
    {
      "id": "data-pitfalls",
      "title": "データ・単位・前処理の落とし穴",
      "source": "techniques/data-pitfalls.md",
      "problems": [
        {
          "type": "line",
          "text": "is_coinbase"
        },
        {
          "type": "line",
          "text": "block_timestamp_month"
        },
        {
          "type": "line",
          "text": "propensity_score"
        },
        {
          "type": "line",
          "text": "beta_gdp"
        }
      ]
    },
    {
      "id": "implementation-hacks",
      "title": "実装上のハック",
      "source": "techniques/implementation-hacks.md",
      "problems": [
        {
          "type": "line",
          "text": "pytensor.scan"
        },
        {
          "type": "line",
          "text": "pm.Potential"
        },
        {
          "type": "line",
          "text": "pm.Deterministic(\"lik_i\", log_lik_elementwise)"
        },
        {
          "type": "line",
          "text": "np.asarray()"
        },
        {
          "type": "line",
          "text": "add_groups"
        },
        {
          "type": "line",
          "text": "xarray.DataTree"
        },
        {
          "type": "line",
          "text": "idata[\"log_likelihood\"] = ..."
        },
        {
          "type": "line",
          "text": "G(t_max) = 0"
        },
        {
          "type": "line",
          "text": "np.random.default_rng(42)"
        },
        {
          "type": "line",
          "text": "dry_run=True"
        },
        {
          "type": "line",
          "text": "maximum_bytes_billed"
        },
        {
          "type": "line",
          "text": "init_dist"
        },
        {
          "type": "line",
          "text": "pm.GaussianRandomWalk(\"x\", sigma=..., shape=n)"
        },
        {
          "type": "line",
          "text": "OverflowError: Python integer ... out of bounds for int8"
        },
        {
          "type": "line",
          "text": "pm.GaussianRandomWalk(\"x\", sigma=..., init_dist=pm.Normal.dist(0,1), steps=n-1)"
        },
        {
          "type": "line",
          "text": "pm.Gamma"
        },
        {
          "type": "line",
          "text": "(alpha,beta)"
        },
        {
          "type": "line",
          "text": "(mu,sigma)"
        },
        {
          "type": "line",
          "text": "alpha_conc"
        },
        {
          "type": "line",
          "text": "(mu, alpha)"
        },
        {
          "type": "line",
          "text": "beta = alpha_conc / mu"
        },
        {
          "type": "line",
          "text": "pm.Deterministic"
        },
        {
          "type": "line",
          "text": "pm.Gamma(alpha=alpha_conc, beta=beta)"
        },
        {
          "type": "line",
          "text": "dt = t[:, None] - t[None, :]"
        },
        {
          "type": "line",
          "text": "dt > 0"
        },
        {
          "type": "line",
          "text": "pt.switch"
        },
        {
          "type": "line",
          "text": "sigma_level"
        },
        {
          "type": "line",
          "text": "exposure_matrix"
        },
        {
          "type": "line",
          "text": "H_i = pt.dot(exposure_matrix, h0) * hazard_ratio"
        },
        {
          "type": "line",
          "text": "cores>1"
        },
        {
          "type": "line",
          "text": "cores=1"
        },
        {
          "type": "line",
          "text": "pm.sample(..., cores=4)"
        },
        {
          "type": "line",
          "text": "progressbar=True"
        },
        {
          "type": "line",
          "text": "pm.sample"
        },
        {
          "type": "line",
          "text": "progressbar=False"
        },
        {
          "type": "line",
          "text": "pm.gp.MarginalApprox"
        },
        {
          "type": "line",
          "text": "sample_prior_predictive"
        },
        {
          "type": "line",
          "text": "pm.gp.MarginalApprox.marginal_likelihood"
        },
        {
          "type": "line",
          "text": "y_obs"
        },
        {
          "type": "line",
          "text": "pm.sample_prior_predictive"
        },
        {
          "type": "line",
          "text": "pm.ICAR"
        },
        {
          "type": "line",
          "text": "random()"
        },
        {
          "type": "line",
          "text": "pm.sample_prior_predictive()"
        },
        {
          "type": "line",
          "text": "NotImplementedError: Cannot sample from ICAR prior"
        },
        {
          "type": "line",
          "text": "pm.model_to_graphviz"
        },
        {
          "type": "line",
          "text": "model_to_graphviz"
        },
        {
          "type": "line",
          "text": "scaling_factor"
        },
        {
          "type": "line",
          "text": "nb_data_funs.R"
        },
        {
          "type": "line",
          "text": "get_scaling_factor"
        },
        {
          "type": "line",
          "text": "sum(Psi * (Q_space @ Psi @ Q_time))"
        },
        {
          "type": "line",
          "text": "A^T=A"
        },
        {
          "type": "line",
          "text": "Q_space"
        },
        {
          "type": "line",
          "text": "Q_time"
        },
        {
          "type": "line",
          "text": "pm.find_MAP"
        },
        {
          "type": "line",
          "text": "scipy.optimize"
        },
        {
          "type": "line",
          "text": "pm.MvNormal"
        },
        {
          "type": "line",
          "text": "(y1,y2)"
        },
        {
          "type": "line",
          "text": "y2|y1"
        },
        {
          "type": "line",
          "text": "pm.Normal"
        },
        {
          "type": "line",
          "text": "beta_cross"
        },
        {
          "type": "line",
          "text": "observed="
        },
        {
          "type": "line",
          "text": "pm.Normal(\"constraint\", mu=expr, sigma=..., observed=0)"
        },
        {
          "type": "line",
          "text": "TypeError: Variables that depend on other nodes cannot be used for observed data"
        },
        {
          "type": "line",
          "text": "pm.Potential(\"sum_to_zero\", -0.5 * (expr**2) / eps**2)"
        },
        {
          "type": "line",
          "text": "torchvision::nms"
        },
        {
          "type": "line",
          "text": "pip install torchvision --index-url https://download.pytorch.org/whl/cpu"
        }
      ]
    },
    {
      "id": "evaluation-metrics",
      "title": "評価指標・推定量",
      "source": "tools/evaluation-metrics.md",
      "problems": [
        {
          "type": "line",
          "text": "pm.compute_log_likelihood"
        },
        {
          "type": "line",
          "text": "az.compare"
        },
        {
          "type": "line",
          "text": "elpd_loo"
        },
        {
          "type": "line",
          "text": "elpd_diff"
        },
        {
          "type": "line",
          "text": "p_LOO"
        },
        {
          "type": "line",
          "text": "regret_t = g(x*) - max(g(x_1),...,g(x_t))"
        },
        {
          "type": "line",
          "text": "g(x*)"
        }
      ]
    },
    {
      "id": "observation-models",
      "title": "観測モデル・尤度分布",
      "source": "tools/observation-models.md",
      "problems": [
        {
          "type": "line",
          "text": "f(x)"
        },
        {
          "type": "line",
          "text": "y ~ Poisson(exp(f(x)))"
        },
        {
          "type": "line",
          "text": "pm.gp.Marginal"
        },
        {
          "type": "line",
          "text": "Bernoulli(p): P(x=1)=p"
        },
        {
          "type": "line",
          "text": "Binomial(n,p): P(k)=C(n,k) p^k (1-p)^{n-k}"
        },
        {
          "type": "line",
          "text": "x_i ~ Binomial(n_i, p_i)"
        },
        {
          "type": "line",
          "text": "p ~ Dirichlet(concentration * base_measure)"
        },
        {
          "type": "line",
          "text": "x ~ Multinomial(N, p)"
        },
        {
          "type": "line",
          "text": "k(t)"
        },
        {
          "type": "line",
          "text": "h_{0,k(t)}"
        },
        {
          "type": "line",
          "text": "pm.Potential"
        },
        {
          "type": "line",
          "text": "pytensor.scan"
        }
      ]
    },
    {
      "id": "mcmc-diagnostics",
      "title": "MCMC診断指標",
      "source": "tools/mcmc-diagnostics.md",
      "problems": [
        {
          "type": "line",
          "text": "ess_bulk"
        },
        {
          "type": "line",
          "text": "ess_tail"
        },
        {
          "type": "line",
          "text": "p_LOO"
        },
        {
          "type": "line",
          "text": "target_accept"
        }
      ]
    },
    {
      "id": "inference-methods",
      "title": "推論エンジン・サンプリング手法",
      "source": "tools/inference-methods.md",
      "problems": [
        {
          "type": "line",
          "text": "pm.gp.Marginal"
        },
        {
          "type": "line",
          "text": "f ~ GP(0, k)"
        },
        {
          "type": "line",
          "text": "y ~ Normal(f(x), sigma)"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        },
        {
          "type": "line",
          "text": "pm.gp.MarginalApprox"
        },
        {
          "type": "line",
          "text": "pm.gp.util.kmeans_inducing_points"
        },
        {
          "type": "line",
          "text": "pm.Potential"
        },
        {
          "type": "line",
          "text": "y_obs"
        },
        {
          "type": "line",
          "text": "pm.sample_prior_predictive"
        }
      ]
    },
    {
      "id": "posterior-pathologies",
      "title": "事後分布の幾何学的病理",
      "source": "tools/posterior-pathologies.md",
      "problems": [
        {
          "type": "line",
          "text": "offset_i ~ Normal(0,1)"
        },
        {
          "type": "line",
          "text": "target_accept"
        },
        {
          "type": "line",
          "text": "offset_raw"
        },
        {
          "type": "line",
          "text": "pt.sort()"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        }
      ]
    },
    {
      "id": "statistical-biases",
      "title": "統計的バイアス・概念",
      "source": "tools/statistical-biases.md",
      "problems": [
        {
          "type": "line",
          "text": "G(t)"
        },
        {
          "type": "line",
          "text": "1/G(t)"
        },
        {
          "type": "line",
          "text": "G(t_max) = 0"
        }
      ]
    },
    {
      "id": "state-space-models",
      "title": "状態空間モデルの型",
      "source": "tools/state-space-models.md",
      "problems": [
        {
          "type": "line",
          "text": "n>127"
        },
        {
          "type": "line",
          "text": "init_dist"
        },
        {
          "type": "line",
          "text": "x_t = x_{t-1} + r(1 - exp(x_{t-1})/K) + process_noise"
        },
        {
          "type": "line",
          "text": "sigma_process"
        },
        {
          "type": "line",
          "text": "sigma_obs"
        },
        {
          "type": "line",
          "text": "P(S_t | S_{t-1})"
        },
        {
          "type": "line",
          "text": "r_t ~ Normal(0, exp(h_t/2))"
        }
      ]
    },
    {
      "id": "prior-distributions",
      "title": "事前分布の選び方",
      "source": "tools/prior-distributions.md",
      "problems": [
        {
          "type": "line",
          "text": "Gamma(alpha>1, beta)"
        },
        {
          "type": "line",
          "text": "Gamma(alpha>1,...)"
        },
        {
          "type": "line",
          "text": "pm.HalfNormal(lower=, upper=)"
        },
        {
          "type": "line",
          "text": "Gamma(alpha=5, beta=1)"
        }
      ]
    },
    {
      "id": "spatial-models",
      "title": "空間モデルの型",
      "source": "tools/spatial-models.md",
      "problems": [
        {
          "type": "line",
          "text": "pm.ICAR"
        },
        {
          "type": "line",
          "text": "random()"
        },
        {
          "type": "line",
          "text": "pm.sample_prior_predictive()"
        },
        {
          "type": "line",
          "text": "pm.model_to_graphviz"
        },
        {
          "type": "line",
          "text": "Q_space"
        },
        {
          "type": "line",
          "text": "Q_time"
        },
        {
          "type": "line",
          "text": "sum(Psi * (Q_space @ Psi @ Q_time))"
        },
        {
          "type": "line",
          "text": "pm.gp.Latent"
        },
        {
          "type": "line",
          "text": "pm.gp.HSGP"
        }
      ]
    },
    {
      "id": "missing-data",
      "title": "欠測データ処理の型",
      "source": "tools/missing-data.md",
      "problems": [
        {
          "type": "line",
          "text": "P(R|Y,X) = P(R|X)"
        },
        {
          "type": "line",
          "text": "P(R|Y,X) = P(R)"
        },
        {
          "type": "line",
          "text": "P(R|Y,X)"
        },
        {
          "type": "line",
          "text": "numpy.ma.MaskedArray"
        },
        {
          "type": "line",
          "text": "y_obs"
        },
        {
          "type": "line",
          "text": "y_mis"
        },
        {
          "type": "line",
          "text": "(y1,y2)"
        },
        {
          "type": "line",
          "text": "beta_cross"
        },
        {
          "type": "line",
          "text": "pm.MvNormal"
        }
      ]
    },
    {
      "id": "pymc-code-patterns",
      "title": "PyMC/ArviZコーディングパターン",
      "source": "tools/pymc-code-patterns.md",
      "problems": [
        {
          "type": "block",
          "text": "trace_reward_all = pm.sample(\n    draws=1000, tune=1500, chains=4, cores=4,\n    nuts_sampler=\"numpyro\", target_accept=0.9,\n    progressbar=False, random_seed=42,\n)"
        },
        {
          "type": "block",
          "text": "with pm.Model() as model_reward_all:\n    mu_logit2 = pm.Normal(\"mu_logit\", mu=np.log(overall_ctr_all/(1-overall_ctr_all)), sigma=1.5)\n    sigma_arm2 = pm.HalfNormal(\"sigma_arm\", sigma=1.0)\n    offset_raw2 = pm.Normal(\"offset_raw\", mu=0, sigma=1, shape=n_arms_all2)\n    logit_theta2 = pm.Deterministic(\"logit_theta\", mu_logit2 + sigma_arm2 * offset_raw2)\n    theta2 = pm.Deterministic(\"theta\", pm.math.sigmoid(logit_theta2))\n    obs2 = pm.Binomial(\"obs\", n=trials_all2, p=theta2, observed=successes_all2)"
        },
        {
          "type": "block",
          "text": "with shark_model:\n    prior = pm.sample_prior_predictive(draws=500, random_seed=42)\n\nobs_prior = prior.prior_predictive[\"obs\"].values.flatten()\nfig, axes = plt.subplots(1, 2, figsize=(12, 5))\naxes[0].hist(obs_prior, bins=range(0, 60), density=True, alpha=0.6, label=\"prior predictive\")\naxes[0].hist(attacks, bins=range(0, 60), density=True, alpha=0.6, label=\"実データ\")\nprint(f\"  99パーセンタイル: {np.percentile(obs_prior, 99):.1f}\")"
        },
        {
          "type": "block",
          "text": "print(f\"divergences: {trace_reward_all.sample_stats.diverging.sum().item()}\")\naz.summary(trace_reward_all, var_names=[\"mu_logit\", \"sigma_arm\"])"
        },
        {
          "type": "block",
          "text": "with model:\n    pm.compute_log_likelihood(idata_type1)\n\ncomparison = az.compare({\"Type I (非構造)\": idata_type1, \"Type IV (構造化)\": idata_type4})"
        },
        {
          "type": "block",
          "text": "with pm.Model() as model:\n    ell = pm.Gamma(\"ell\", alpha=5, beta=1)\n    eta = pm.HalfNormal(\"eta\", sigma=1.0)\n    sigma = pm.HalfNormal(\"sigma\", sigma=0.2)\n    cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)\n    gp = pm.gp.Marginal(cov_func=cov_func)\n    y_obs = gp.marginal_likelihood(\"y_obs\", X=X, y=y, sigma=sigma)"
        },
        {
          "type": "block",
          "text": "y_obs = gp.marginal_likelihood(\"y_obs\", X=X, y=y, sigma=sigma)\npm.model_to_graphviz(model)"
        },
        {
          "type": "block",
          "text": "rng = np.random.default_rng(42)\nidata = pm.sample(..., random_seed=42)"
        },
        {
          "type": "block",
          "text": "logit_p = (\n    beta0 + day_effect[day_idx_sub] + hour_effect[hour_idx_sub]\n    + beta1 * group_idx_sub + beta2 * log_ads_std_sub + beta3 * group_idx_sub * log_ads_std_sub\n)"
        },
        {
          "type": "block",
          "text": "n_div = int(idata.sample_stats[\"diverging\"].sum())\nrhat_ds = az.rhat(idata, var_names=[\"alpha\", \"beta_gdp\", \"beta_urban\", \"sigma\"])\nrhat_max = max(float(rhat_ds[v]) for v in rhat_ds.data_vars)\nif n_div > 0 or rhat_max > 1.01:\n    print(f\"  [警告] divergences={n_div}, max_rhat={rhat_max:.3f}\")"
        },
        {
          "type": "block",
          "text": "with pm.Model() as bb_model:\n    mu = pm.Beta(\"mu\", alpha=4, beta=12)\n    kappa = pm.Exponential(\"kappa\", lam=0.01)\n    alpha = pm.Deterministic(\"alpha\", mu * kappa)\n    beta = pm.Deterministic(\"beta\", (1 - mu) * kappa)\n    p = pm.Beta(\"p\", alpha=alpha, beta=beta, shape=n_players)"
        },
        {
          "type": "block",
          "text": "def leapfrog(x0, p0, eps, n_steps):\n    x, p = x0, p0\n    xs, ps = [x], [p]\n    for _ in range(n_steps):\n        p = p - 0.5 * eps * grad_U(x)\n        x = x + eps * p\n        p = p - 0.5 * eps * grad_U(x)\n        xs.append(x)\n        ps.append(p)\n    return np.array(xs), np.array(ps)"
        },
        {
          "type": "block",
          "text": "def euler_step(S_prev, I_prev, beta, gamma, N, dt):\n    dS = -beta * S_prev * I_prev / N * dt\n    dI = (beta * S_prev * I_prev / N - gamma * I_prev) * dt\n    S_new = S_prev + dS\n    I_new = pt.maximum(I_prev + dI, 0.0)\n    return S_new, I_new\n\n(S_path, I_path), _ = pytensor.scan(\n    fn=euler_step,\n    outputs_info=[S0, I0],\n    non_sequences=[beta, gamma, N_pop, dt],\n    n_steps=total_steps,\n)"
        },
        {
          "type": "block",
          "text": "def forward_step(x_t, Gamma_prev, P, mu, sigma):\n    Gamma_pred = pt.dot(Gamma_prev, P)\n    log_emission = -0.5 * pt.log(2 * np.pi * sigma ** 2) - 0.5 * ((x_t - mu) / sigma) ** 2\n    emission = pt.exp(log_emission) + 1e-12\n    Gamma_next = Gamma_pred * emission\n    return Gamma_next / pt.sum(Gamma_next)\n\nprobabilities, _ = pytensor.scan(\n    fn=forward_step, sequences=X, outputs_info=init_prob, non_sequences=[P, mu, sigma])\n\npm.Potential('marginal_likelihood', pt.sum(pt.log(pt.sum(probabilities, axis=1))))"
        },
        {
          "type": "block",
          "text": "# Q = D - W (ICAR精度行列、特異行列)\nQ_laplacian = np.diag(degree) - W\nscaling_factor = np.exp(np.mean(np.log(np.diag(pinv(Q_laplacian)))))\n\ndef sample_icar_prior(Q, n_samples, rng, zero_sum_stdev=0.001):\n    \"\"\"pm.ICAR(sigma=1)の事前分布から厳密にサンプリングする(閉形式の共分散を使用)\"\"\"\n    n = Q.shape[0]\n    precision = Q + np.ones((n, n)) / (zero_sum_stdev * n) ** 2\n    cov = inv(precision)\n    return rng.multivariate_normal(np.zeros(n), cov, size=n_samples)"
        },
        {
          "type": "block",
          "text": "log_hazard = pt.log(k) + (k - 1) * pt.log(t_obs) - k * pt.log(lam)\nlog_survival = -pt.pow(t_obs / lam, k)\nlog_lik = event_obs * log_hazard + log_survival\npm.Potential(\"loglike\", pt.sum(log_lik))"
        },
        {
          "type": "block",
          "text": "def kaplan_meier(t_obs, event_obs):\n    order = np.argsort(t_obs)\n    t_sorted = t_obs[order]; e_sorted = event_obs[order]\n    unique_times = np.unique(t_sorted[e_sorted == 1])\n    survival = 1.0; km_times = [0]; km_surv = [1.0]\n    for t in unique_times:\n        n_event = np.sum((t_sorted == t) & (e_sorted == 1))\n        n_at_risk = np.sum(t_sorted >= t)\n        survival *= (1 - n_event / n_at_risk)\n        km_times.append(t); km_surv.append(survival)\n    return np.array(km_times), np.array(km_surv)"
        },
        {
          "type": "block",
          "text": "h = h0[interval_idx] * m[value_bin]\nlog_lik = n_events * pt.log(h) - h * exposure_days\npm.Potential(\"loglike\", pt.sum(log_lik))"
        },
        {
          "type": "block",
          "text": "def build_exposure_matrix(interval_idx, exposure, breaks):\n    n_int = len(breaks) - 1\n    widths = np.diff(breaks)\n    mat = np.zeros((len(interval_idx), n_int))\n    for j in range(n_int):\n        mat[interval_idx > j, j] = widths[j]\n        mat[interval_idx == j, j] = exposure[interval_idx == j]\n    return mat\n# H_i = 通過した全区間の累積ハザード\nH_i = pt.dot(exposure_matrix, h0) * hazard_ratio\nlog_lik = event_obs * pt.log(h_i) - H_i"
        },
        {
          "type": "block",
          "text": "tau = pm.Uniform('tau', lower=0, upper=n-1)\ns = pm.HalfNormal('s', sigma=5)\nweight = pm.math.sigmoid((idx - tau) / s)\nmu = mu1 + (mu2 - mu1) * weight"
        },
        {
          "type": "block",
          "text": "def acquisition(kind, mu, sd, y_best, rng, cov=None, xi=0.01, kappa=2.0):\n    if kind == \"PI\":\n        z = (mu - y_best - xi) / sd\n        return norm.cdf(z)\n    if kind == \"EI\":\n        z = (mu - y_best - xi) / sd\n        val = (mu - y_best - xi) * norm.cdf(z) + sd * norm.pdf(z)\n        return np.where(sd < 1e-9, 0.0, val)\n    if kind == \"UCB\":\n        return mu + kappa * sd\n    if kind == \"GP-TS\":\n        jitter = 1e-8 * np.eye(cov.shape[0])\n        return rng.multivariate_normal(mu, cov + jitter)\n    raise ValueError(kind)"
        },
        {
          "type": "line",
          "text": "pm.sample()"
        },
        {
          "type": "line",
          "text": "nuts_sampler=\"numpyro\""
        },
        {
          "type": "line",
          "text": "progressbar=False"
        },
        {
          "type": "line",
          "text": "pm.sample("
        },
        {
          "type": "line",
          "text": "raw = Normal(0,1)"
        },
        {
          "type": "line",
          "text": "pm.Deterministic"
        },
        {
          "type": "line",
          "text": "_raw"
        },
        {
          "type": "line",
          "text": "_offset"
        },
        {
          "type": "line",
          "text": "pm.sample_prior_predictive"
        },
        {
          "type": "line",
          "text": "sample_prior_predictive"
        },
        {
          "type": "line",
          "text": "az.summary"
        },
        {
          "type": "line",
          "text": "az.summary("
        },
        {
          "type": "line",
          "text": ".sample_stats.diverging.sum()"
        },
        {
          "type": "line",
          "text": "pm.compute_log_likelihood"
        },
        {
          "type": "line",
          "text": "az.compare"
        },
        {
          "type": "line",
          "text": "eta**2 * pm.gp.cov.ExpQuad(...)"
        },
        {
          "type": "line",
          "text": "pm.gp.Marginal"
        },
        {
          "type": "line",
          "text": "pm.model_to_graphviz(model)"
        },
        {
          "type": "line",
          "text": "random_seed=42"
        },
        {
          "type": "line",
          "text": "np.random.default_rng(42)"
        },
        {
          "type": "line",
          "text": "np.random.seed(SEED)"
        },
        {
          "type": "line",
          "text": "pm.sample(..., random_seed=42)"
        },
        {
          "type": "line",
          "text": "np.random.default_rng"
        },
        {
          "type": "line",
          "text": "np.random.seed"
        },
        {
          "type": "line",
          "text": "random_seed="
        },
        {
          "type": "line",
          "text": "default_rng(42)"
        },
        {
          "type": "line",
          "text": "pm.sample_posterior_predictive(idata)"
        },
        {
          "type": "line",
          "text": "az.plot_ppc"
        },
        {
          "type": "line",
          "text": "dims="
        },
        {
          "type": "line",
          "text": "pm.Model(coords=...)"
        },
        {
          "type": "line",
          "text": "day_idx"
        },
        {
          "type": "line",
          "text": "pm.Model(coords=coords)"
        },
        {
          "type": "line",
          "text": "summary[\"r_hat\"].max()"
        },
        {
          "type": "line",
          "text": "pytensor.scan"
        },
        {
          "type": "line",
          "text": "pm.Potential"
        },
        {
          "type": "line",
          "text": "pm.ICAR(sigma=1, W=W, ...)"
        },
        {
          "type": "line",
          "text": "sigma_phi"
        },
        {
          "type": "line",
          "text": "norm.cdf"
        },
        {
          "type": "line",
          "text": "norm.pdf"
        },
        {
          "type": "line",
          "text": "lifelines.concordance_index"
        }
      ]
    }
  ]
};
