"""
tools/state-space-models.md に埋め込む可視化画像を生成するスクリプト。

  1. GaussianRandomWalk: 振幅が時間とともに緩やかに変化するデータに対し、
     固定パラメータモデルでは捉えられず、GaussianRandomWalkなら追従できることを示す
  2. process noise と observation noise の非識別性: ローカルレベルモデルで
     両者の事後分布に強い負の相関(ridge)が生じ、個々には推定しづらいことを示す

実行方法:
    source .venv/bin/activate
    python scripts/generate_state_space_models_plots.py

出力先: assets/state-space-models/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "state-space-models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_gaussian_random_walk_amplitude():
    """振幅が緩やかに変化する周期データに対し、固定振幅モデルでは
    変化を捉えられず、GaussianRandomWalkによる時変振幅なら追従できることを示す。"""

    rng = np.random.default_rng(21)
    T = 200
    K = 20
    block_len = T // K
    t = np.arange(T)
    omega = 2 * np.pi / 20

    true_amp = 3.0 + 2.0 * np.sin(2 * np.pi * t / 100)
    y = true_amp * np.sin(omega * t) + rng.normal(0, 0.5, size=T)
    block_idx = t // block_len
    true_amp_block = np.array([true_amp[block_idx == k].mean() for k in range(K)])

    sin_t, cos_t = np.sin(omega * t), np.cos(omega * t)

    with pm.Model():
        tau1 = pm.HalfNormal("tau1", 1.0)
        tau2 = pm.HalfNormal("tau2", 1.0)
        beta1 = pm.GaussianRandomWalk("beta1", sigma=tau1, init_dist=pm.Normal.dist(0, 2), steps=K - 1)
        beta2 = pm.GaussianRandomWalk("beta2", sigma=tau2, init_dist=pm.Normal.dist(0, 2), steps=K - 1)
        pm.Deterministic("amplitude", pm.math.sqrt(beta1**2 + beta2**2))
        mu = beta1[block_idx] * sin_t + beta2[block_idx] * cos_t
        sigma_obs = pm.HalfNormal("sigma_obs", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)
        idata_rw = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    with pm.Model():
        beta1 = pm.Normal("beta1", 0.0, 3.0)
        beta2 = pm.Normal("beta2", 0.0, 3.0)
        pm.Deterministic("amplitude", pm.math.sqrt(beta1**2 + beta2**2))
        mu = beta1 * sin_t + beta2 * cos_t
        sigma_obs = pm.HalfNormal("sigma_obs", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)
        idata_static = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    amp_rw = idata_rw.posterior["amplitude"].values.reshape(-1, K).mean(axis=0)
    amp_static = float(idata_static.posterior["amplitude"].values.mean())

    rmse_rw = float(np.sqrt(np.mean((amp_rw - true_amp_block) ** 2)))
    rmse_static = float(np.sqrt(np.mean((amp_static - true_amp_block) ** 2)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, true_amp, color="gray", lw=1.5, label="真の振幅 A(t)")
    block_centers = block_len * (np.arange(K) + 0.5)
    ax.step(block_centers, amp_rw, where="mid", color=COLOR_OK, lw=2,
            label=f"GaussianRandomWalkモデル(RMSE={rmse_rw:.2f})")
    ax.axhline(amp_static, color=COLOR_DIVERGENT, lw=2, ls="--",
               label=f"固定振幅モデル(RMSE={rmse_static:.2f})")
    ax.set_xlabel("時刻 t")
    ax.set_ylabel("振幅")
    ax.set_title("GaussianRandomWalkによる時変振幅の追従")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_random_walk_amplitude.png")
    plt.close(fig)

    print(f"gaussian_random_walk_amplitude.png saved "
          f"(RMSE: RandomWalk={rmse_rw:.3f}, 固定振幅={rmse_static:.3f})")


def plot_process_obs_noise_nonidentifiability():
    """ローカルレベルモデル(GaussianRandomWalk + 観測ノイズ)で、
    process noiseとobservation noiseの事後分布に強い負の相関(ridge)が生じることを示す。"""

    rng = np.random.default_rng(2)
    T = 20
    true_sigma_process, true_sigma_obs = 1.0, 1.0
    x_true = np.cumsum(rng.normal(0, true_sigma_process, size=T))
    y = x_true + rng.normal(0, true_sigma_obs, size=T)

    with pm.Model():
        sigma_process = pm.HalfNormal("sigma_process", 2.0)
        sigma_obs = pm.HalfNormal("sigma_obs", 2.0)
        x = pm.GaussianRandomWalk("x", sigma=sigma_process, init_dist=pm.Normal.dist(0, 1), steps=T - 1)
        pm.Normal("y", mu=x, sigma=sigma_obs, observed=y)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.95, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    sp = idata.posterior["sigma_process"].values.flatten()
    so = idata.posterior["sigma_obs"].values.flatten()
    corr = float(np.corrcoef(sp, so)[0, 1])
    total = sp + so
    ratio_marginal_spread = (sp.std() + so.std()) / total.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(sp, so, s=4, alpha=0.15, color=COLOR_ALT)
    axes[0].axvline(true_sigma_process, color="gray", lw=0.8, ls=":")
    axes[0].axhline(true_sigma_obs, color="gray", lw=0.8, ls=":")
    axes[0].scatter([true_sigma_process], [true_sigma_obs], color=COLOR_DIVERGENT, marker="x", s=80,
                     label=f"真の値({true_sigma_process}, {true_sigma_obs})", zorder=3)
    axes[0].set_xlabel(r"$\sigma_{process}$")
    axes[0].set_ylabel(r"$\sigma_{obs}$")
    axes[0].set_title(f"事後サンプルの相関: r={corr:.2f}")
    axes[0].legend(fontsize=9)

    axes[1].hist(sp, bins=40, alpha=0.5, color=COLOR_OK, density=True, label=r"$\sigma_{process}$の周辺分布")
    axes[1].hist(so, bins=40, alpha=0.5, color=COLOR_ALT, density=True, label=r"$\sigma_{obs}$の周辺分布")
    axes[1].hist(total, bins=40, histtype="step", color="black", lw=1.5, density=True,
                 label=r"和 $\sigma_{process}+\sigma_{obs}$の分布")
    axes[1].set_xlabel("値")
    axes[1].set_ylabel("密度")
    axes[1].set_title("個々の周辺分布は広いが、和はより狭い")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("process noiseとobservation noiseの非識別性(ridge構造)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "process_obs_noise_nonidentifiability.png")
    plt.close(fig)

    print(f"process_obs_noise_nonidentifiability.png saved "
          f"(corr={corr:.3f}, sd(sigma_process)={sp.std():.3f}, sd(sigma_obs)={so.std():.3f}, "
          f"sd(sum)={total.std():.3f})")


def plot_changepoint_tau_recovery():
    """変化点モデルを実際にPyMCでサンプリングし、離散パラメータtauの事後分布が
    真の変化点付近に集まること、および区間ごとの平均レベルmu1/mu2が正しく
    復元されることを示す。"""

    rng = np.random.default_rng(11)
    T = 80
    true_tau = 40
    mu1_true, mu2_true = 2.0, 3.2
    sigma_true = 2.0

    t = np.arange(T)
    mu_t_true = np.where(t < true_tau, mu1_true, mu2_true)
    y = mu_t_true + rng.normal(0, sigma_true, T)

    with pm.Model():
        tau = pm.DiscreteUniform("tau", lower=1, upper=T - 2)
        mu1 = pm.Normal("mu1", 0, 10)
        mu2 = pm.Normal("mu2", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        mu_t = pm.math.switch(t < tau, mu1, mu2)
        pm.Normal("y", mu=mu_t, sigma=sigma, observed=y)
        idata = pm.sample(2000, tune=2000, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    tau_draws = idata.posterior["tau"].values.flatten()
    tau_mode = int(np.bincount(tau_draws).argmax())
    mu1_est = float(idata.posterior["mu1"].values.mean())
    mu2_est = float(idata.posterior["mu2"].values.mean())
    ess_tau = float(az.ess(idata, var_names=["tau"])["tau"].values)
    ess_mu1 = float(az.ess(idata, var_names=["mu1"])["mu1"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(t, y, s=10, alpha=0.5, color="black", label="観測データ")
    axes[0].step(t, mu_t_true, where="post", color=COLOR_DIVERGENT, lw=2, label="真の水準")
    mu_t_est = np.where(t < tau_mode, mu1_est, mu2_est)
    axes[0].step(t, mu_t_est, where="post", color=COLOR_OK, lw=2, ls="--",
                 label=f"推定水準(tau={tau_mode})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y")
    axes[0].set_title("データと推定された変化点前後の水準")
    axes[0].legend(fontsize=8.5)

    axes[1].hist(tau_draws, bins=np.arange(1, T - 1) - 0.5, color=COLOR_ALT, alpha=0.7)
    axes[1].axvline(true_tau, color=COLOR_DIVERGENT, lw=2, ls="--", label=f"真のtau={true_tau}")
    axes[1].set_xlabel("tau(変化点の時点)")
    axes[1].set_ylabel("頻度")
    axes[1].set_title(f"tauの事後分布(ESS={ess_tau:.0f})\n連続パラメータmu1のESS={ess_mu1:.0f}")
    axes[1].legend(fontsize=9)

    fig.suptitle("変化点モデル: tauの事後分布と区間ごとの水準の復元", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "changepoint_tau_recovery.png")
    plt.close(fig)

    print(f"changepoint_tau_recovery.png saved "
          f"(true_tau={true_tau}, tau_mode={tau_mode}, mu1={mu1_est:.2f}(true {mu1_true}), "
          f"mu2={mu2_est:.2f}(true {mu2_true}), ess_tau={ess_tau:.0f}, ess_mu1={ess_mu1:.0f})")


def plot_markov_switching_transition_recovery():
    """2レジームMarkov-Switchingモデルをforward algorithm周辺化尤度で
    実際にPyMCでサンプリングし、遷移確率(レジームの持続性)とレジームごとの
    平均を復元できることを示す。順序制約(mu1=mu0+gap, gap>0)でラベル
    スイッチングを回避する。"""

    rng = np.random.default_rng(5)
    T = 300
    p_stay0_true, p_stay1_true = 0.95, 0.90
    mu0_true, mu1_true = 0.0, 3.0
    sigma_true = 1.0

    regime = np.zeros(T, dtype=int)
    for i in range(1, T):
        if regime[i - 1] == 0:
            regime[i] = 0 if rng.uniform() < p_stay0_true else 1
        else:
            regime[i] = 1 if rng.uniform() < p_stay1_true else 0
    mu_t_true = np.where(regime == 0, mu0_true, mu1_true)
    y = mu_t_true + rng.normal(0, sigma_true, T)

    def forward_loglik(y_obs, p_stay0, p_stay1, mu0, mu1, sigma):
        e0 = pt.exp(-0.5 * ((y_obs - mu0) / sigma) ** 2) / (sigma * pt.sqrt(2 * np.pi))
        e1 = pt.exp(-0.5 * ((y_obs - mu1) / sigma) ** 2) / (sigma * pt.sqrt(2 * np.pi))

        pi0 = (1 - p_stay1) / (2 - p_stay0 - p_stay1)
        pi1 = 1 - pi0
        alpha0_init = pi0 * e0[0]
        alpha1_init = pi1 * e1[0]
        c0 = alpha0_init + alpha1_init
        alpha0_init, alpha1_init = alpha0_init / c0, alpha1_init / c0

        def step(e0_t, e1_t, alpha0_prev, alpha1_prev, p_stay0_ns, p_stay1_ns):
            pred0 = alpha0_prev * p_stay0_ns + alpha1_prev * (1 - p_stay1_ns)
            pred1 = alpha1_prev * p_stay1_ns + alpha0_prev * (1 - p_stay0_ns)
            a0, a1 = pred0 * e0_t, pred1 * e1_t
            c_t = a0 + a1
            return a0 / c_t, a1 / c_t, pt.log(c_t)

        (_, _, loglik_seq), _ = pytensor.scan(
            fn=step, sequences=[e0[1:], e1[1:]],
            outputs_info=[alpha0_init, alpha1_init, None],
            non_sequences=[p_stay0, p_stay1], strict=True,
        )
        return pt.log(c0) + pt.sum(loglik_seq)

    with pm.Model():
        p_stay0 = pm.Beta("p_stay0", 2, 2)
        p_stay1 = pm.Beta("p_stay1", 2, 2)
        mu0 = pm.Normal("mu0", 0, 5)
        gap = pm.HalfNormal("gap", 5)  # 順序制約: mu1=mu0+gap>mu0 でラベルスイッチングを回避
        mu1 = pm.Deterministic("mu1", mu0 + gap)
        sigma = pm.HalfNormal("sigma", 3)

        ll = forward_loglik(pt.as_tensor_variable(y), p_stay0, p_stay1, mu0, mu1, sigma)
        pm.Potential("loglik", ll)

        idata = pm.sample(1500, tune=2000, chains=4, target_accept=0.95,
                           random_seed=3, progressbar=False,
                           compute_convergence_checks=False)

    n_div = int(idata.sample_stats["diverging"].sum())
    p0_est = float(idata.posterior["p_stay0"].values.mean())
    p1_est = float(idata.posterior["p_stay1"].values.mean())
    mu0_est = float(idata.posterior["mu0"].values.mean())
    mu1_est = float(idata.posterior["mu1"].values.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    axes[0].fill_between(np.arange(T), y.min() - 1, y.max() + 1, where=(regime == 1),
                          color=COLOR_ALT, alpha=0.15, step="post", label="真のレジーム1")
    axes[0].plot(np.arange(T), y, color="black", lw=0.8, alpha=0.7)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y")
    axes[0].set_ylim(y.min() - 1, y.max() + 1)
    axes[0].set_title("観測データと真のレジーム(網掛け)")
    axes[0].legend(fontsize=8.5, loc="upper right")

    labels = ["p_stay0", "p_stay1", "mu0", "mu1"]
    true_vals = [p_stay0_true, p_stay1_true, mu0_true, mu1_true]
    est_vals = [p0_est, p1_est, mu0_est, mu1_est]
    x_pos = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x_pos - width / 2, true_vals, width=width, color="black", alpha=0.7, label="真値")
    axes[1].bar(x_pos + width / 2, est_vals, width=width, color=COLOR_OK, label="事後平均")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f"遷移確率・レジーム平均の復元\n(divergence={n_div})")
    axes[1].legend(fontsize=9)

    fig.suptitle("Markov-Switching Model: forward algorithmによる周辺化推定", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "markov_switching_transition_recovery.png")
    plt.close(fig)

    print(f"markov_switching_transition_recovery.png saved "
          f"(divergence={n_div}, p_stay0={p0_est:.3f}(true {p_stay0_true}), "
          f"p_stay1={p1_est:.3f}(true {p_stay1_true}), mu0={mu0_est:.3f}(true {mu0_true}), "
          f"mu1={mu1_est:.3f}(true {mu1_true}))")


def plot_sv_volatility_recovery():
    """Stochastic Volatility(SV)モデルを非中心化パラメータ化で実際に
    PyMCでサンプリングし、対数ボラティリティh_tの事後分布(平均+90%区間)が
    真の変動パターンをどれだけ復元できるかを示す。"""

    rng = np.random.default_rng(9)
    T = 100
    phi_true = 0.95
    sigma_eta_true = 0.25
    mu_h_true = -1.0

    h_true = np.zeros(T)
    h_true[0] = mu_h_true
    for t in range(1, T):
        h_true[t] = mu_h_true + phi_true * (h_true[t - 1] - mu_h_true) + rng.normal(0, sigma_eta_true)
    returns = rng.normal(0, np.exp(h_true / 2))

    with pm.Model():
        mu_h = pm.Normal("mu_h", -1.0, 2.0)
        phi_raw = pm.Beta("phi_raw", 20, 1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma_eta = pm.HalfNormal("sigma_eta", 0.5)

        h_raw = pm.Normal("h_raw", 0, 1, shape=T)  # 非中心化
        h_list = [mu_h + sigma_eta * h_raw[0] / pt.sqrt(1 - phi ** 2)]
        for t in range(1, T):
            h_list.append(mu_h + phi * (h_list[-1] - mu_h) + sigma_eta * h_raw[t])
        h_est_expr = pt.stack(h_list)
        pm.Deterministic("h", h_est_expr)

        pm.Normal("returns", mu=0, sigma=pt.exp(h_est_expr / 2), observed=returns)

        idata = pm.sample(600, tune=1500, chains=4, target_accept=0.95,
                           random_seed=4, progressbar=False,
                           compute_convergence_checks=False)

    n_div = int(idata.sample_stats["diverging"].sum())
    phi_est = float(idata.posterior["phi"].mean())
    h_draws = idata.posterior["h"].values.reshape(-1, T)
    h_mean = h_draws.mean(axis=0)
    h_lo, h_hi = np.percentile(h_draws, [5, 95], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(returns, color="black", lw=0.8)
    axes[0].set_ylabel("リターン")
    axes[0].set_title("観測データ(リターン): ボラティリティクラスタリング")

    axes[1].fill_between(np.arange(T), h_lo, h_hi, color=COLOR_OK, alpha=0.25, label="事後90%区間")
    axes[1].plot(h_mean, color=COLOR_OK, lw=1.8, label="事後平均")
    axes[1].plot(h_true, color=COLOR_DIVERGENT, lw=1.3, ls="--", label="真の対数ボラティリティ")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("対数ボラティリティ h_t")
    axes[1].set_title(f"対数ボラティリティの復元(divergence={n_div}, phi事後平均={phi_est:.3f}, 真値{phi_true})")
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Stochastic Volatilityモデル: 対数ボラティリティh_tの復元", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "sv_volatility_recovery.png")
    plt.close(fig)

    print(f"sv_volatility_recovery.png saved "
          f"(divergence={n_div}, phi推定={phi_est:.3f}(true {phi_true}))")


if __name__ == "__main__":
    plot_gaussian_random_walk_amplitude()
    plot_process_obs_noise_nonidentifiability()
    plot_changepoint_tau_recovery()
    plot_markov_switching_transition_recovery()
    plot_sv_volatility_recovery()
