"""
techniques/reparameterization.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際にサンプリングし、再パラメータ化(reparameterization)が
非識別性・多峰性をどう解消するかを before/after で描画する:
  1. 三角関数の極形式(A, φ) vs 直交形式(β1, β2)
  2. Ridge型非識別性: (κ, β) vs 比M=κ/β

実行方法:
    source .venv/bin/activate
    python scripts/generate_reparameterization_plots.py

出力先: assets/reparameterization/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "reparameterization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_trig_reparameterization():
    """極形式(A, phi)は位相の周期性ゆえに事前分布のレンジ次第で
    見せかけの多峰性を生むが、直交形式(beta1, beta2)は単峰であることを示す。"""

    rng = np.random.default_rng(1)
    omega = 2 * np.pi / 7.0  # 周期7、既知として外生的に固定(periodogram相当)
    t = np.linspace(0, 40, 200)
    true_A, true_phi = 2.0, 0.6
    y = true_A * np.sin(omega * t + true_phi) + rng.normal(0, 0.3, size=t.size)

    # 極形式: phiの事前分布が複数周期分にまたがるため、周期のズレ分だけ
    # 「見た目上の」複数モードが生まれる(尤度自体は単峰)
    mode_starts = [true_phi - 2 * np.pi, true_phi, true_phi + 2 * np.pi, true_phi]
    with pm.Model():
        A = pm.HalfNormal("A", 5.0)
        phi = pm.Uniform("phi", -2 * np.pi, 4 * np.pi)
        mu = A * pm.math.sin(omega * t + phi)
        pm.Normal("obs", mu, 0.3, observed=y)
        idata_polar = pm.sample(
            1500, tune=1000, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, initvals=[{"phi": s, "A": true_A} for s in mode_starts],
            compute_convergence_checks=False,
        )

    # 直交形式: beta1 = A*cos(phi), beta2 = A*sin(phi) は線形回帰係数そのもの
    with pm.Model():
        beta1 = pm.Normal("beta1", 0.0, 5.0)
        beta2 = pm.Normal("beta2", 0.0, 5.0)
        mu = beta1 * np.sin(omega * t) + beta2 * np.cos(omega * t)
        pm.Normal("obs", mu, 0.3, observed=y)
        idata_cart = pm.sample(
            1500, tune=1000, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    import arviz as az
    rhat_phi = float(az.rhat(idata_polar, var_names=["phi"])["phi"].values)
    rhat_b1 = float(az.rhat(idata_cart, var_names=["beta1"])["beta1"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    n_chains = idata_polar.posterior.sizes["chain"]
    for c in range(n_chains):
        axes[0].scatter(
            idata_polar.posterior["phi"].values[c], idata_polar.posterior["A"].values[c],
            s=4, alpha=0.3, color=COLOR_CHAIN[c % len(COLOR_CHAIN)], label=f"chain {c}",
        )
    axes[0].set_xlabel("φ (phase)")
    axes[0].set_ylabel("A (amplitude)")
    axes[0].set_title(f"極形式 A·sin(ωt+φ)\nr_hat[φ]={rhat_phi:.2f}(周期ズレ分だけ見せかけの多峰性)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=3)

    for c in range(n_chains):
        axes[1].scatter(
            idata_cart.posterior["beta1"].values[c], idata_cart.posterior["beta2"].values[c],
            s=4, alpha=0.3, color=COLOR_CHAIN[c % len(COLOR_CHAIN)],
        )
    axes[1].set_xlabel("β1 (= A·cosφ)")
    axes[1].set_ylabel("β2 (= A·sinφ)")
    axes[1].set_title(f"直交形式 β1·sin(ωt)+β2·cos(ωt)\nr_hat[β1]={rhat_b1:.2f}(単峰)")

    fig.suptitle("三角関数パラメータの再パラメータ化: 極形式(左) vs 直交形式(右)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "trig_reparameterization.png")
    plt.close(fig)
    print(f"trig_reparameterization.png saved (r_hat: polar phi={rhat_phi:.2f}, cartesian beta1={rhat_b1:.2f})")


def plot_ridge_ratio_reparameterization():
    """比 M=kappa/beta が本質的な意味を持つ場合、(kappa, beta)を独立に
    サンプルするとridge状の非識別性でdivergenceが起きるが、Mを直接
    パラメータ化するとdivergenceが解消することを示す。"""

    true_M = 2.0

    # 元のパラメータ化: kappa, beta を独立に事前分布からサンプル
    # -> 尤度は比 kappa/beta にしか制約されないため、原点を通るray状のridgeができる
    # (target_accept=0.6はPyMCのデフォルト0.8より弱く、幾何学的な病理を意図的に露出させる設定)
    with pm.Model():
        kappa = pm.Gamma("kappa", alpha=1.0, beta=0.3)
        beta = pm.Gamma("beta", alpha=1.0, beta=0.3)
        pm.Normal("obs", kappa / beta, 0.01, observed=np.array([true_M]))
        idata_raw = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.6, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    # 再パラメータ化: 比M自体を直接サンプルし、betaはMと無関係な自由パラメータとして残す
    with pm.Model():
        M = pm.Gamma("M", alpha=4.0, beta=2.0)  # 平均2付近
        beta_free = pm.Gamma("beta_free", alpha=2.0, beta=0.3)
        pm.Deterministic("kappa_derived", M * beta_free)
        pm.Normal("obs", M, 0.01, observed=np.array([true_M]))
        idata_reparam = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.6, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    div_raw = idata_raw.sample_stats["diverging"].values.flatten()
    div_reparam = idata_reparam.sample_stats["diverging"].values.flatten()
    n_div_raw, n_div_reparam = int(div_raw.sum()), int(div_reparam.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    k = idata_raw.posterior["kappa"].values.flatten()
    b = idata_raw.posterior["beta"].values.flatten()
    axes[0].scatter(b[~div_raw], k[~div_raw], s=4, alpha=0.25, color=COLOR_OK)
    axes[0].scatter(b[div_raw], k[div_raw], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[0].set_xlabel("β")
    axes[0].set_ylabel("κ")
    axes[0].set_title(f"元のパラメータ化 (κ, β)\nκ/β=M にしか制約されずray状のridge\ndivergence: {n_div_raw}/{len(div_raw)}")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    m = idata_reparam.posterior["M"].values.flatten()
    bf = idata_reparam.posterior["beta_free"].values.flatten()
    axes[1].scatter(bf[~div_reparam], m[~div_reparam], s=4, alpha=0.25, color=COLOR_ALT)
    axes[1].scatter(bf[div_reparam], m[div_reparam], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[1].set_xlabel("β_free(自由パラメータ)")
    axes[1].set_ylabel("M = κ/β(直接推定)")
    axes[1].set_title(f"再パラメータ化 (M, β_free)\nMを直接観測するのでridgeが解消\ndivergence: {n_div_reparam}/{len(div_reparam)}")
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Ridge型非識別性: 比パラメータ化による解消", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "ridge_ratio_reparameterization.png")
    plt.close(fig)
    print(f"ridge_ratio_reparameterization.png saved (divergence: {n_div_raw}/{len(div_raw)} -> {n_div_reparam}/{len(div_reparam)})")


def plot_noncentered_ctr_funnel():
    """腕ごとのCTRを推定する階層ロジスティック回帰で、中心化パラメータ化が
    funnel状のdivergenceを起こし、非中心化パラメータ化で解消することを示す。"""

    rng = np.random.default_rng(3)
    n_arms = np.array([20, 20, 18, 15, 15, 10, 10, 8, 8, 5])  # 試行回数が少ない腕を含む
    true_mu_logit = -2.4  # 全体平均CTR ≈ 8.3%
    true_sigma_arm = 0.15
    J = len(n_arms)
    theta_true = true_mu_logit + true_sigma_arm * rng.normal(size=J)
    p_true = 1 / (1 + np.exp(-theta_true))
    y_obs = rng.binomial(n_arms, p_true)

    with pm.Model():
        mu_logit = pm.Normal("mu_logit", -2.0, 1.0)
        sigma_arm = pm.HalfNormal("sigma_arm", 1.0)
        theta_logit = pm.Normal("theta_logit", mu_logit, sigma_arm, shape=J)
        p = pm.Deterministic("p", pm.math.invlogit(theta_logit))
        pm.Binomial("y", n=n_arms, p=p, observed=y_obs)
        idata_centered = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.8, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    with pm.Model():
        mu_logit = pm.Normal("mu_logit", -2.0, 1.0)
        sigma_arm = pm.HalfNormal("sigma_arm", 1.0)
        offset_raw = pm.Normal("offset_raw", 0.0, 1.0, shape=J)
        theta_logit = pm.Deterministic("theta_logit", mu_logit + sigma_arm * offset_raw)
        p = pm.Deterministic("p", pm.math.invlogit(theta_logit))
        pm.Binomial("y", n=n_arms, p=p, observed=y_obs)
        idata_noncentered = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.8, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    div_c = idata_centered.sample_stats["diverging"].values.flatten()
    div_n = idata_noncentered.sample_stats["diverging"].values.flatten()
    n_div_c, n_div_n = int(div_c.sum()), int(div_n.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    sigma_c = idata_centered.posterior["sigma_arm"].values.flatten()
    theta0_c = idata_centered.posterior["theta_logit"].values[:, :, 0].flatten()
    axes[0].scatter(theta0_c[~div_c], sigma_c[~div_c], s=4, alpha=0.25, color=COLOR_OK)
    axes[0].scatter(theta0_c[div_c], sigma_c[div_c], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[0].set_title(f"中心化パラメータ化\nθ_1 ~ Normal(μ, σ)\ndivergence: {n_div_c}/{len(div_c)}")
    axes[0].set_xlabel("θ_1 (腕1のlogit-CTR)")
    axes[0].set_ylabel("σ_arm(腕間のばらつき)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    sigma_n = idata_noncentered.posterior["sigma_arm"].values.flatten()
    theta0_n = idata_noncentered.posterior["theta_logit"].values[:, :, 0].flatten()
    axes[1].scatter(theta0_n[~div_n], sigma_n[~div_n], s=4, alpha=0.25, color=COLOR_ALT)
    axes[1].scatter(theta0_n[div_n], sigma_n[div_n], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[1].set_title(f"非中心化パラメータ化\nθ_1 = μ + σ・offset_1\ndivergence: {n_div_n}/{len(div_n)}")
    axes[1].set_xlabel("θ_1 (腕1のlogit-CTR)")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("階層ロジスティック回帰(腕ごとのCTR)におけるfunnel回避", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "noncentered_ctr_funnel.png")
    plt.close(fig)
    print(f"noncentered_ctr_funnel.png saved (divergence: {n_div_c}/{len(div_c)} -> {n_div_n}/{len(div_n)})")


def plot_gp_hsgp_mean_basis_fix():
    """非ガウス尤度(Poisson)のHSGP回帰で、平均関数muを自由パラメータのまま
    多数の基底関数(m=20)を使うと深刻なdivergence/r_hat悪化を起こすが、
    muを固定定数にし基底関数数を絞る(m=6)とほぼ解消することを示す。
    mu単体の推定値はどちらも真値からズレる(ridge型の非識別性)が、
    実際にフィットされる合成量mu+fはどちらの設定でもほぼ同じ精度で
    真の対数レートを再現することも合わせて示す。"""

    rng = np.random.default_rng(6)
    N = 150
    X = np.linspace(0, 10, N)
    mu_true = np.log(30.0)
    true_f = 1.0 * np.sin(0.4 * X)
    log_rate_true = mu_true + true_f
    counts = rng.poisson(np.exp(log_rate_true))

    def fit(free_mu, m, eta_upper, seed):
        with pm.Model():
            if free_mu:
                mu = pm.Normal("mu", np.log(counts.mean() + 1), 2.0)
            else:
                mu = np.log(counts.mean() + 1)

            eta = pm.HalfNormal("eta", eta_upper)
            ls = 2.5  # 固定(GP自体の長さスケール非識別性は本デモの対象外)
            cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ls)
            gp = pm.gp.HSGP(m=[m], c=2.0, cov_func=cov)
            f = gp.prior("f", X=X[:, None])

            log_rate = mu + f
            pm.Poisson("y", mu=pm.math.exp(log_rate), observed=counts)

            idata = pm.sample(1000, tune=2500, chains=4, target_accept=0.95,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)

        n_div = int(idata.sample_stats["diverging"].sum())
        rhat_ds = az.rhat(idata)
        rhat_max = max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars)
        f_mean = idata.posterior["f"].values.reshape(-1, N).mean(axis=0)
        mu_mean = float(idata.posterior["mu"].values.mean()) if free_mu else float(mu)
        return n_div, rhat_max, f_mean, mu_mean

    n_div_broken, rhat_broken, f_broken, mu_broken = fit(True, 20, 5.0, 1)
    n_div_fixed, rhat_fixed, f_fixed, mu_fixed = fit(False, 6, 2.0, 1)

    lograte_broken = mu_broken + f_broken
    lograte_fixed = mu_fixed + f_fixed
    total_draws = 4000

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    labels = [f"muを自由推定\nm=20(壊れた設定)", f"muを固定定数\nm=6(修正後)"]
    x_pos = np.arange(2)
    divs = [n_div_broken, n_div_fixed]
    colors = [COLOR_DIVERGENT, COLOR_OK]
    axes[0].bar(x_pos, divs, color=colors, width=0.5)
    for i, (d, rhat) in enumerate(zip(divs, [rhat_broken, rhat_fixed])):
        axes[0].annotate(f"{d}/{total_draws}\nr_hat_max={rhat:.2f}", (i, d),
                          xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("divergence数")
    axes[0].set_title("divergence数とr_hat")

    axes[1].plot(X, log_rate_true, color="black", lw=2, label="真の対数レート")
    axes[1].plot(X, lograte_broken, color=COLOR_DIVERGENT, lw=1.5, ls="--",
                 label=f"壊れた設定のmu+f(mu={mu_broken:.2f})")
    axes[1].plot(X, lograte_fixed, color=COLOR_OK, lw=1.5, ls=":",
                 label=f"修正後のmu+f(mu={mu_fixed:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("log(rate)")
    axes[1].set_title(f"mu単体は真値(mu={mu_true:.2f})からズレるが\nmu+fはどちらもほぼ真の曲線を再現")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("HSGP: muを固定し基底関数数を絞ることでdivergenceを解消する\n(点推定mu+fは元々妥当だが、サンプリング自体の信頼性が違う)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "gp_hsgp_mean_basis_fix.png")
    plt.close(fig)

    print(f"gp_hsgp_mean_basis_fix.png saved "
          f"(divergence: {n_div_broken}/{total_draws} -> {n_div_fixed}/{total_draws}, "
          f"r_hat_max: {rhat_broken:.2f} -> {rhat_fixed:.2f}, "
          f"mu: broken={mu_broken:.2f} fixed={mu_fixed:.2f} true={mu_true:.2f})")


if __name__ == "__main__":
    plot_trig_reparameterization()
    plot_ridge_ratio_reparameterization()
    plot_noncentered_ctr_funnel()
    plot_gp_hsgp_mean_basis_fix()
