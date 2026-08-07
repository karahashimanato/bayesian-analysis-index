"""
tools/observation-models.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際に階層ベイズモデルをサンプリングし、
  1. Beta-Binomial: 観測回数の少ない個体ほど事後推定が全体平均へ強く縮小する(部分プーリング)
  2. Gamma-Poisson: overdispersedなカウントデータに対し、Poisson固定分散モデルは
     予測区間を過小評価し、Gamma-Poisson階層モデルは正しく捉える
ことを描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_observation_models_plots.py

出力先: assets/observation-models/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "observation-models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_beta_binomial_shrinkage():
    """打率のような個体ごとの成功率を階層Beta-Binomialモデルで推定し、
    観測回数が少ない個体ほど事後推定が全体平均へ強く縮小することを示す。"""

    rng = np.random.default_rng(5)
    J = 18
    n_ab = rng.choice([10, 15, 25, 40, 80, 150, 300, 500], size=J)
    true_mu, true_kappa = 0.27, 80.0
    true_p = rng.beta(true_mu * true_kappa, (1 - true_mu) * true_kappa, size=J)
    x_obs = rng.binomial(n_ab, true_p)
    obs_rate = x_obs / n_ab

    with pm.Model():
        mu = pm.Beta("mu", 2, 5)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=0.02)
        alpha = pm.Deterministic("alpha", mu * kappa)
        beta = pm.Deterministic("beta", (1 - mu) * kappa)
        p = pm.Beta("p", alpha=alpha, beta=beta, shape=J)
        pm.Binomial("x", n=n_ab, p=p, observed=x_obs)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    p_post_mean = idata.posterior["p"].values.mean(axis=(0, 1))
    mu_post = float(idata.posterior["mu"].values.mean())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    lims = (0.0, 0.6)
    ax.plot(lims, lims, "--", color="gray", lw=1, label="y = x(縮小なし)", zorder=1)
    ax.axhline(mu_post, color="black", lw=0.8, ls=":", label=f"全体平均 μ={mu_post:.3f}", zorder=1)

    for orate, pmean, n in zip(obs_rate, p_post_mean, n_ab):
        ax.plot([orate, orate], [orate, pmean], color="gray", lw=0.6, alpha=0.5, zorder=2)

    sizes = 15 + 60 * (n_ab / n_ab.max())
    sc = ax.scatter(obs_rate, p_post_mean, s=sizes, c=n_ab, cmap="viridis",
                     edgecolor="black", linewidth=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("観測打席数 n")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("観測された打率(x/n)")
    ax.set_ylabel("事後平均(部分プーリング後)")
    ax.set_title("階層Beta-Binomialモデルによる部分プーリング\n観測回数が少ない個体ほど全体平均へ強く縮小する")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_binomial_shrinkage.png")
    plt.close(fig)

    shrink = p_post_mean - obs_rate
    small_n_mask = n_ab <= 15
    large_n_mask = n_ab >= 300
    print(f"beta_binomial_shrinkage.png saved (mu_post={mu_post:.3f}, "
          f"mean|shrink| n<=15: {np.abs(shrink[small_n_mask]).mean():.3f}, "
          f"n>=300: {np.abs(shrink[large_n_mask]).mean():.3f})")


def plot_gamma_poisson_overdispersion():
    """overdispersedなカウントデータに対し、Poisson固定分散モデルの予測区間が
    過小評価になる一方、Gamma-Poisson階層モデルは実測の散らばりを正しく捉えることを示す。"""

    rng = np.random.default_rng(9)
    n = 40
    true_mean = 8.0
    true_alpha_conc = 3.0  # 小さいほどoverdispersionが強い
    lam_i = rng.gamma(true_alpha_conc, true_mean / true_alpha_conc, size=n)
    y_obs = rng.poisson(lam_i)

    with pm.Model():
        lam = pm.Gamma("lam", alpha=2.0, beta=0.2)
        pm.Poisson("y", mu=lam, observed=y_obs)
        idata_poisson = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
        ppc_poisson = pm.sample_posterior_predictive(idata_poisson, progressbar=False)

    with pm.Model():
        mu = pm.Gamma("mu", alpha=2.0, beta=0.2)
        alpha_conc = pm.Gamma("alpha_conc", alpha=2.0, beta=0.3)
        beta = pm.Deterministic("beta", alpha_conc / mu)
        lam_i_gp = pm.Gamma("lam_i", alpha=alpha_conc, beta=beta, shape=n)
        pm.Poisson("y", mu=lam_i_gp, observed=y_obs)
        idata_gp = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
        ppc_gp = pm.sample_posterior_predictive(idata_gp, progressbar=False)

    y_pred_poisson = ppc_poisson.posterior_predictive["y"].values.reshape(-1, n)
    y_pred_gp = ppc_gp.posterior_predictive["y"].values.reshape(-1, n)

    lo_p, hi_p = np.percentile(y_pred_poisson.flatten(), [2.5, 97.5])
    lo_gp, hi_gp = np.percentile(y_pred_gp.flatten(), [2.5, 97.5])

    coverage_poisson = np.mean((y_obs >= lo_p) & (y_obs <= hi_p))
    coverage_gp = np.mean((y_obs >= lo_gp) & (y_obs <= hi_gp))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    bins = np.arange(0, max(y_obs.max(), y_pred_gp.max()) + 2) - 0.5

    for ax, y_pred, lo, hi, cov, color, title in [
        (axes[0], y_pred_poisson, lo_p, hi_p, coverage_poisson, COLOR_OK, "Poisson固定分散モデル"),
        (axes[1], y_pred_gp, lo_gp, hi_gp, coverage_gp, COLOR_ALT, "Gamma-Poisson階層モデル"),
    ]:
        ax.hist(y_pred.flatten(), bins=bins, density=True, color=color, alpha=0.4, label="事後予測分布")
        ax.hist(y_obs, bins=bins, density=True, histtype="step", color="black", lw=1.5, label="実測データ")
        ax.axvspan(lo, hi, color=color, alpha=0.15, label=f"95%予測区間 [{lo:.0f}, {hi:.0f}]")
        ax.set_title(f"{title}\n実測データの区間内カバー率: {cov*100:.0f}%")
        ax.set_xlabel("カウント数")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("密度")
    fig.suptitle("Gamma-Poissonによるoverdispersion補正", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "gamma_poisson_overdispersion.png")
    plt.close(fig)
    print(f"gamma_poisson_overdispersion.png saved "
          f"(obs var/mean={y_obs.var()/y_obs.mean():.2f}, "
          f"Poisson 95%PI=[{lo_p:.0f},{hi_p:.0f}] coverage={coverage_poisson*100:.0f}%, "
          f"Gamma-Poisson 95%PI=[{lo_gp:.0f},{hi_gp:.0f}] coverage={coverage_gp*100:.0f}%)")


if __name__ == "__main__":
    plot_beta_binomial_shrinkage()
    plot_gamma_poisson_overdispersion()
