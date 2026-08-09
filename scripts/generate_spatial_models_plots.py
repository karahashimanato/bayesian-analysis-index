"""
tools/spatial-models.md に埋め込む可視化画像を生成するスクリプト。

8x8格子グラフ(4近傍)上の疾病マッピング風データ(Poisson相対リスクモデル)を
共通のテストベッドとして、ICAR/BYM/BYM2の3モデルを実際にPyMCでフィットする。

  1. ICAR: 空間構造項phiのみのモデルで、真の空間場を復元できることを示す。
  2. BYM: 非構造項theta・空間構造項phiが事後分布上で分離しにくい
     (ridge型非識別性)ことを示す。
  3. BYM2: sigma/rhoへの再パラメータ化でBYMのr_hat悪化が解消することを示す。

実行方法:
    source .venv/bin/activate
    python scripts/generate_spatial_models_plots.py

出力先: assets/spatial-models/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "spatial-models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

SIDE = 8
N = SIDE * SIDE


def _build_grid_adjacency():
    def idx(r, c):
        return r * SIDE + c

    W = np.zeros((N, N), dtype=int)
    for r in range(SIDE):
        for c in range(SIDE):
            i = idx(r, c)
            if r > 0:
                W[i, idx(r - 1, c)] = 1
                W[idx(r - 1, c), i] = 1
            if c > 0:
                W[i, idx(r, c - 1)] = 1
                W[idx(r, c - 1), i] = 1
    return W


def _simulate_bym_data(rng):
    W = _build_grid_adjacency()
    rr, cc = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    phi_true = 0.15 * (rr.flatten() + cc.flatten()) - 0.15 * (SIDE - 1)
    phi_true -= phi_true.mean()
    theta_true = rng.normal(0, 0.3, N)
    E = rng.uniform(50, 150, N)
    log_rr_true = phi_true + theta_true
    counts = rng.poisson(E * np.exp(log_rr_true))
    return W, phi_true, theta_true, E, counts


def plot_icar_field_recovery():
    """ICAR単独モデル(空間構造項phiのみ)を実際にPyMCでフィットし、
    格子グラフ上の真の空間場を復元できることを示す。"""

    rng = np.random.default_rng(3)
    W, phi_true, theta_true, E, counts = _simulate_bym_data(rng)

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_phi = pm.HalfNormal("sigma_phi", 1)
        phi = pm.ICAR("phi", W=W, sigma=sigma_phi)
        log_rr = beta0 + phi
        pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    phi_est = idata.posterior["phi"].values.reshape(-1, N).mean(axis=0)
    n_div = int(idata.sample_stats["diverging"].sum())
    corr = np.corrcoef(phi_true, phi_est)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(phi_true.reshape(SIDE, SIDE), cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    axes[0].set_title("真の空間場 phi_true")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(phi_est.reshape(SIDE, SIDE), cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    axes[1].set_title(f"ICARで推定した phi(相関={corr:.2f})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"ICAR: 8x8格子上の真の空間場を復元(divergence={n_div})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "icar_field_recovery.png")
    plt.close(fig)

    print(f"icar_field_recovery.png saved (divergence={n_div}, phi_true-phi_est相関={corr:.3f})")


def plot_bym_nonidentifiability():
    """BYMモデル(非構造項theta+空間構造項phi)を実際にPyMCでフィットし、
    thetaとphiが事後分布上でridge型の非識別性を持つことを示す。"""

    rng = np.random.default_rng(3)
    W, phi_true, theta_true, E, counts = _simulate_bym_data(rng)

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_theta = pm.HalfNormal("sigma_theta", 1)
        sigma_phi = pm.HalfNormal("sigma_phi", 1)
        theta = pm.Normal("theta", 0, sigma_theta, shape=N)
        phi = pm.ICAR("phi", W=W, sigma=sigma_phi)
        log_rr = beta0 + theta + phi
        pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    theta_draws = idata.posterior["theta"].values.reshape(-1, N)
    phi_draws = idata.posterior["phi"].values.reshape(-1, N)
    corr0 = np.corrcoef(theta_draws[:, 0], phi_draws[:, 0])[0, 1]

    rhat_theta = float(az.rhat(idata, var_names=["sigma_theta"])["sigma_theta"].values)
    rhat_phi = float(az.rhat(idata, var_names=["sigma_phi"])["sigma_phi"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(theta_draws[:, 0], phi_draws[:, 0], s=4, alpha=0.15, color=COLOR_ALT)
    axes[0].set_xlabel("theta[0](非構造項)")
    axes[0].set_ylabel("phi[0](空間構造項)")
    axes[0].set_title(f"地区0の事後サンプル相関: r={corr0:.2f}\n(データはtheta+phiの合計しか制約しない)")

    labels = ["sigma_theta\n(非構造項の分散)", "sigma_phi\n(空間構造項の分散)"]
    rhats = [rhat_theta, rhat_phi]
    colors = [COLOR_DIVERGENT if r > 1.01 else COLOR_OK for r in rhats]
    axes[1].bar(labels, rhats, color=colors, width=0.5)
    axes[1].axhline(1.01, color="black", lw=1, ls="--", label="r_hat=1.01の目安")
    for i, v in enumerate(rhats):
        axes[1].annotate(f"{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=10)
    axes[1].set_ylabel("r_hat")
    axes[1].set_title("sigma_thetaとsigma_phiのr_hat")
    axes[1].legend(fontsize=9)

    fig.suptitle("BYM: thetaとphiのridge型非識別性", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "bym_nonidentifiability.png")
    plt.close(fig)

    print(f"bym_nonidentifiability.png saved (theta[0]-phi[0]相関={corr0:.3f}, "
          f"rhat_sigma_theta={rhat_theta:.3f}, rhat_sigma_phi={rhat_phi:.3f})")


def plot_bym2_reparameterization_fix():
    """BYM2(sigma・rhoへの再パラメータ化)がBYMのr_hat悪化を解消することを、
    実際にPyMCで両方フィットして比較する。"""

    rng = np.random.default_rng(3)
    W, phi_true, theta_true, E, counts = _simulate_bym_data(rng)
    D = W.sum(axis=1)
    # BYM2のスケーリング係数(グラフラプラシアンの一般化逆行列の対角成分の幾何平均)
    Q = np.diag(D) - W
    Q_pert = Q + np.eye(N) * 1e-6
    Q_inv_diag = np.diag(np.linalg.inv(Q_pert))
    scale = float(np.exp(np.mean(np.log(Q_inv_diag))))

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_theta = pm.HalfNormal("sigma_theta", 1)
        sigma_phi = pm.HalfNormal("sigma_phi", 1)
        theta = pm.Normal("theta", 0, sigma_theta, shape=N)
        phi = pm.ICAR("phi", W=W, sigma=sigma_phi)
        log_rr = beta0 + theta + phi
        pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
        idata_bym = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        rho = pm.Beta("rho", 2, 2)
        theta_star = pm.Normal("theta_star", 0, 1, shape=N)
        phi_star = pm.ICAR("phi_star", W=W, sigma=1)
        combined = sigma * (pt.sqrt(1 - rho) * theta_star + pt.sqrt(rho / scale) * phi_star)
        log_rr = beta0 + combined
        pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
        idata_bym2 = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                                random_seed=2, progressbar=False,
                                compute_convergence_checks=False)

    rhat_bym = max(float(az.rhat(idata_bym, var_names=["sigma_theta"])["sigma_theta"].values),
                    float(az.rhat(idata_bym, var_names=["sigma_phi"])["sigma_phi"].values))
    rhat_bym2 = max(float(az.rhat(idata_bym2, var_names=["sigma"])["sigma"].values),
                     float(az.rhat(idata_bym2, var_names=["rho"])["rho"].values))
    rho_mean = float(idata_bym2.posterior["rho"].mean())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = ["BYM\n(sigma_theta/sigma_phi)", "BYM2\n(sigma/rho)"]
    rhats = [rhat_bym, rhat_bym2]
    colors = [COLOR_DIVERGENT if r > 1.01 else COLOR_OK for r in rhats]
    ax.bar(labels, rhats, color=colors, width=0.5)
    ax.axhline(1.01, color="black", lw=1, ls="--", label="r_hat=1.01の目安")
    for i, v in enumerate(rhats):
        ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylabel("分散成分パラメータの最大r_hat")
    ax.set_title(f"BYM2の再パラメータ化でr_hatが改善する\n(BYM2のrho事後平均={rho_mean:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bym2_reparameterization_fix.png")
    plt.close(fig)

    print(f"bym2_reparameterization_fix.png saved "
          f"(BYM max r_hat={rhat_bym:.3f}, BYM2 max r_hat={rhat_bym2:.3f}, rho事後平均={rho_mean:.3f})")


if __name__ == "__main__":
    plot_icar_field_recovery()
    plot_bym_nonidentifiability()
    plot_bym2_reparameterization_fix()
