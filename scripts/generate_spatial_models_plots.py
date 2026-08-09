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


def plot_lgcp_latent_vs_hsgp():
    """LGCP(空間点過程)を12x12格子に離散化し、厳密GP(pm.gp.Latent)と
    HSGP近似を実際にPyMCでフィットして、強度場の復元結果と実行時間を比較する。"""

    rng = np.random.default_rng(17)

    side = 12
    xs, ys = np.meshgrid(np.linspace(0, 10, side), np.linspace(0, 10, side), indexing="ij")
    coords = np.column_stack([xs.flatten(), ys.flatten()])

    true_log_intensity = -1.0 + 2.0 * np.exp(
        -((xs.flatten() - 5) ** 2 + (ys.flatten() - 6) ** 2) / 4
    )
    cell_area = (10 / side) ** 2
    counts = rng.poisson(np.exp(true_log_intensity) * cell_area)

    import time

    t0 = time.perf_counter()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=3, beta=1)
        eta = pm.HalfNormal("eta", 2.0)
        cov = eta ** 2 * pm.gp.cov.Matern52(2, ls=ell)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=coords)
        mu0 = pm.Normal("mu0", -1.0, 2.0)
        pm.Poisson("y", mu=pm.math.exp(mu0 + f) * cell_area, observed=counts)
        idata_latent = pm.sample(300, tune=500, chains=2, target_accept=0.9,
                                  random_seed=1, progressbar=False,
                                  compute_convergence_checks=False)
    t_latent = time.perf_counter() - t0
    n_div_latent = int(idata_latent.sample_stats["diverging"].sum())
    f_latent = (idata_latent.posterior["mu0"] + idata_latent.posterior["f"]).values
    log_int_latent = f_latent.reshape(-1, side * side).mean(axis=0)

    t0 = time.perf_counter()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=3, beta=1)
        eta = pm.HalfNormal("eta", 2.0)
        cov = eta ** 2 * pm.gp.cov.Matern52(2, ls=ell)
        gp = pm.gp.HSGP(m=[12, 12], c=2.0, cov_func=cov)
        f = gp.prior("f", X=coords)
        mu0 = pm.Normal("mu0", -1.0, 2.0)
        pm.Poisson("y", mu=pm.math.exp(mu0 + f) * cell_area, observed=counts)
        idata_hsgp = pm.sample(300, tune=500, chains=2, target_accept=0.9,
                                random_seed=2, progressbar=False,
                                compute_convergence_checks=False)
    t_hsgp = time.perf_counter() - t0
    n_div_hsgp = int(idata_hsgp.sample_stats["diverging"].sum())
    f_hsgp = (idata_hsgp.posterior["mu0"] + idata_hsgp.posterior["f"]).values
    log_int_hsgp = f_hsgp.reshape(-1, side * side).mean(axis=0)

    speedup = t_latent / t_hsgp

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8])

    vmin, vmax = -1.5, 1.5
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(true_log_intensity.reshape(side, side).T, origin="lower",
                      cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.set_title("真の対数強度場")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(log_int_latent.reshape(side, side).T, origin="lower",
                      cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title(f"pm.gp.Latent(厳密)推定\ndivergence={n_div_latent}")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(log_int_hsgp.reshape(side, side).T, origin="lower",
                      cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title(f"pm.gp.HSGP(近似)推定\ndivergence={n_div_hsgp}")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, :])
    labels = ["pm.gp.Latent\n(厳密GP)", "pm.gp.HSGP\n(基底関数近似)"]
    times = [t_latent, t_hsgp]
    colors = [COLOR_DIVERGENT, COLOR_OK]
    ax3.barh(labels, times, color=colors, height=0.5, log=True)
    for i, v in enumerate(times):
        ax3.annotate(f"{v:.1f}s", (v, i), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=11)
    ax3.set_xlabel("実行時間(秒、対数軸)")
    ax3.set_title(f"12x12格子(144セル)での実行時間比較: {speedup:.1f}倍高速化")

    fig.suptitle("LGCP: 厳密GPとHSGP近似の強度場推定と実行時間比較", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "lgcp_latent_vs_hsgp.png")
    plt.close(fig)

    corr_latent = np.corrcoef(true_log_intensity, log_int_latent)[0, 1]
    corr_hsgp = np.corrcoef(true_log_intensity, log_int_hsgp)[0, 1]
    print(f"lgcp_latent_vs_hsgp.png saved (t_latent={t_latent:.1f}s divergence={n_div_latent} "
          f"corr={corr_latent:.3f}, t_hsgp={t_hsgp:.1f}s divergence={n_div_hsgp} "
          f"corr={corr_hsgp:.3f}, speedup={speedup:.1f}x)")


def _build_chain_laplacian(t):
    W = np.zeros((t, t))
    for i in range(t - 1):
        W[i, i + 1] = W[i + 1, i] = 1
    D = W.sum(axis=1)
    return np.diag(D) - W


def plot_knorr_held_type_i_vs_iv():
    """空間時系列BYM(Knorr-Held型)のType I(非構造交互作用)とType IV
    (クロネッカー積構造の交互作用)を5x5格子x10時点の合成データで実際に
    PyMCでフィットし、交互作用の復元結果とLOOでの予測性能を比較する。"""

    rng = np.random.default_rng(11)
    side = 5
    n_space = side * side
    t_len = 10

    def _adj(s):
        n = s * s
        W = np.zeros((n, n))
        for r in range(s):
            for c in range(s):
                i = r * s + c
                if r > 0:
                    j = (r - 1) * s + c
                    W[i, j] = W[j, i] = 1
                if c > 0:
                    j = r * s + (c - 1)
                    W[i, j] = W[j, i] = 1
        return W

    W_space = _adj(side)
    Q_space = np.diag(W_space.sum(axis=1)) - W_space
    Q_time = _build_chain_laplacian(t_len)

    rr, cc = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    hotspot = np.exp(-((rr.flatten() - 2) ** 2 + (cc.flatten() - 2) ** 2) / 3.0)
    tt = np.arange(t_len)
    psi_true = 0.5 * np.outer(hotspot, tt / (t_len - 1))
    beta0_true = -0.5
    S_true = 0.3 * (rr.flatten() - cc.flatten()) / side
    delta_true = 0.05 * tt

    E = rng.uniform(50, 150, (n_space, t_len))
    log_rr_true = beta0_true + S_true[:, None] + delta_true[None, :] + psi_true
    counts = rng.poisson(E * np.exp(log_rr_true))
    counts_flat = counts.flatten()

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_S = pm.HalfNormal("sigma_S", 1)
        S = pm.ICAR("S", W=W_space.astype(int), sigma=sigma_S)
        sigma_delta = pm.HalfNormal("sigma_delta", 1)
        delta_raw = pm.Normal("delta_raw", 0, 1, shape=t_len)
        delta = pm.Deterministic("delta", pt.cumsum(delta_raw) * sigma_delta * 0.3)
        sigma_psi = pm.HalfNormal("sigma_psi", 0.5)
        psi = pm.Normal("psi", 0, sigma_psi, shape=(n_space, t_len))
        log_rr = beta0 + S[:, None] + delta[None, :] + psi
        mu = E * pm.math.exp(log_rr)
        pm.Poisson("y", mu=mu.flatten(), observed=counts_flat)
        idata1 = pm.sample(800, tune=1500, chains=4, target_accept=0.95,
                            random_seed=1, progressbar=False,
                            compute_convergence_checks=False)
        pm.compute_log_likelihood(idata1)
    ndiv1 = int(idata1.sample_stats["diverging"].sum())
    psi_est1 = idata1.posterior["psi"].values.reshape(-1, n_space, t_len).mean(axis=0)

    # Type IV: 固有基底によるsum-to-zero制約付きクロネッカー積構造(Clayton制約の直接実装)
    eigval_s, eigvec_s = np.linalg.eigh(Q_space)
    eigval_t, eigvec_t = np.linalg.eigh(Q_time)
    V_s, lam_s = eigvec_s[:, 1:], eigval_s[1:]
    V_t, lam_t = eigvec_t[:, 1:], eigval_t[1:]
    inv_scale = 1.0 / np.sqrt(np.outer(lam_s, lam_t))

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_S = pm.HalfNormal("sigma_S", 1)
        S = pm.ICAR("S", W=W_space.astype(int), sigma=sigma_S)
        sigma_delta = pm.HalfNormal("sigma_delta", 1)
        delta_raw = pm.Normal("delta_raw", 0, 1, shape=t_len)
        delta = pm.Deterministic("delta", pt.cumsum(delta_raw) * sigma_delta * 0.3)
        sigma_psi = pm.HalfNormal("sigma_psi", 0.5)
        xi_raw = pm.Normal("xi_raw", 0, 1, shape=(n_space - 1, t_len - 1))
        xi = xi_raw * pt.as_tensor(inv_scale) * sigma_psi
        psi = pm.Deterministic("psi", pt.as_tensor(V_s) @ xi @ pt.as_tensor(V_t).T)
        log_rr = beta0 + S[:, None] + delta[None, :] + psi
        mu = E * pm.math.exp(log_rr)
        pm.Poisson("y", mu=mu.flatten(), observed=counts_flat)
        idata4 = pm.sample(800, tune=1500, chains=4, target_accept=0.95,
                            random_seed=2, progressbar=False,
                            compute_convergence_checks=False)
        pm.compute_log_likelihood(idata4)
    ndiv4 = int(idata4.sample_stats["diverging"].sum())
    psi_est4 = idata4.posterior["psi"].values.reshape(-1, n_space, t_len).mean(axis=0)

    loo1 = az.loo(idata1)
    loo4 = az.loo(idata4)
    cmp = az.compare({"Type I(非構造)": idata1, "Type IV(クロネッカー積構造)": idata4})

    corr1 = np.corrcoef(psi_true.flatten(), psi_est1.flatten())[0, 1]
    corr4 = np.corrcoef(psi_true.flatten(), psi_est4.flatten())[0, 1]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8])

    vmax_true = np.abs(psi_true).max()
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(psi_true, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-vmax_true, vmax=vmax_true)
    ax0.set_title("真の交互作用 psi_true")
    ax0.set_xlabel("時間 t")
    ax0.set_ylabel("地区(空間)")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    # 推定パネルは真値より振幅が縮小する(事後平均の収縮)ため、パネルごとに色スケールを揃える
    vmax_est = max(np.abs(psi_est1).max(), np.abs(psi_est4).max())
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(psi_est1, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-vmax_est, vmax=vmax_est)
    ax1.set_title(f"Type I推定(相関={corr1:.2f})\ndivergence={ndiv1}")
    ax1.set_xlabel("時間 t")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(psi_est4, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-vmax_est, vmax=vmax_est)
    ax2.set_title(f"Type IV推定(相関={corr4:.2f})\ndivergence={ndiv4}")
    ax2.set_xlabel("時間 t")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, :])
    labels = list(cmp.index)
    elpd_diffs = [float(cmp.loc[lbl, "elpd_diff"]) for lbl in labels]
    dses = [float(cmp.loc[lbl, "dse"]) for lbl in labels]
    colors = [COLOR_OK, COLOR_ALT]
    ax3.barh(labels, elpd_diffs, xerr=dses, color=colors, height=0.5, capsize=4)
    for i, v in enumerate(elpd_diffs):
        ax3.annotate(f"{v:+.1f}", (v, i), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=10)
    ax3.axvline(0, color="black", lw=1)
    ax3.set_xlabel("最良モデルからのelpd差(0=最良、誤差棒=dse)")
    ax3.set_title("Type IとType IVのLOO予測性能比較(誤差棒がdseを超えて0から離れなければ有意差なし)")

    fig.suptitle("空間時系列BYM: Type I(非構造)とType IV(クロネッカー積構造)の交互作用比較", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "knorr_held_type_i_vs_iv.png")
    plt.close(fig)

    print(f"knorr_held_type_i_vs_iv.png saved (Type I: divergence={ndiv1} corr={corr1:.3f} "
          f"elpd={float(loo1.elpd):.1f}se={float(loo1.se):.1f}, "
          f"Type IV: divergence={ndiv4} corr={corr4:.3f} "
          f"elpd={float(loo4.elpd):.1f}se={float(loo4.se):.1f})")


if __name__ == "__main__":
    plot_icar_field_recovery()
    plot_bym_nonidentifiability()
    plot_bym2_reparameterization_fix()
    plot_lgcp_latent_vs_hsgp()
    plot_knorr_held_type_i_vs_iv()
