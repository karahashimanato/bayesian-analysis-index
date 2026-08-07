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

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

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


if __name__ == "__main__":
    plot_gaussian_random_walk_amplitude()
    plot_process_obs_noise_nonidentifiability()
