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


if __name__ == "__main__":
    plot_trig_reparameterization()
    plot_ridge_ratio_reparameterization()
