"""
tools/prior-distributions.md に埋め込む可視化画像を生成するスクリプト。

Beta(mu*kappa, (1-mu)*kappa) の再パラメータ化で、全体平均muを固定したまま
集中度kappaを変えると、個体差の大きさ(分布の広がり)がどう変化するかを描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_prior_distributions_plots.py

出力先: assets/prior-distributions/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_style import COLOR_CHAIN, COLOR_DIVERGENT, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "prior-distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_beta_kappa_concentration():
    """全体平均muを固定し、集中度kappaを変えたときにBeta(mu*kappa, (1-mu)*kappa)の
    広がり(=想定する個体差の大きさ)がどう変化するかを示す。"""

    mu = 0.3
    kappas = [5, 20, 80, 300]
    x = np.linspace(0.001, 0.999, 500)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for kappa, color in zip(kappas, COLOR_CHAIN):
        alpha, beta = mu * kappa, (1 - mu) * kappa
        sd = np.sqrt(mu * (1 - mu) / (kappa + 1))
        axes[0].plot(x, stats.beta.pdf(x, alpha, beta), color=color,
                     label=f"κ={kappa} (sd={sd:.3f})")
    axes[0].axvline(mu, color="gray", lw=0.8, ls=":", label=f"μ={mu}(固定)")
    axes[0].set_xlabel("p(個体ごとの確率・割合)")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"μ={mu}を固定し、κだけを変えた\nBeta(μκ, (1-μ)κ)の形状")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    kappa_range = np.logspace(np.log10(2), np.log10(1000), 200)
    sd_range = np.sqrt(mu * (1 - mu) / (kappa_range + 1))
    axes[1].plot(kappa_range, sd_range, color=COLOR_DIVERGENT)
    for kappa, color in zip(kappas, COLOR_CHAIN):
        sd = np.sqrt(mu * (1 - mu) / (kappa + 1))
        axes[1].scatter([kappa], [sd], color=color, zorder=3, s=50)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("κ(集中度、対数軸)")
    axes[1].set_ylabel("Beta分布の標準偏差(=個体差の大きさ)")
    axes[1].set_title("κが大きいほど個体差は\n急激に小さくなる")

    fig.suptitle("Beta(μκ, (1-μ)κ)の再パラメータ化: κは個体差の大きさを表す", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "beta_kappa_concentration.png")
    plt.close(fig)

    sds = [np.sqrt(mu * (1 - mu) / (k + 1)) for k in kappas]
    print("beta_kappa_concentration.png saved (" +
          ", ".join(f"kappa={k}: sd={s:.3f}" for k, s in zip(kappas, sds)) + ")")


if __name__ == "__main__":
    plot_beta_kappa_concentration()
