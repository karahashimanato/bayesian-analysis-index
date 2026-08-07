"""
techniques/prior-predictive-check.md に埋め込む可視化画像を生成するスクリプト。

「分母のパラメータが0に近づくと分散が発散する」病理(Gamma-Poisson階層モデルの
μ²/α_concなど)を、Exponential事前分布とGamma(shape>1)事前分布の
prior predictiveの違いとして実際にサンプリングして描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_prior_predictive_plots.py

出力先: assets/prior-predictive/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_style import COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "prior-predictive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_denominator_variance_explosion():
    """分母に来るパラメータαの事前分布次第で、分散 mu^2/alpha の
    prior predictiveが暴走するかどうかが決まることを示す。"""

    rng = np.random.default_rng(0)
    n = 50000
    mu = 10.0  # 観測データの平均スケール(固定)

    alpha_exp = rng.exponential(1.0, n)       # Exponential(1): 0で密度が最大
    alpha_gamma = rng.gamma(2.0, 1.0, n)      # Gamma(shape=2): 0で密度がゼロ

    var_exp = mu**2 / alpha_exp
    var_gamma = mu**2 / alpha_gamma

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # 左: alphaそのものの事前分布(0付近の密度の違いが原因)
    x = np.linspace(0.001, 4, 500)
    axes[0].plot(x, stats.expon.pdf(x, scale=1.0), color=COLOR_DIVERGENT,
                 label="Exponential(1): 0で密度が最大")
    axes[0].plot(x, stats.gamma.pdf(x, a=2.0, scale=1.0), color=COLOR_OK,
                 label="Gamma(shape=2): 0で密度がゼロ")
    axes[0].set_xlabel("α(分母のパラメータ)")
    axes[0].set_ylabel("density")
    axes[0].set_title("原因: αの事前分布が\n0付近にどれだけ質量を置くか")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # 右: 結果として生じる分散 mu^2/alpha の prior predictive(対数軸)
    bins = np.logspace(0, 6, 60)
    axes[1].hist(var_exp, bins=bins, alpha=0.6, color=COLOR_DIVERGENT,
                 label=f"Exponential(1)由来\n(99%ile={np.percentile(var_exp,99):,.0f}, "
                       f"最大={var_exp.max():,.0f})")
    axes[1].hist(var_gamma, bins=bins, alpha=0.6, color=COLOR_OK,
                 label=f"Gamma(shape=2)由来\n(99%ile={np.percentile(var_gamma,99):,.0f}, "
                       f"最大={var_gamma.max():,.0f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("prior predictiveの分散 = μ²/α (μ=10で固定)")
    axes[1].set_ylabel("count")
    axes[1].set_title("結果: 分散のprior predictiveが\n暴走するかどうか")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("「分母のパラメータが0に近づくと分散が発散する」病理", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "denominator_variance_explosion.png")
    plt.close(fig)
    print(f"denominator_variance_explosion.png saved "
          f"(Exponential p99={np.percentile(var_exp,99):.0f} max={var_exp.max():.0f}, "
          f"Gamma p99={np.percentile(var_gamma,99):.0f} max={var_gamma.max():.0f})")


if __name__ == "__main__":
    plot_denominator_variance_explosion()
