"""
techniques/prior-predictive-check.md に埋め込む可視化画像を生成するスクリプト。

「分母のパラメータが0に近づくと分散が発散する」病理(Gamma-Poisson階層モデルの
μ²/α_concなど)を、Exponential事前分布とGamma(shape>1)事前分布の
prior predictiveの違いとして実際にサンプリングして描画する。

「判断基準は極値だけでなく割合・信用区間幅で見る」の実例として、理論上の
範囲(min/max)が同じ2つのprior predictive分布(Beta(5,5)とBeta(0.3,0.3)を
同じ[0,1000]にスケール)が、現実的な範囲に入る質量の割合ではどれだけ
異なるかを比較する。

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


def plot_extreme_vs_proportion():
    """理論上の範囲(min/max)が同じでも、現実的な範囲に入る質量の割合は
    形状次第で大きく異なることを示す。"""

    rng = np.random.default_rng(1)
    n = 20000
    scale = 1000.0
    band_lo, band_hi = 300.0, 700.0

    well = rng.beta(5.0, 5.0, n) * scale       # 中央に集中
    mis = rng.beta(0.3, 0.3, n) * scale        # 両端(0, 1000)に偏るU字型

    prop_well = np.mean((well >= band_lo) & (well <= band_hi))
    prop_mis = np.mean((mis >= band_lo) & (mis <= band_hi))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bins = np.linspace(0, scale, 60)
    axes[0].axvspan(band_lo, band_hi, color="gray", alpha=0.15, label="現実的な範囲[300,700]")
    axes[0].hist(well, bins=bins, alpha=0.6, color=COLOR_OK,
                 label=f"Beta(5,5)(min={well.min():.0f}, max={well.max():.0f})")
    axes[0].hist(mis, bins=bins, alpha=0.6, color=COLOR_DIVERGENT,
                 label=f"Beta(0.3,0.3)(min={mis.min():.0f}, max={mis.max():.0f})")
    axes[0].set_xlabel("prior predictiveの値")
    axes[0].set_ylabel("count")
    axes[0].set_title("理論上の範囲[0,1000]はほぼ同じ")
    axes[0].legend(loc="upper center", fontsize=8, framealpha=0.9)

    labels = ["Beta(5,5)\n(中央集中)", "Beta(0.3,0.3)\n(両端に偏る)"]
    props = [prop_well * 100, prop_mis * 100]
    axes[1].bar(labels, props, color=[COLOR_OK, COLOR_DIVERGENT], alpha=0.85)
    for i, p in enumerate(props):
        axes[1].text(i, p + 1.5, f"{p:.1f}%", ha="center", fontsize=11)
    axes[1].set_ylabel("現実的な範囲[300,700]に入る割合(%)")
    axes[1].set_title("割合で見ると大きく異なる")
    axes[1].set_ylim(0, 100)

    fig.suptitle("min/maxの範囲が同じでも、現実的な範囲に入る割合は形状次第で大きく異なる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "extreme_vs_proportion.png")
    plt.close(fig)
    print(f"extreme_vs_proportion.png saved "
          f"(Beta(5,5): min={well.min():.0f} max={well.max():.0f} 帯域内={prop_well*100:.1f}%, "
          f"Beta(0.3,0.3): min={mis.min():.0f} max={mis.max():.0f} 帯域内={prop_mis*100:.1f}%)")


if __name__ == "__main__":
    plot_denominator_variance_explosion()
    plot_extreme_vs_proportion()
