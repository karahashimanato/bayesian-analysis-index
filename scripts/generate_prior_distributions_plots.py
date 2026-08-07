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


def plot_positive_scale_zero_density():
    """0付近に密度の山を持つ分布(Exponential, HalfNormal, HalfCauchy)と、
    0での密度がゼロになる分布(Gamma(alpha>1,...))を、0近傍に注目して
    密度を重ね描きする。"""

    x = np.linspace(0.001, 4, 500)
    x_zoom = np.linspace(0.001, 0.6, 500)

    dists = [
        ("Exponential(1)(0で密度が最大)", stats.expon(scale=1.0), COLOR_DIVERGENT),
        ("HalfNormal(σ=1)(0で密度が最大)", stats.halfnorm(scale=1.0), COLOR_CHAIN[1]),
        ("HalfCauchy(β=1)(0で密度が最大)", stats.halfcauchy(scale=1.0), COLOR_CHAIN[2]),
        ("Gamma(shape=2, rate=1)(0で密度がゼロ)", stats.gamma(a=2.0, scale=1.0), COLOR_CHAIN[0]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for label, dist, color in dists:
        axes[0].plot(x, dist.pdf(x), color=color, label=label)
        axes[1].plot(x_zoom, dist.pdf(x_zoom), color=color, label=label)

    axes[0].set_xlabel("x(分母に来るパラメータ)")
    axes[0].set_ylabel("density")
    axes[0].set_title("全体の形状")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_xlabel("x(0近傍を拡大)")
    axes[1].set_ylabel("density")
    axes[1].set_title("0近傍を拡大: Gammaだけ密度がゼロから立ち上がる")
    axes[1].set_xlim(0, 0.6)

    fig.suptitle("0付近の密度の山の有無が、分母的パラメータの分散爆発リスクを決める", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "positive_scale_zero_density.png")
    plt.close(fig)

    pdf_at_0 = {label: float(dist.pdf(0.001)) for label, dist, _ in dists}
    print("positive_scale_zero_density.png saved (" +
          ", ".join(f"{label}: pdf(0.001)={v:.2f}" for label, v in pdf_at_0.items()) + ")")


def plot_changepoint_discrete_vs_continuous():
    """変化点位置の事前分布として、DiscreteUniformのPMFと、
    シグモイド関数による連続緩和(急峻さ違い)を並べて示す。"""

    T = 20
    tau_vals = np.arange(1, T)
    pmf = np.full_like(tau_vals, 1.0 / len(tau_vals), dtype=float)

    t = np.linspace(0, T, 400)
    tau0 = 10.0
    steepness_list = [0.5, 2.0, 8.0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(tau_vals, pmf, color=COLOR_CHAIN[0], width=0.7)
    axes[0].set_xlabel("τ(変化点の時点)")
    axes[0].set_ylabel("P(τ)")
    axes[0].set_title(f"DiscreteUniform(1, {T-1})\n(離散、Compound StepでESSが低下しやすい)")

    for k, color in zip(steepness_list, COLOR_CHAIN[1:]):
        sig = 1.0 / (1.0 + np.exp(-k * (t - tau0)))
        axes[1].plot(t, sig, color=color, label=f"急峻さ k={k}")
    axes[1].axvline(tau0, color="gray", lw=0.8, ls=":", label=f"τ0={tau0:.0f}(中心)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("sigmoid(k(t-τ0))")
    axes[1].set_title("連続緩和: kが大きいほど\n離散のswitchに近づく")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("変化点位置: 離散一様分布 vs シグモイドによる連続緩和", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "changepoint_discrete_vs_continuous.png")
    plt.close(fig)

    print(f"changepoint_discrete_vs_continuous.png saved "
          f"(DiscreteUniform(1,{T-1}), steepness={steepness_list})")


def plot_normal_sigma_implausible_mass():
    """回帰係数の事前分布Normal(0, σ)について、σを変えたときの密度形状と、
    「非現実的に大きい効果」とみなす閾値を超える確率質量がどう増えるかを示す。"""

    threshold = 5.0  # 標準化された説明変数に対し、係数の絶対値がこれを超えると
    # sigmoid/expなど典型的なリンク関数がほぼ0/1・飽和してしまう非現実的な大きさとみなす
    sigmas = [0.5, 1.0, 2.0, 5.0]
    x = np.linspace(-10, 10, 500)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for sigma, color in zip(sigmas, COLOR_CHAIN):
        axes[0].plot(x, stats.norm.pdf(x, 0, sigma), color=color, label=f"σ={sigma}")
    axes[0].axvspan(threshold, 10, color=COLOR_DIVERGENT, alpha=0.12)
    axes[0].axvspan(-10, -threshold, color=COLOR_DIVERGENT, alpha=0.12,
                     label=f"|β|>{threshold:.0f}(非現実的に大きい効果)")
    axes[0].set_xlabel("β(標準化された説明変数の回帰係数)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Normal(0, σ)の形状とσの違い")
    axes[0].legend(fontsize=8.5)

    p_extreme = [2 * stats.norm.sf(threshold, 0, s) for s in sigmas]
    axes[1].bar([f"σ={s}" for s in sigmas], p_extreme, color=COLOR_CHAIN)
    for i, p in enumerate(p_extreme):
        axes[1].annotate(f"{p:.1%}", (i, p), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=9)
    axes[1].set_ylabel(f"P(|β| > {threshold:.0f})")
    axes[1].set_title(f"σを緩めるほど非現実的な領域\n(|β|>{threshold:.0f})の事前確率質量が増える")

    fig.suptitle("Normal(0, σ)の σ は「非現実的に大きい効果」をどれだけ許すかを決める", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "normal_sigma_implausible_mass.png")
    plt.close(fig)

    print("normal_sigma_implausible_mass.png saved (" +
          ", ".join(f"sigma={s}: P(|beta|>{threshold:.0f})={p:.1%}" for s, p in zip(sigmas, p_extreme)) + ")")


def plot_lognormal_symmetric_log_scale():
    """LogNormalが線形スケールでは右に強く歪む一方、対数スケールでは
    対称な釣鐘型になることを示す。"""

    true_scale = 100.0
    sigma_log = 0.5
    mu_log = np.log(true_scale)

    x = np.linspace(1, 400, 1000)
    log_x = np.linspace(np.log(1), np.log(400), 1000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(x, stats.lognorm.pdf(x, sigma_log, scale=np.exp(mu_log)), color=COLOR_CHAIN[0])
    axes[0].axvline(true_scale, color="gray", lw=0.8, ls=":", label=f"中央値={true_scale:.0f}")
    axes[0].set_xlabel("x(線形スケール)")
    axes[0].set_ylabel("density")
    axes[0].set_title("線形スケールでは右に強く歪む")
    axes[0].legend(fontsize=9)

    # 対数スケール上の密度(変数変換: log(x)の密度はNormal(mu_log, sigma_log)そのもの)
    axes[1].plot(log_x, stats.norm.pdf(log_x, mu_log, sigma_log), color=COLOR_DIVERGENT)
    axes[1].axvline(mu_log, color="gray", lw=0.8, ls=":", label=f"中央値=log({true_scale:.0f})")
    axes[1].set_xlabel("log(x)(対数スケール)")
    axes[1].set_ylabel("density")
    axes[1].set_title("対数スケールでは対称な釣鐘型\n(log(x) ~ Normal(μ, σ)そのもの)")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"LogNormal(μ=log({true_scale:.0f}), σ={sigma_log}): 対数スケールで対称な不確実性を表現する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "lognormal_symmetric_log_scale.png")
    plt.close(fig)

    skew = float(stats.lognorm.stats(sigma_log, scale=np.exp(mu_log), moments="s"))
    print(f"lognormal_symmetric_log_scale.png saved (linear-scale skewness={skew:.2f}, log-scale skewness=0)")


if __name__ == "__main__":
    plot_beta_kappa_concentration()
    plot_positive_scale_zero_density()
    plot_changepoint_discrete_vs_continuous()
    plot_normal_sigma_implausible_mass()
    plot_lognormal_symmetric_log_scale()
