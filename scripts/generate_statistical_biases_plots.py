"""
tools/statistical-biases.md に埋め込む可視化画像を生成するスクリプト。

ベイズロジスティック回帰をPyMCで実際にサンプリングし、Jensen不等式により
「係数の点推定から素朴に計算した予測確率」と「事後分布全体を通した予測確率の平均」
がどれだけズレるかを描画する。

また、Ecological Bias(集計データと個体データの乖離)の実例として、
群レベルの交絡変数が個体レベルのxとyの両方に影響する疑似データを生成し、
群固定効果を入れた個体レベル回帰(真の群内効果を復元)と、群平均どうしを
単純回帰した集計レベル回帰(群レベルの交絡に引きずられる)の係数を比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_statistical_biases_plots.py

出力先: assets/statistical-biases/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "statistical-biases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_jensen_inequality_gap():
    """ロジスティック回帰の係数不確実性が、Jensen不等式により
    予測確率の点推定と事後平均のあいだにどれだけのズレを生むかを示す。"""

    rng = np.random.default_rng(11)
    n = 15
    true_b0, true_b1 = -0.3, 1.2
    x_obs = rng.uniform(-3, 3, size=n)
    p_true = 1 / (1 + np.exp(-(true_b0 + true_b1 * x_obs)))
    y_obs = rng.binomial(1, p_true)

    with pm.Model():
        b0 = pm.Normal("b0", 0.0, 3.0)
        b1 = pm.Normal("b1", 0.0, 3.0)
        p = pm.Deterministic("p", pm.math.invlogit(b0 + b1 * x_obs))
        pm.Bernoulli("y", p=p, observed=y_obs)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    b0s = idata.posterior["b0"].values.flatten()
    b1s = idata.posterior["b1"].values.flatten()

    xg = np.linspace(-8, 8, 300)
    naive = 1 / (1 + np.exp(-(b0s.mean() + b1s.mean() * xg)))
    posterior_mean = np.mean(1 / (1 + np.exp(-(b0s[:, None] + b1s[:, None] * xg[None, :]))), axis=0)
    gap = posterior_mean - naive
    i_max = np.argmax(np.abs(gap))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(xg, naive, "--", color=COLOR_OK, label=r"素朴な点推定: sigmoid($\bar\beta_0+\bar\beta_1 x$)")
    axes[0].plot(xg, posterior_mean, "-", color=COLOR_ALT, label=r"事後平均: $E[\mathrm{sigmoid}(\beta_0+\beta_1 x)]$")
    axes[0].scatter(x_obs, y_obs, s=20, alpha=0.5, color="gray", label="観測データ", zorder=3)
    axes[0].axvline(xg[i_max], color="black", lw=0.8, ls=":")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("予測確率 p")
    axes[0].set_title("2本の予測曲線のズレ")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

    axes[1].plot(xg, gap, color=COLOR_ALT)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvline(xg[i_max], color="black", lw=0.8, ls=":")
    axes[1].scatter([xg[i_max]], [gap[i_max]], color=COLOR_ALT, zorder=3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("ズレ(事後平均 − 素朴な点推定)")
    axes[1].set_title(f"最大ズレ: {gap[i_max]:+.3f}(x={xg[i_max]:.1f})")

    fig.suptitle("Jensen不等式によるロジスティック回帰の予測確率のズレ", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "jensen_inequality_gap.png")
    plt.close(fig)
    print(f"jensen_inequality_gap.png saved (max gap={gap[i_max]:+.3f} at x={xg[i_max]:.2f}, "
          f"b1 posterior sd={b1s.std():.3f})")


def plot_ecological_bias():
    """群レベルの交絡変数z_gが個体レベルのx・yの両方に影響する疑似データで、
    群固定効果を入れた個体レベル回帰(真の群内効果beta_withinを復元)と、
    群平均どうしを単純回帰した集計レベル回帰(符号すら反転しうる)を比較する。"""

    rng = np.random.default_rng(23)
    G = 20
    n_per_group = 25
    beta_within = -0.6
    gamma = 2.5

    z_g = rng.normal(0, 1, G)
    group_idx = np.repeat(np.arange(G), n_per_group)
    within_noise = rng.normal(0, 0.7, G * n_per_group)
    x = 2 * z_g[group_idx] + within_noise
    y = (5 + beta_within * within_noise + gamma * z_g[group_idx]
         + rng.normal(0, 0.5, G * n_per_group))

    xbar = np.array([x[group_idx == g].mean() for g in range(G)])
    ybar = np.array([y[group_idx == g].mean() for g in range(G)])

    with pm.Model():
        alpha_g = pm.Normal("alpha_g", 0, 5, shape=G)
        beta = pm.Normal("beta", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha_g[group_idx] + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_indiv = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                 random_seed=4, progressbar=False,
                                 compute_convergence_checks=False)

    with pm.Model():
        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 5)
        sigma_agg = pm.HalfNormal("sigma_agg", 2)
        mu = a + b * xbar
        pm.Normal("ybar", mu=mu, sigma=sigma_agg, observed=ybar)
        idata_agg = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=4, progressbar=False,
                               compute_convergence_checks=False)

    beta_indiv = idata_indiv.posterior["beta"].values.flatten()
    b_agg = idata_agg.posterior["b"].values.flatten()
    alpha_mean = idata_indiv.posterior["alpha_g"].values.reshape(-1, G).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    sc = axes[0].scatter(x, y, c=group_idx, cmap="viridis", s=14, alpha=0.55)
    xg = np.linspace(x.min(), x.max(), 50)
    for g in rng.choice(G, size=6, replace=False):
        mask = group_idx == g
        xr = np.linspace(x[mask].min(), x[mask].max(), 20)
        axes[0].plot(xr, alpha_mean[g] + beta_indiv.mean() * xr, color="black", lw=1.0, alpha=0.7)
    axes[0].scatter(xbar, ybar, color="red", marker="D", s=45, zorder=4, label="群平均")
    axes[0].plot(xg, np.mean(idata_agg.posterior["a"].values) + b_agg.mean() * xg,
                 color="red", lw=2, ls="--", label="集計回帰(群平均どうし)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("群内では負の傾き、群平均どうしでは正の傾き")
    axes[0].legend(fontsize=8, loc="upper left")

    labels = ["真の群内効果\n(beta_within)", "個体レベル推定\n(群固定効果あり)", "集計レベル推定\n(群平均どうしの回帰)"]
    means = [beta_within, beta_indiv.mean(), b_agg.mean()]
    errs = [0, beta_indiv.std(), b_agg.std()]
    colors = ["black", COLOR_OK, COLOR_ALT]
    axes[1].bar(labels, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
    axes[1].axhline(0, color="gray", lw=0.8)
    for i, m in enumerate(means):
        axes[1].text(i, m / 2, f"{m:+.2f}", ha="center", va="center", fontsize=10, color="white")
    axes[1].set_ylabel("xの係数")
    axes[1].set_title("係数の符号が反転する")

    fig.suptitle("Ecological Bias: 群レベルの交絡により、個体レベルの真の関係と\n集計(群平均)レベルの見かけの関係が逆符号になる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "ecological_bias.png")
    plt.close(fig)
    print(f"ecological_bias.png saved (true beta_within={beta_within}, "
          f"individual-level estimate={beta_indiv.mean():+.3f}, "
          f"aggregate-level estimate={b_agg.mean():+.3f})")


if __name__ == "__main__":
    plot_jensen_inequality_gap()
    plot_ecological_bias()
