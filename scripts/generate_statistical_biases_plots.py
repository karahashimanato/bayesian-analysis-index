"""
tools/statistical-biases.md に埋め込む可視化画像を生成するスクリプト。

ベイズロジスティック回帰をPyMCで実際にサンプリングし、Jensen不等式により
「係数の点推定から素朴に計算した予測確率」と「事後分布全体を通した予測確率の平均」
がどれだけズレるかを描画する。

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


if __name__ == "__main__":
    plot_jensen_inequality_gap()
