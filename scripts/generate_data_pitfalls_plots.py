"""
techniques/data-pitfalls.md に埋め込む可視化画像を生成するスクリプト。

「ランダム化されていない観測変数から因果を主張しない」の実例として、
交絡変数が観測変数(X)と結果(Y)の両方に影響する状況で、
素朴な回帰(Xのみ)・交絡変数で調整した回帰・ランダム化されたX、の
3通りのベイズ線形回帰の事後分布を比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_data_pitfalls_plots.py

出力先: assets/data-pitfalls/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "data-pitfalls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_confounding_bias():
    """交絡変数Zが観測変数X(広告接触回数)と結果Y(CVR)の両方に影響する場合、
    Xのみの素朴な回帰は真の効果を過大評価するが、Zで調整するかXをランダム化
    すれば真の効果を回復できることを示す。"""

    rng = np.random.default_rng(0)
    n = 400
    true_beta_x = 0.2   # Xの真の因果効果(小さい)
    beta_z = 1.5         # 交絡変数Zの効果(大きい)
    noise_sd = 0.5

    # 観測データ: Z(顧客の関心度)がXの割り当て(ターゲティング)にも影響する
    z_obs = rng.normal(0, 1, n)
    x_obs = 0.8 * z_obs + rng.normal(0, 0.6, n)  # Xはランダム化されていない
    y_obs = true_beta_x * x_obs + beta_z * z_obs + rng.normal(0, noise_sd, n)

    # ランダム化データ: Xの割り当てがZと独立
    z_rand = rng.normal(0, 1, n)
    x_rand = rng.normal(0, 1, n)  # Zと無相関にランダム割り当て
    y_rand = true_beta_x * x_rand + beta_z * z_rand + rng.normal(0, noise_sd, n)

    def fit_naive(x, y):
        with pm.Model():
            b0 = pm.Normal("b0", 0, 5)
            bx = pm.Normal("bx", 0, 5)
            sigma = pm.HalfNormal("sigma", 2)
            pm.Normal("y", mu=b0 + bx * x, sigma=sigma, observed=y)
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["bx"].values.flatten()

    def fit_adjusted(x, z, y):
        with pm.Model():
            b0 = pm.Normal("b0", 0, 5)
            bx = pm.Normal("bx", 0, 5)
            bz = pm.Normal("bz", 0, 5)
            sigma = pm.HalfNormal("sigma", 2)
            pm.Normal("y", mu=b0 + bx * x + bz * z, sigma=sigma, observed=y)
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["bx"].values.flatten()

    bx_naive = fit_naive(x_obs, y_obs)
    bx_adjusted = fit_adjusted(x_obs, z_obs, y_obs)
    bx_randomized = fit_naive(x_rand, y_rand)  # ランダム化済みなのでXのみでよい

    fig, ax = plt.subplots(figsize=(8, 5.5))
    datasets = [
        ("観測変数(素朴にXのみ回帰)\n= 交絡Zで歪んだ因果主張", bx_naive, COLOR_DIVERGENT),
        ("観測変数(Zで調整して回帰)", bx_adjusted, COLOR_ALT),
        ("ランダム化変数(Xのみ回帰)", bx_randomized, COLOR_OK),
    ]
    positions = [2, 1, 0]
    for (label, samples, color), pos in zip(datasets, positions):
        parts = ax.violinplot([samples], positions=[pos], orientation="horizontal", widths=0.7,
                               showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color(color)
        mean = samples.mean()
        ax.annotate(f"事後平均={mean:.3f}", (mean, pos), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, color=color)

    ax.axvline(true_beta_x, color="black", lw=1.2, ls="--", label=f"真の因果効果={true_beta_x}")
    ax.set_yticks(positions)
    ax.set_yticklabels([d[0] for d in datasets], fontsize=9.5)
    ax.set_xlabel("Xの回帰係数の事後分布")
    ax.set_title("交絡変数のあるXを素朴に回帰すると真の効果を過大評価する\n"
                  "(Zで調整するか、Xをランダム化すれば真の効果を回復できる)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confounding_bias.png")
    plt.close(fig)

    print(f"confounding_bias.png saved "
          f"(真の効果={true_beta_x}, 素朴={bx_naive.mean():.3f}, "
          f"Z調整={bx_adjusted.mean():.3f}, ランダム化={bx_randomized.mean():.3f})")


if __name__ == "__main__":
    plot_confounding_bias()
