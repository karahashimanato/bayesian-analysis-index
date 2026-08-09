"""
tools/missing-data.md に埋め込む可視化画像を生成するスクリプト。

  1. MCAR/MAR/MNAR: 3つの欠測メカニズムを合成データで再現し、
     値と欠測確率の関係が質的に異なることを示す。
  2. フルベイズ同時モデル: PyMCのマスク配列による自動補完を使い、
     完全ケース分析(CC)と比べて周辺平均のバイアスがMCAR/MARでは
     大きく減るが、MNARでは残ることを示す。

実行方法:
    source .venv/bin/activate
    python scripts/generate_missing_data_plots.py

出力先: assets/missing-data/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "missing-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_mcar_mar_mnar_mechanisms():
    """MCAR/MAR/MNARの3つの欠測メカニズムを合成データで再現し、
    値yと欠測確率の関係が質的に異なることを示す。"""

    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 1, n)

    p_mcar = np.full(n, 0.35)
    p_mar = 1 / (1 + np.exp(-(x - 0.5)))
    p_mnar = 1 / (1 + np.exp(-(y - y.mean())))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)
    for ax, (name, p) in zip(axes, [
        ("MCAR\n(欠測確率はyにもxにも依存しない)", p_mcar),
        ("MAR\n(欠測確率は観測済みxに依存)", p_mar),
        ("MNAR\n(欠測確率はy自体に依存)", p_mnar),
    ]):
        miss = rng.uniform(0, 1, n) < p
        ax.scatter(x[~miss], y[~miss], s=10, alpha=0.4, color=COLOR_OK, label="観測")
        ax.scatter(x[miss], y[miss], s=10, alpha=0.5, color=COLOR_DIVERGENT, label="欠測")
        ax.set_xlabel("x(観測済み)")
        ax.set_title(name, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("y")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("MCAR/MAR/MNAR: yの値と欠測/観測の分布の関係", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "mcar_mar_mnar_mechanisms.png")
    plt.close(fig)

    print("mcar_mar_mnar_mechanisms.png saved")


def plot_full_bayes_bias_comparison():
    """完全ケース分析(CC)とフルベイズ同時モデル(PyMCのマスク配列による
    自動補完)の周辺平均バイアスを、MCAR/MAR/MNARの3条件で比較する。"""

    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    true_beta0, true_beta1 = 2.0, 1.5
    y = true_beta0 + true_beta1 * x + rng.normal(0, 1, n)
    true_mean = y.mean()

    miss_mcar = rng.uniform(0, 1, n) < 0.3
    p_mar = 1 / (1 + np.exp(-(x - 0.5)))
    miss_mar = rng.uniform(0, 1, n) < p_mar
    p_mnar = 1 / (1 + np.exp(-(y - y.mean())))
    miss_mnar = rng.uniform(0, 1, n) < p_mnar

    def fit_full_bayes(miss, seed):
        y_masked = np.ma.masked_array(y, mask=miss)
        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 10)
            beta1 = pm.Normal("beta1", 0, 10)
            sigma = pm.HalfNormal("sigma", 5)
            mu = beta0 + beta1 * x
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_masked)
            idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        y_mis_mean = idata.posterior["y_obs_unobserved"].values.reshape(-1, miss.sum()).mean(axis=0)
        y_full_est = y.copy()
        y_full_est[miss] = y_mis_mean
        return y_full_est.mean()

    results = {}
    for name, miss, seed in [("MCAR", miss_mcar, 1), ("MAR", miss_mar, 2), ("MNAR", miss_mnar, 3)]:
        cc_bias = y[~miss].mean() - true_mean
        fb_bias = fit_full_bayes(miss, seed) - true_mean
        results[name] = (cc_bias, fb_bias, miss.mean())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = list(results.keys())
    cc_biases = [results[k][0] for k in labels]
    fb_biases = [results[k][1] for k in labels]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, cc_biases, width=width, color=COLOR_DIVERGENT, label="完全ケース分析(CC)")
    ax.bar(x_pos + width / 2, fb_biases, width=width, color=COLOR_OK, label="フルベイズ同時モデル")
    ax.axhline(0, color="black", lw=0.8)
    for i, (cc, fb) in enumerate(zip(cc_biases, fb_biases)):
        ax.annotate(f"{cc:+.3f}", (i - width / 2, cc), xytext=(0, 4 if cc >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{fb:+.3f}", (i + width / 2, fb), xytext=(0, 4 if fb >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{k}\n(欠測率{results[k][2]:.0%})" for k in labels])
    ax.set_ylabel("周辺平均の推定バイアス")
    ax.set_title("フルベイズ同時モデルはMAR下でCCよりバイアスを大きく減らすが\nMNAR下では改善するもののバイアスが残る")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_bayes_bias_comparison.png")
    plt.close(fig)

    print("full_bayes_bias_comparison.png saved (" +
          ", ".join(f"{k}: CC={results[k][0]:+.3f} FullBayes={results[k][1]:+.3f}" for k in labels) + ")")


if __name__ == "__main__":
    plot_mcar_mar_mnar_mechanisms()
    plot_full_bayes_bias_comparison()
