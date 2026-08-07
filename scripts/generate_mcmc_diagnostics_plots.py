"""
tools/mcmc-diagnostics.md に埋め込む可視化画像を生成するスクリプト。

変化点(changepoint)モデルを実際にPyMCでサンプリングし、離散変数(変化点位置tau、
Compound StepでMetropolis法が使われる)のESSが連続変数(Poisson率)より
低くなりやすいことを描画する。

また、Neal's funnel(急峻に曲がった事後分布)をtarget_accept=0.8/0.99の
2通りでサンプリングし、divergence数とESSがどうトレードオフするかを描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_mcmc_diagnostics_plots.py

出力先: assets/mcmc-diagnostics/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from plot_style import COLOR_ALT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "mcmc-diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_discrete_ess_gap():
    """変化点モデル(tau=離散, lambda1/lambda2=連続)で、同じdraws数でも
    離散変数のESSが連続変数より1桁近く低くなることを示す。"""

    rng = np.random.default_rng(2)
    T = 100
    true_tau = 45
    true_lambda1, true_lambda2 = 3.0, 8.0
    counts = np.concatenate([
        rng.poisson(true_lambda1, size=true_tau),
        rng.poisson(true_lambda2, size=T - true_tau),
    ])
    idx = np.arange(T)

    with pm.Model():
        tau = pm.DiscreteUniform("tau", lower=0, upper=T - 1)
        lambda1 = pm.Exponential("lambda1", 1.0)
        lambda2 = pm.Exponential("lambda2", 1.0)
        rate = pt.switch(idx < tau, lambda1, lambda2)
        pm.Poisson("obs", rate, observed=counts)
        idata = pm.sample(4000, tune=2000, chains=4, random_seed=0,
                           progressbar=False, compute_convergence_checks=False)

    ess = az.ess(idata, var_names=["tau", "lambda1", "lambda2"])
    ess_tau = float(ess["tau"].values)
    ess_l1 = float(ess["lambda1"].values)
    ess_l2 = float(ess["lambda2"].values)
    total_draws = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    names = ["tau\n(離散, Metropolis)", "lambda1\n(連続, NUTS)", "lambda2\n(連続, NUTS)"]
    values = [ess_tau, ess_l1, ess_l2]
    colors = [COLOR_ALT, COLOR_OK, COLOR_OK]
    bars = ax.bar(names, values, color=colors, alpha=0.85)
    ax.axhline(total_draws, color="black", linestyle=":", linewidth=1,
               label=f"総draws数 = {total_draws}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + total_draws * 0.02,
                f"{v:.0f}\n({v/total_draws*100:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("ESS (ess_bulk)")
    ax.set_title(f"変化点モデル: 離散変数(tau)のESSは連続変数の約"
                 f"{ess_l1/ess_tau:.0f}分の1\n(PyMCがCompound StepでtauにMetropolis法を使うため)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "discrete_ess_gap.png")
    plt.close(fig)
    print(f"discrete_ess_gap.png saved (ESS tau={ess_tau:.0f}, lambda1={ess_l1:.0f}, "
          f"lambda2={ess_l2:.0f}, total_draws={total_draws})")


def plot_target_accept_tradeoff():
    """Neal's funnel(急峻に曲がった事後分布)をtarget_accept=0.8/0.99の
    2通りでサンプリングし、divergence数の減少とESSの変化を比較する。"""

    D = 9

    def build_model():
        with pm.Model() as model:
            v = pm.Normal("v", 0, 3)
            pm.Normal("x", 0, pt.exp(v / 2), shape=D)
        return model

    results = {}
    for ta in [0.8, 0.99]:
        model = build_model()
        with model:
            idata = pm.sample(2000, tune=1500, chains=4, target_accept=ta,
                               random_seed=5, progressbar=False,
                               compute_convergence_checks=False)
        n_div = int(idata.sample_stats["diverging"].sum())
        ess_v = float(az.ess(idata, var_names=["v"])["v"].values)
        results[ta] = (n_div, ess_v)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    tas = [0.8, 0.99]
    divs = [results[ta][0] for ta in tas]
    esss = [results[ta][1] for ta in tas]
    colors = [COLOR_ALT, COLOR_OK]

    axes[0].bar([str(ta) for ta in tas], divs, color=colors, alpha=0.85)
    axes[0].set_xlabel("target_accept")
    axes[0].set_ylabel("divergence数(全chain合計)")
    axes[0].set_title("divergence数")
    for i, d in enumerate(divs):
        axes[0].text(i, d, f"{d}", ha="center", va="bottom", fontsize=10)

    axes[1].bar([str(ta) for ta in tas], esss, color=colors, alpha=0.85)
    axes[1].set_xlabel("target_accept")
    axes[1].set_ylabel("ESS (v, ess_bulk)")
    axes[1].set_title("ESS")
    for i, e in enumerate(esss):
        axes[1].text(i, e, f"{e:.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Neal's funnelでtarget_accept=0.8→0.99にすると\n"
                  f"divergenceは{divs[0]}→{divs[1]}に減るが、ESSは{esss[0]:.0f}→{esss[1]:.0f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_accept_tradeoff.png")
    plt.close(fig)
    print(f"target_accept_tradeoff.png saved "
          f"(0.8: divergence={divs[0]}, ess={esss[0]:.0f} / "
          f"0.99: divergence={divs[1]}, ess={esss[1]:.0f})")


if __name__ == "__main__":
    plot_discrete_ess_gap()
    plot_target_accept_tradeoff()
