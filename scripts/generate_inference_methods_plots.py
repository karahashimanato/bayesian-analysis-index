"""
tools/inference-methods.md に埋め込む可視化画像を生成するスクリプト。

階層Thompson Samplingが「事後分布の確信が強まりすぎて探索が止まる」
ロックインを起こすメカニズムを、早期探索が偏ったスナップショットに対する
「次にその腕が選ばれる確率(P(argmax))」の違いとして、独立モデルと比較して示す。

実行方法:
    source .venv/bin/activate
    python scripts/generate_inference_methods_plots.py

出力先: assets/inference-methods/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "inference-methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_thompson_sampling_lockin():
    """階層Thompson Samplingが、探索不足の腕を早期の不運な観測データだけで
    「有望でない」と過信し、探索を止めてしまう(ロックイン)ことを示す。"""

    n_arms = 5
    true_p = np.array([0.10, 0.10, 0.10, 0.10, 0.18])
    best_arm = 4
    arm_labels = [f"腕{i}" for i in range(n_arms)]

    rng = np.random.default_rng(8)
    trials = np.array([500, 500, 500, 500, 20])  # 真に最良の腕4だけ試行が少ない
    succ = rng.binomial(trials, true_p)

    with pm.Model():
        mu = pm.Beta("mu", 2, 2)
        kappa = pm.Gamma("kappa", alpha=4.0, beta=0.04)  # 「腕同士は似ている」という事前の想定(平均100)
        alpha = pm.Deterministic("alpha", mu * kappa)
        beta = pm.Deterministic("beta", (1 - mu) * kappa)
        p = pm.Beta("p", alpha=alpha, beta=beta, shape=n_arms)
        pm.Binomial("y", n=trials, p=p, observed=succ)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
    p_hier = idata.posterior["p"].values.reshape(-1, n_arms)
    prob_best_hier = (p_hier.argmax(axis=1)[:, None] == np.arange(n_arms)).mean(axis=0)

    rng_indep = np.random.default_rng(0)
    n_mc = p_hier.shape[0]
    p_indep = rng_indep.beta(1 + succ, 1 + (trials - succ), size=(n_mc, n_arms))
    prob_best_indep = (p_indep.argmax(axis=1)[:, None] == np.arange(n_arms)).mean(axis=0)

    false_leader = int(np.argmax(prob_best_hier[:4]))  # 階層モデルが誤って推す腕

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    x = np.arange(n_arms)
    width = 0.35
    b1 = axes[0].bar(x - width / 2, prob_best_hier, width, color=COLOR_DIVERGENT, label="階層モデル")
    b2 = axes[0].bar(x + width / 2, prob_best_indep, width, color=COLOR_OK, label="独立モデル")
    for bars in (b1, b2):
        for rect in bars:
            axes[0].annotate(f"{rect.get_height():.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                              ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lbl}\nCTR{p:.0%}\nn={t}" for lbl, p, t in zip(arm_labels, true_p, trials)],
                             fontsize=8)
    axes[0].set_ylabel("P(次にこの腕が選ばれる) = P(argmax p)")
    axes[0].set_title(f"独立モデルは真に最良の腕{best_arm}を推すが、\n階層モデルは試行数の多い腕{false_leader}を誤って推す")
    axes[0].legend(fontsize=9)

    for p_samples, color, label, ls in [
        (p_indep[:, best_arm], COLOR_OK, "独立モデル", "-"),
        (p_hier[:, best_arm], COLOR_DIVERGENT, "階層モデル", "-"),
    ]:
        axes[1].hist(p_samples, bins=50, density=True, color=color, alpha=0.5, label=label)
    axes[1].axvline(true_p[best_arm], color="black", lw=1.2, ls="--", label=f"真の値={true_p[best_arm]:.2f}")
    axes[1].set_xlabel("腕4のCTR事後分布")
    axes[1].set_ylabel("density")
    axes[1].set_title("階層モデルは他の腕に引っ張られ\n事後分布が狭く低い位置に固まる")
    axes[1].legend(fontsize=9)

    fig.suptitle("階層Thompson Samplingのロックイン: 探索不足の腕への過信", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "thompson_sampling_lockin.png")
    plt.close(fig)

    print(f"thompson_sampling_lockin.png saved "
          f"(P(腕4選択)| hier={prob_best_hier[best_arm]:.3f}, indep={prob_best_indep[best_arm]:.3f}; "
          f"腕4事後平均| hier={p_hier[:,best_arm].mean():.3f}, indep={p_indep[:,best_arm].mean():.3f})")


if __name__ == "__main__":
    plot_thompson_sampling_lockin()
