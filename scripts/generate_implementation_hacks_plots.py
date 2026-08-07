"""
techniques/implementation-hacks.md に埋め込む可視化画像を生成するスクリプト。

「相関の強い状態空間パラメータではmean-field/fullrank ADVIとも分散を誤推定
しうる」の実例として、強く相関したGaussianRandomWalk(ローカルレベル
トレンド)を含む状態空間モデルを、NUTS・mean-field ADVI・fullrank ADVIの
3通りでフィットし、sigma_level(トレンドの変化幅)の事後推定を比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_implementation_hacks_plots.py

出力先: assets/implementation-hacks/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "implementation-hacks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_advi_variance_inflation():
    """強く相関したGaussianRandomWalkを含むローカルレベルモデルで、
    mean-field/fullrank ADVIがNUTSに比べsigma_levelの事後分布を
    過大評価する(不確実性を誤推定する)ことを示す。"""

    rng = np.random.default_rng(7)
    T = 80
    true_sigma_level = 0.3
    true_sigma_obs = 1.0

    level = np.cumsum(rng.normal(0, true_sigma_level, T))
    y = level + rng.normal(0, true_sigma_obs, T)

    def build_model():
        with pm.Model() as model:
            sigma_level = pm.HalfNormal("sigma_level", 1.0)
            sigma_obs = pm.HalfNormal("sigma_obs", 2.0)
            x = pm.GaussianRandomWalk("x", sigma=sigma_level,
                                       init_dist=pm.Normal.dist(0, 1), steps=T - 1)
            pm.Normal("y", mu=x, sigma=sigma_obs, observed=y)
        return model

    model_nuts = build_model()
    with model_nuts:
        idata_nuts = pm.sample(2000, tune=1500, chains=4, target_accept=0.95,
                                random_seed=1, progressbar=False,
                                compute_convergence_checks=False)

    model_mf = build_model()
    with model_mf:
        approx_mf = pm.fit(30000, method="advi", random_seed=1, progressbar=False)
        idata_mf = approx_mf.sample(4000)

    model_fr = build_model()
    with model_fr:
        approx_fr = pm.fit(30000, method="fullrank_advi", random_seed=1, progressbar=False)
        idata_fr = approx_fr.sample(4000)

    sigma_nuts = idata_nuts.posterior["sigma_level"].values.flatten()
    sigma_mf = idata_mf.posterior["sigma_level"].values.flatten()
    sigma_fr = idata_fr.posterior["sigma_level"].values.flatten()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    datasets = [
        ("NUTS", sigma_nuts, COLOR_OK),
        ("mean-field ADVI", sigma_mf, COLOR_DIVERGENT),
        ("fullrank ADVI", sigma_fr, COLOR_ALT),
    ]
    for label, samples, color in datasets:
        ax.hist(samples, bins=60, density=True, color=color, alpha=0.5, label=f"{label}(平均={samples.mean():.3f})")
    ax.axvline(true_sigma_level, color="black", lw=1.5, ls="--", label=f"真の値={true_sigma_level}")
    ax.set_xlabel("sigma_level の事後分布")
    ax.set_ylabel("density")
    ax.set_title("強く相関したGaussianRandomWalkでは\nmean-field/fullrank ADVIともsigma_levelを過大評価しうる")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "advi_variance_inflation.png")
    plt.close(fig)

    print(f"advi_variance_inflation.png saved "
          f"(真値={true_sigma_level}, NUTS平均={sigma_nuts.mean():.3f}, "
          f"mean-field平均={sigma_mf.mean():.3f}, fullrank平均={sigma_fr.mean():.3f})")


if __name__ == "__main__":
    plot_advi_variance_inflation()
