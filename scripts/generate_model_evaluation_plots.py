"""
techniques/model-evaluation.md に埋め込む可視化画像を生成するスクリプト。

ローカルレベル(GaussianRandomWalk)モデルを実際にPyMCでサンプリングし、
評価期間nを伸ばすと累積効果の信用区間幅がn^3のオーダーで急拡大する様子を描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_model_evaluation_plots.py

出力先: assets/model-evaluation/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "model-evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_cumulative_effect_variance_growth():
    """局所レベル(ランダムウォーク)モデルの事後sigmaを実際にサンプリングし、
    累積効果の信用区間幅が評価期間nに対してn^1.5(分散はn^3)で拡大することを示す。"""

    rng = np.random.default_rng(0)
    T = 200
    true_sigma_level = 0.5
    obs_sigma = 1.0

    level = np.cumsum(rng.normal(0, true_sigma_level, size=T))
    y = level + rng.normal(0, obs_sigma, size=T)

    with pm.Model():
        sigma_level = pm.HalfNormal("sigma_level", 1.0)
        level_rw = pm.GaussianRandomWalk("level_rw", sigma=sigma_level, shape=T)
        pm.Normal("obs", level_rw, obs_sigma, observed=y)
        idata = pm.sample(1500, tune=2000, chains=4, target_accept=0.95,
                           random_seed=0, progressbar=False)

    sigma_draws = idata.posterior["sigma_level"].values.flatten()

    horizons = np.array([7, 14, 30, 60, 90, 147])
    n_sim = 4000
    widths = []
    rng2 = np.random.default_rng(1)
    for n in horizons:
        sigmas = rng2.choice(sigma_draws, size=n_sim)
        increments = rng2.normal(0, 1, size=(n_sim, n)) * sigmas[:, None]
        level_paths = np.cumsum(increments, axis=1)
        cumulative_effect = level_paths.sum(axis=1)
        lo, hi = np.percentile(cumulative_effect, [2.5, 97.5])
        widths.append(hi - lo)
    widths = np.array(widths)

    # 理論式: cumulative effectの分散 = sigma^2 * n(n+1)(2n+1)/6 (Var[level_t]=sigma^2 t の和)
    sigma_mean = sigma_draws.mean()
    theory_width = 2 * 1.96 * sigma_mean * np.sqrt(horizons * (horizons + 1) * (2 * horizons + 1) / 6)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(horizons, widths, "o-", color=COLOR_OK, label="posterior sigmaからのMonteCarlo幅(95%CI)")
    ax.loglog(horizons, theory_width, "--", color=COLOR_ALT,
              label=r"理論式: $2\times1.96\sigma\sqrt{n(n{+}1)(2n{+}1)/6}$")
    ax.set_xlabel("評価期間 n(日)")
    ax.set_ylabel("累積効果の95%信用区間幅")
    ratio = widths[-1] / widths[0]
    ax.set_title(f"累積効果の信用区間幅は評価期間nの1.5乗(分散はn³)で拡大\n"
                 f"n={horizons[0]}日→{horizons[-1]}日で幅は約{ratio:.0f}倍")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(n) for n in horizons])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cumulative_effect_variance_growth.png")
    plt.close(fig)
    print(f"cumulative_effect_variance_growth.png saved "
          f"(width[{horizons[0]}]={widths[0]:.2f}, width[{horizons[-1]}]={widths[-1]:.2f}, ratio={ratio:.1f}x, "
          f"sigma_level posterior mean={sigma_mean:.3f})")


if __name__ == "__main__":
    plot_cumulative_effect_variance_growth()
