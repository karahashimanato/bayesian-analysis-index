"""
techniques/diagnostics.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際にサンプリング(NUTS/ADVI/Metropolis)を実行し、診断ワークフロー
で使う3つの切り分けポイントを実データで描画する:
  1. ADVI(mean-field変分推論)とNUTSの不確実性の乖離
  2. チェーン長不足と真の多峰性のr_hatの挙動の違い
  3. Divergent pointsの分布パターン(局所集中 vs 分散)

実行方法:
    source .venv/bin/activate
    python scripts/generate_diagnostics_plots.py

出力先: assets/diagnostics/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_advi_vs_nuts():
    """相関の強い2パラメータ posteriorで、mean-field ADVIがNUTSより
    不確実性を過小評価する様子を比較する(Ridge型非識別性の応用)。"""

    true_sum = 5.0
    obs_noise = 0.15

    with pm.Model():
        kappa = pm.Normal("kappa", 0.0, 10.0)
        beta = pm.Normal("beta", 0.0, 10.0)
        pm.Normal("obs", kappa + beta, obs_noise, observed=np.array([true_sum]))
        idata_nuts = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                                random_seed=0, progressbar=False)
        approx = pm.fit(n=30000, method="advi", progressbar=False, random_seed=0)
        idata_advi = approx.sample(4000)

    k_n, b_n = idata_nuts.posterior["kappa"].values.flatten(), idata_nuts.posterior["beta"].values.flatten()
    k_a, b_a = idata_advi.posterior["kappa"].values.flatten(), idata_advi.posterior["beta"].values.flatten()
    ratio = k_n.std() / k_a.std()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(k_n, b_n, s=4, alpha=0.2, color=COLOR_OK, label=f"NUTS (SD[κ]={k_n.std():.2f})")
    ax.scatter(k_a, b_a, s=4, alpha=0.35, color=COLOR_ALT, label=f"ADVI mean-field (SD[κ]={k_a.std():.2f})")
    ax.set_xlabel("κ")
    ax.set_ylabel("β")
    ax.set_title(f"ADVI(mean-field) vs NUTS の不確実性比較\n"
                 f"ADVIはκ,βの相関を無視するため、SDを約{ratio:.0f}倍過小評価")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "advi_vs_nuts.png")
    plt.close(fig)
    print(f"advi_vs_nuts.png saved (NUTS SD={k_n.std():.2f}, ADVI SD={k_a.std():.2f}, ratio={ratio:.1f}x)")


def plot_chain_length_vs_multimodality():
    """r_hatが悪化したとき、draws(チェーン長)を伸ばして改善するか否かで
    「チェーン長不足」と「真の多峰性」を切り分けられることを示す。"""

    # (a) チェーン長不足: 単峰・だが小さいstep固定のMetropolisで意図的に混合を遅くする
    def sample_unimodal(n_draws):
        with pm.Model():
            pm.Normal("mu", 0.0, 3.0)
            idata = pm.sample(
                n_draws, tune=150, chains=4, step=pm.Metropolis(scaling=0.05),
                random_seed=3, progressbar=False, compute_convergence_checks=False,
            )
        return float(az.rhat(idata, var_names=["mu"])["mu"].values)

    # (b) 真の多峰性: 周期パラメータの位相(multimodality.pngと同じモデル)
    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, 150)
    true_phase = 0.8
    y = np.sin(t + true_phase) + rng.normal(0, 0.15, size=t.size)
    mode_starts = [0.8 - 2 * np.pi, 0.8, 0.8 + 2 * np.pi, 0.8]

    def sample_multimodal(n_draws):
        with pm.Model():
            phase = pm.Uniform("phase", -2 * np.pi, 4 * np.pi)
            mu = pm.math.sin(t + phase)
            pm.Normal("obs", mu, 0.15, observed=y)
            idata = pm.sample(
                n_draws, tune=1000, chains=4, target_accept=0.8, random_seed=0,
                progressbar=False, initvals=[{"phase": s} for s in mode_starts],
                compute_convergence_checks=False,
            )
        return float(az.rhat(idata, var_names=["phase"])["phase"].values)

    short_draws, long_draws = 150, 3000
    rhat_unimodal_short = sample_unimodal(short_draws)
    rhat_unimodal_long = sample_unimodal(long_draws)
    short_draws_mm, long_draws_mm = 1500, 8000
    rhat_multimodal_short = sample_multimodal(short_draws_mm)
    rhat_multimodal_long = sample_multimodal(long_draws_mm)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    groups = ["チェーン長不足\n(単峰・混合が遅いだけ)", "真の多峰性\n(位相パラメータ)"]
    short_vals = [rhat_unimodal_short, rhat_multimodal_short]
    long_vals = [rhat_unimodal_long, rhat_multimodal_long]
    x = np.arange(2)
    width = 0.32

    ax.bar(x - width / 2, short_vals, width, color=COLOR_DIVERGENT, alpha=0.85,
           label=f"短いchain ({short_draws}/{short_draws_mm} draws)")
    ax.bar(x + width / 2, long_vals, width, color=COLOR_OK, alpha=0.85,
           label=f"長いchain ({long_draws}/{long_draws_mm} draws)")
    ax.axhline(1.01, color="black", linestyle=":", linewidth=1, label="r_hat=1.01(健全の目安)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("r_hat")
    ax.set_title("draws(チェーン長)を伸ばすとr_hatは改善するか?\n"
                  "→ 改善すればチェーン長不足、改善しなければ真の多峰性")
    for xi, sv, lv in zip(x, short_vals, long_vals):
        ax.text(xi - width / 2, sv + 0.03, f"{sv:.2f}", ha="center", fontsize=9)
        ax.text(xi + width / 2, lv + 0.03, f"{lv:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(short_vals + long_vals) * 1.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chain_length_vs_multimodality.png")
    plt.close(fig)
    print(f"chain_length_vs_multimodality.png saved "
          f"(unimodal: {rhat_unimodal_short:.2f}->{rhat_unimodal_long:.2f}, "
          f"multimodal: {rhat_multimodal_short:.2f}->{rhat_multimodal_long:.2f})")


def plot_divergent_points_pattern():
    """同じFunnelモデルをtarget_accept違いでサンプリングし、divergenceの
    現れ方が「局所集中」か「広く分散」かで病理の種類を切り分けられることを示す。"""

    def sample(target_accept):
        with pm.Model():
            log_sigma = pm.Normal("log_sigma", 0.0, 3.0)
            sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma / 2))
            pm.Normal("theta", 0.0, sigma)
            return pm.sample(
                2000, tune=1000, chains=4, target_accept=target_accept,
                random_seed=0, progressbar=False, compute_convergence_checks=False,
            )

    idata_low = sample(0.6)
    idata_high = sample(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, idata, ta, title in [
        (axes[0], idata_low, 0.6, "target_accept=0.6\n(ステップサイズ不足を示唆)"),
        (axes[1], idata_high, 0.99, "target_accept=0.99\n(構造的病理=funnelの首に局所集中)"),
    ]:
        theta = idata.posterior["theta"].values.flatten()
        log_sigma = idata.posterior["log_sigma"].values.flatten()
        div = idata.sample_stats["diverging"].values.flatten()
        n_div = int(div.sum())
        div_ls = log_sigma[div]
        iqr = np.percentile(div_ls, 75) - np.percentile(div_ls, 25) if n_div > 0 else float("nan")

        ax.scatter(theta[~div], log_sigma[~div], s=4, alpha=0.25, color=COLOR_OK)
        ax.scatter(theta[div], log_sigma[div], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
        ax.set_title(f"{title}\ndivergence: {n_div}/{len(div)}、divergence位置のIQR(log σ)={iqr:.2f}")
        ax.set_xlabel("θ")
        lo, hi = np.percentile(theta, [0.5, 99.5])
        pad = (hi - lo) * 0.1
        ax.set_xlim(lo - pad, hi + pad)

    axes[0].set_ylabel("log(σ)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Divergent pointsの分布パターン: 分散(左) vs 局所集中(右)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "divergent_points_pattern.png")
    plt.close(fig)
    print("divergent_points_pattern.png saved")


if __name__ == "__main__":
    plot_advi_vs_nuts()
    plot_chain_length_vs_multimodality()
    plot_divergent_points_pattern()
