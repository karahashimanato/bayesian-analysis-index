"""
tools/posterior-pathologies.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際に NUTS サンプリングを実行し、divergence や r_hat といった
本物の診断結果に基づいて Funnel・Ridge型非識別性・マルチモダリティの
3病理を描画する。

実行方法:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r scripts/requirements.txt
    python scripts/generate_pathology_plots.py

出力先: assets/pathologies/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "pathologies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 病理ドキュメント全体で色の意味を統一する
COLOR_OK = "#4C72B0"       # 通常のドロー(非発散)
COLOR_DIVERGENT = "#D55E00"  # divergence を起こしたドロー
COLOR_CHAIN = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # chainごとの色

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "font.family": "IPAGothic",  # 日本語ラベルの文字化け(tofu)を防ぐ(bold書体は無いため通常太さを使用)
        "axes.unicode_minus": False,
    }
)


def plot_funnel():
    """中心化 vs 非中心化パラメータ化での Neal's funnel を比較する。

    尤度は付けず、階層モデルの事前分布そのものが持つ幾何学的構造
    (θ ~ Normal(μ, σ)、σ = exp(log_sigma/2))だけで funnel が
    生じることを示す(古典的な Neal's funnel の定式化)。
    """

    with pm.Model() as centered:
        mu = pm.Normal("mu", 0.0, 1.0)
        log_sigma = pm.Normal("log_sigma", 0.0, 3.0)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma / 2))
        theta = pm.Normal("theta", mu, sigma)
        idata_centered = pm.sample(
            2000, tune=1000, chains=4, target_accept=0.8,
            random_seed=0, progressbar=False,
        )

    with pm.Model() as noncentered:
        mu = pm.Normal("mu", 0.0, 1.0)
        log_sigma = pm.Normal("log_sigma", 0.0, 3.0)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma / 2))
        offset = pm.Normal("offset", 0.0, 1.0)
        theta = pm.Deterministic("theta", mu + sigma * offset)
        idata_noncentered = pm.sample(
            2000, tune=1000, chains=4, target_accept=0.8,
            random_seed=0, progressbar=False,
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, idata, title in [
        (axes[0], idata_centered, "中心化パラメータ化\n(θ ~ Normal(μ, σ))"),
        (axes[1], idata_noncentered, "非中心化パラメータ化\n(θ = μ + σ・offset)"),
    ]:
        log_sigma = idata.posterior["log_sigma"].values.flatten()
        theta = idata.posterior["theta"].values.flatten()
        div = idata.sample_stats["diverging"].values.flatten()
        n_div = int(div.sum())
        n_total = div.size

        ax.scatter(theta[~div], log_sigma[~div], s=4, alpha=0.3, color=COLOR_OK, label="通常のドロー")
        ax.scatter(theta[div], log_sigma[div], s=14, alpha=0.9, color=COLOR_DIVERGENT, label="divergence")
        ax.set_title(f"{title}\ndivergence: {n_div}/{n_total}")
        ax.set_xlabel("θ")
        # ごく少数の極端な外れ値(裾)に軸スケールを引っ張られないよう、
        # 分位点ベースでクリップして漏斗の形状を見やすくする
        lo, hi = np.percentile(theta, [0.5, 99.5])
        pad = (hi - lo) * 0.1
        ax.set_xlim(lo - pad, hi + pad)

    axes[0].set_ylabel("log(σ)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Funnel(漏斗状の病理、Neal's funnel)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "funnel.png")
    plt.close(fig)
    print(f"funnel.png saved (centered div={int(idata_centered.sample_stats['diverging'].values.sum())}, "
          f"noncentered div={int(idata_noncentered.sample_stats['diverging'].values.sum())})")


def plot_ridge():
    """比が意味を持つ2パラメータで生じる ridge 型非識別性を可視化する。"""

    true_sum = 5.0
    obs_noise = 0.15

    with pm.Model() as model:
        kappa = pm.Normal("kappa", 0.0, 10.0)
        beta = pm.Normal("beta", 0.0, 10.0)
        pm.Normal("obs", kappa + beta, obs_noise, observed=np.array([true_sum]))
        idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=0, progressbar=False)

    kappa_s = idata.posterior["kappa"].values.flatten()
    beta_s = idata.posterior["beta"].values.flatten()
    corr = np.corrcoef(kappa_s, beta_s)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(kappa_s, beta_s, s=4, alpha=0.25, color=COLOR_OK)
    xs = np.linspace(kappa_s.min(), kappa_s.max(), 10)
    ax.plot(xs, true_sum - xs, color=COLOR_DIVERGENT, linewidth=1.5, linestyle="--",
            label=f"κ + β = {true_sum:g}(尤度が実質フラットな方向)")
    ax.set_xlabel("κ")
    ax.set_ylabel("β")
    ax.set_title(f"Ridge型非識別性\n(κとβの相関係数: {corr:.3f})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ridge.png")
    plt.close(fig)
    print(f"ridge.png saved (corr={corr:.3f})")


def plot_multimodality():
    """周期パラメータ(位相)に起因するマルチモダリティを可視化する。

    長い時系列にすることで位相方向の尤度の谷を深くし、chainごとに
    初期値を別々の山の近くへ置くことで、実際に「chainが別の峰から
    抜け出せなくなる」状態(r_hatの悪化)を再現する。
    """

    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, 150)  # 長い時系列ほど位相方向の谷が深くなる
    true_phase = 0.8
    y = np.sin(t + true_phase) + rng.normal(0, 0.15, size=t.size)

    # Uniform(-2π, 4π) の範囲には 0.8, 0.8-2π, 0.8+2π の3つの山が存在する
    mode_starts = [0.8 - 2 * np.pi, 0.8, 0.8 + 2 * np.pi, 0.8]

    with pm.Model() as model:
        phase = pm.Uniform("phase", -2 * np.pi, 4 * np.pi)
        mu = pm.math.sin(t + phase)
        pm.Normal("obs", mu, 0.15, observed=y)
        idata = pm.sample(
            1500, tune=1000, chains=4, target_accept=0.8,
            random_seed=0, progressbar=False,
            initvals=[{"phase": s} for s in mode_starts],
        )

    rhat = float(az.rhat(idata, var_names=["phase"])["phase"].values)
    div = int(idata.sample_stats["diverging"].values.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    phase_by_chain = idata.posterior["phase"].values  # shape (chain, draw)
    for c in range(phase_by_chain.shape[0]):
        axes[0].plot(phase_by_chain[c], color=COLOR_CHAIN[c % len(COLOR_CHAIN)],
                     linewidth=0.6, alpha=0.8, label=f"chain {c}")
    axes[0].set_xlabel("draw")
    axes[0].set_ylabel("phase")
    axes[0].set_title("トレースプロット\n(chainごとに異なる山へ収束)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    for c in range(phase_by_chain.shape[0]):
        axes[1].hist(phase_by_chain[c], bins=40, color=COLOR_CHAIN[c % len(COLOR_CHAIN)],
                     alpha=0.5, label=f"chain {c}")
    axes[1].axvline(true_phase, color="black", linestyle=":", linewidth=1, label="真の位相 (mod 2π)")
    axes[1].axvline(true_phase + 2 * np.pi, color="black", linestyle=":", linewidth=1)
    axes[1].axvline(true_phase - 2 * np.pi, color="black", linestyle=":", linewidth=1)
    axes[1].set_xlabel("phase")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"chain別ヒストグラム\nr_hat={rhat:.2f}, divergence={div}")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("マルチモダリティ(多峰性): 位相パラメータの周期的多峰性", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "multimodality.png")
    plt.close(fig)
    print(f"multimodality.png saved (r_hat={rhat:.2f}, divergence={div})")


if __name__ == "__main__":
    plot_funnel()
    plot_ridge()
    plot_multimodality()
