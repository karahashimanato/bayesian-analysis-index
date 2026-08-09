"""
techniques/eda.md に埋め込む可視化画像を生成するスクリプト。

モデリング前のEDAで実際に確認する4つの観点(過分散、説明変数間の相関、
欠測パターン、季節性)を、それぞれ合成データで再現して可視化する。
いずれも記述統計・グラフ描画のみで、MCMCサンプリングは使わない。

実行方法:
    source .venv/bin/activate
    python scripts/generate_eda_plots.py

出力先: assets/eda/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_overdispersion_check():
    """負の二項分布的な過分散データ(分散/平均比≈68)を生成し、
    Poisson(平均のみ)を仮定した場合の期待分布と実データの分布が
    大きく食い違うことを示す。"""

    rng = np.random.default_rng(1)
    n = 2000
    mean_target = 8.0
    var_target = 68.0 * mean_target  # 分散/平均比≈68
    # 負の二項分布(Gamma-Poisson)でこの平均・分散を再現する
    p = mean_target / var_target
    r = mean_target * p / (1 - p)
    counts = rng.negative_binomial(r, p, n)

    var_mean_ratio = counts.var() / counts.mean()

    bins = np.arange(0, np.percentile(counts, 99.5) + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(counts, bins=bins, density=True, color=COLOR_DIVERGENT, alpha=0.6,
            label=f"実データ(分散/平均比={var_mean_ratio:.1f})")
    x_pois = np.arange(0, np.percentile(counts, 99.5) + 1)
    ax.plot(x_pois, stats.poisson.pmf(x_pois, counts.mean()), "o-", color=COLOR_OK,
            ms=4, lw=1.5, label=f"Poisson(平均={counts.mean():.1f})が仮定する分布\n(分散/平均比=1)")
    ax.set_xlabel("カウント")
    ax.set_ylabel("density")
    ax.set_title(f"分散/平均比={var_mean_ratio:.0f}の過分散データにPoissonは全く合わない\n"
                 f"(裾が長く伸び、Poissonが想定する範囲を大きく超える)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overdispersion_check.png")
    plt.close(fig)

    print(f"overdispersion_check.png saved (mean={counts.mean():.2f}, var={counts.var():.1f}, "
          f"var/mean={var_mean_ratio:.1f})")


def plot_covariate_correlation_check():
    """3つの候補説明変数(office/electronics/google_brand相当)の相関を
    ヒートマップで可視化し、多重共線性リスクのある組み合わせ(相関0.95)を
    特定して除外する判断根拠を示す。"""

    rng = np.random.default_rng(2)
    n = 200
    electronics = rng.normal(100, 20, n)
    office = electronics * 0.95 + rng.normal(0, electronics.std() * np.sqrt(1 - 0.95 ** 2), n)
    google_brand = rng.normal(50, 15, n) + 0.2 * electronics

    data = np.column_stack([office, electronics, google_brand])
    labels = ["office", "electronics", "google_brand"]
    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, label="相関係数")
    ax.set_title(f"候補説明変数間の相関: office-electronics間が{corr[0,1]:.2f}と高く\n"
                 f"多重共線性のリスクからofficeを候補から除外")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "covariate_correlation_check.png")
    plt.close(fig)

    print(f"covariate_correlation_check.png saved (office-electronics corr={corr[0,1]:.2f}, "
          f"office-google_brand corr={corr[0,2]:.2f}, electronics-google_brand corr={corr[1,2]:.2f})")


def plot_missingness_pattern():
    """人口規模が小さい国ほど欠測率が高いこと(MAR的な関連)と、
    2つの健康指標の欠測フラグ同士が強く共起することを可視化する。"""

    rng = np.random.default_rng(3)
    n = 400
    population = rng.lognormal(mean=np.log(500_000), sigma=1.5, size=n)
    small_pop = population < 300_000

    # 人口が小さい国ほど欠測率が高い(MAR的なメカニズム)
    miss_prob = np.where(small_pop, 0.439, 0.017)
    missing_indicator1 = rng.uniform(0, 1, n) < miss_prob

    # 2つの指標の欠測フラグは強く共起する(相関0.88相当)
    co_occur = rng.uniform(0, 1, n) < 0.88
    missing_indicator2 = np.where(co_occur, missing_indicator1, rng.uniform(0, 1, n) < miss_prob.mean())

    rate_small = missing_indicator1[small_pop].mean()
    rate_large = missing_indicator1[~small_pop].mean()
    flag_corr = np.corrcoef(missing_indicator1.astype(float), missing_indicator2.astype(float))[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(["人口30万人未満", "人口30万人以上"], [rate_small, rate_large],
                color=[COLOR_DIVERGENT, COLOR_OK])
    for i, v in enumerate([rate_small, rate_large]):
        axes[0].annotate(f"{v:.1%}", (i, v), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=10)
    axes[0].set_ylabel("指標1の欠測率")
    axes[0].set_title("人口規模と欠測率の関係\n(小さいほど欠測しやすい)")

    ct = np.array([
        [np.sum(~missing_indicator1 & ~missing_indicator2), np.sum(~missing_indicator1 & missing_indicator2)],
        [np.sum(missing_indicator1 & ~missing_indicator2), np.sum(missing_indicator1 & missing_indicator2)],
    ])
    im = axes[1].imshow(ct, cmap="Purples")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["指標2:観測", "指標2:欠測"])
    axes[1].set_yticklabels(["指標1:観測", "指標1:欠測"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(ct[i, j]), ha="center", va="center",
                         color="white" if ct[i, j] > ct.max() / 2 else "black", fontsize=12)
    axes[1].set_title(f"2指標の欠測フラグの共起\n(相関={flag_corr:.2f})")

    fig.suptitle("欠測パターンの可視化: 誰が・何と関連して欠けているか", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "missingness_pattern.png")
    plt.close(fig)

    print(f"missingness_pattern.png saved (欠測率: 小規模国={rate_small:.1%}, 大規模国={rate_large:.1%}, "
          f"欠測フラグ相関={flag_corr:.2f})")


def plot_seasonality_periodogram():
    """週次季節性を持つ日次カウントデータに対し、曜日別の水準差と
    周期図(パワースペクトル)から周期7日の季節性を検出することを示す。"""

    rng = np.random.default_rng(4)
    n_days = 365
    day_of_week = np.arange(n_days) % 7  # 0=月, ..., 5=土, 6=日
    weekday_level = 365.0
    weekend_level = 236.0
    level = np.where(day_of_week < 5, weekday_level, weekend_level)
    counts = rng.normal(level, 20, n_days)

    weekday_mean = counts[day_of_week < 5].mean()
    weekend_mean = counts[day_of_week >= 5].mean()

    # 周期図(パワースペクトル): FFTで周期成分を検出
    detrended = counts - counts.mean()
    freqs = np.fft.rfftfreq(n_days, d=1.0)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    day_labels = ["月", "火", "水", "木", "金", "土", "日"]
    means_by_day = [counts[day_of_week == d].mean() for d in range(7)]
    colors = [COLOR_OK] * 5 + [COLOR_DIVERGENT] * 2
    axes[0].bar(day_labels, means_by_day, color=colors)
    axes[0].axhline(weekday_mean, color=COLOR_OK, lw=1, ls="--", alpha=0.6)
    axes[0].axhline(weekend_mean, color=COLOR_DIVERGENT, lw=1, ls="--", alpha=0.6)
    axes[0].set_ylabel("平均件数")
    axes[0].set_title(f"曜日別の平均件数\n平日={weekday_mean:.0f}件 vs 週末={weekend_mean:.0f}件")

    mask = (periods > 2) & (periods < 30)
    axes[1].plot(periods[mask], power[mask], color=COLOR_ALT, lw=1.5)
    axes[1].axvline(7, color=COLOR_DIVERGENT, lw=1.5, ls="--", label="周期7日")
    axes[1].set_xlabel("周期(日)")
    axes[1].set_ylabel("パワー")
    axes[1].set_title("周期図: 周期7日に明確なピーク")
    axes[1].legend(fontsize=9)

    fig.suptitle("季節性・周期性のEDA: 曜日別集計と周期図で週次季節性を確認", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "seasonality_periodogram.png")
    plt.close(fig)

    print(f"seasonality_periodogram.png saved (平日={weekday_mean:.0f}件, 週末={weekend_mean:.0f}件, "
          f"周期図ピーク={periods[mask][np.argmax(power[mask])]:.1f}日)")


if __name__ == "__main__":
    plot_overdispersion_check()
    plot_covariate_correlation_check()
    plot_missingness_pattern()
    plot_seasonality_periodogram()
