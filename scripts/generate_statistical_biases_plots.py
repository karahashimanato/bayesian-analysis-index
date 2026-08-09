"""
tools/statistical-biases.md に埋め込む可視化画像を生成するスクリプト。

ベイズロジスティック回帰をPyMCで実際にサンプリングし、Jensen不等式により
「係数の点推定から素朴に計算した予測確率」と「事後分布全体を通した予測確率の平均」
がどれだけズレるかを描画する。

また、Ecological Bias(集計データと個体データの乖離)の実例として、
群レベルの交絡変数が個体レベルのxとyの両方に影響する疑似データを生成し、
群固定効果を入れた個体レベル回帰(真の群内効果を復元)と、群平均どうしを
単純回帰した集計レベル回帰(群レベルの交絡に引きずられる)の係数を比較する。

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


def plot_ecological_bias():
    """群レベルの交絡変数z_gが個体レベルのx・yの両方に影響する疑似データで、
    群固定効果を入れた個体レベル回帰(真の群内効果beta_withinを復元)と、
    群平均どうしを単純回帰した集計レベル回帰(符号すら反転しうる)を比較する。"""

    rng = np.random.default_rng(23)
    G = 20
    n_per_group = 25
    beta_within = -0.6
    gamma = 2.5

    z_g = rng.normal(0, 1, G)
    group_idx = np.repeat(np.arange(G), n_per_group)
    within_noise = rng.normal(0, 0.7, G * n_per_group)
    x = 2 * z_g[group_idx] + within_noise
    y = (5 + beta_within * within_noise + gamma * z_g[group_idx]
         + rng.normal(0, 0.5, G * n_per_group))

    xbar = np.array([x[group_idx == g].mean() for g in range(G)])
    ybar = np.array([y[group_idx == g].mean() for g in range(G)])

    with pm.Model():
        alpha_g = pm.Normal("alpha_g", 0, 5, shape=G)
        beta = pm.Normal("beta", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha_g[group_idx] + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_indiv = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                 random_seed=4, progressbar=False,
                                 compute_convergence_checks=False)

    with pm.Model():
        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 5)
        sigma_agg = pm.HalfNormal("sigma_agg", 2)
        mu = a + b * xbar
        pm.Normal("ybar", mu=mu, sigma=sigma_agg, observed=ybar)
        idata_agg = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=4, progressbar=False,
                               compute_convergence_checks=False)

    beta_indiv = idata_indiv.posterior["beta"].values.flatten()
    b_agg = idata_agg.posterior["b"].values.flatten()
    alpha_mean = idata_indiv.posterior["alpha_g"].values.reshape(-1, G).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    sc = axes[0].scatter(x, y, c=group_idx, cmap="viridis", s=14, alpha=0.55)
    xg = np.linspace(x.min(), x.max(), 50)
    for g in rng.choice(G, size=6, replace=False):
        mask = group_idx == g
        xr = np.linspace(x[mask].min(), x[mask].max(), 20)
        axes[0].plot(xr, alpha_mean[g] + beta_indiv.mean() * xr, color="black", lw=1.0, alpha=0.7)
    axes[0].scatter(xbar, ybar, color="red", marker="D", s=45, zorder=4, label="群平均")
    axes[0].plot(xg, np.mean(idata_agg.posterior["a"].values) + b_agg.mean() * xg,
                 color="red", lw=2, ls="--", label="集計回帰(群平均どうし)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("群内では負の傾き、群平均どうしでは正の傾き")
    axes[0].legend(fontsize=8, loc="upper left")

    labels = ["真の群内効果\n(beta_within)", "個体レベル推定\n(群固定効果あり)", "集計レベル推定\n(群平均どうしの回帰)"]
    means = [beta_within, beta_indiv.mean(), b_agg.mean()]
    errs = [0, beta_indiv.std(), b_agg.std()]
    colors = ["black", COLOR_OK, COLOR_ALT]
    axes[1].bar(labels, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
    axes[1].axhline(0, color="gray", lw=0.8)
    for i, m in enumerate(means):
        axes[1].text(i, m / 2, f"{m:+.2f}", ha="center", va="center", fontsize=10, color="white")
    axes[1].set_ylabel("xの係数")
    axes[1].set_title("係数の符号が反転する")

    fig.suptitle("Ecological Bias: 群レベルの交絡により、個体レベルの真の関係と\n集計(群平均)レベルの見かけの関係が逆符号になる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "ecological_bias.png")
    plt.close(fig)
    print(f"ecological_bias.png saved (true beta_within={beta_within}, "
          f"individual-level estimate={beta_indiv.mean():+.3f}, "
          f"aggregate-level estimate={b_agg.mean():+.3f})")


def _kaplan_meier(t_obs, event):
    order = np.argsort(t_obs)
    t_sorted, e_sorted = t_obs[order], event[order]
    event_times = np.unique(t_sorted[e_sorted == 1])
    km_t, km_s = [0.0], [1.0]
    s = 1.0
    for ut in event_times:
        n_risk = np.sum(t_obs >= ut)
        d = np.sum((t_obs == ut) & (event == 1))
        s *= (1 - d / n_risk)
        km_t.append(float(ut))
        km_s.append(s)
    return np.array(km_t), np.array(km_s)


def plot_ipcw_zero_division():
    """打ち切り生存確率G(t)を打ち切りイベントに対するKaplan-Meier推定量として
    実際に計算し、行政打ち切り(全員がt_maxで打ち切られる)によりG(t_max)=0と
    なって1/G(t)がゼロ除算を起こすこと、およびクリップによる回避を示す。"""

    rng = np.random.default_rng(31)
    n = 500
    true_rate = 0.15
    dropout_rate = 0.04  # 追跡離脱による打ち切り(t_max以前にも発生)
    t_event = rng.exponential(1 / true_rate, n)
    t_dropout = rng.exponential(1 / dropout_rate, n)
    t_max = 10.0
    t_raw = np.minimum(t_event, t_dropout)
    event = (t_raw <= t_max).astype(int) * (t_event <= t_dropout).astype(int)
    t_obs = np.minimum(t_raw, t_max)

    # G(t): 打ち切りイベント(event=0)を「イベント」とみなしたKM推定量
    cens_indicator = 1 - event
    km_t, km_G = _kaplan_meier(t_obs, cens_indicator)

    eps = 0.05
    t_eval = np.linspace(0.01, t_max, 300)
    G_at_eval = np.array([km_G[km_t <= t][-1] for t in t_eval])
    weight_raw = 1 / np.clip(G_at_eval, 1e-12, None)
    t_eval_clipped = np.minimum(t_eval, t_max - eps)
    G_clipped = np.array([km_G[km_t <= t][-1] for t in t_eval_clipped])
    weight_clipped = 1 / np.clip(G_clipped, 1e-3, None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.step(km_t, km_G, where="post", color=COLOR_OK, lw=2)
    ax.axvline(t_max, color="black", lw=1, ls=":", label=f"t_max={t_max}")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("打ち切り生存確率 G(t)")
    ax.set_title(f"G(t)は行政打ち切りによりt_maxでちょうど0になる\n(G(t_max)={km_G[-1]:.3f})")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t_eval, weight_raw, color=COLOR_ALT, lw=2, label="1/G(t)(クリップなし)")
    ax.plot(t_eval, weight_clipped, color=COLOR_OK, lw=2, ls="--",
            label=f"1/G(t)(t_maxを{eps}だけクリップ)")
    ax.axvline(t_max, color="black", lw=1, ls=":")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("IPCW重み 1/G(t)")
    ax.set_ylim(0, np.percentile(weight_raw[np.isfinite(weight_raw)], 99) * 1.5)
    ax.set_title("t_max直前で重みが爆発しゼロ除算になる\n(評価時刻をわずかにクリップすると有限に保たれる)")
    ax.legend(fontsize=9)

    fig.suptitle("IPCW: 打ち切り生存確率G(t)の逆数はt_maxでゼロ除算を起こしうる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "ipcw_zero_division.png")
    plt.close(fig)
    print(f"ipcw_zero_division.png saved (G(t_max)={km_G[-1]:.4f}, "
          f"max weight_raw(finite)={weight_raw[np.isfinite(weight_raw)].max():.2f}, "
          f"max weight_clipped={weight_clipped.max():.2f})")


if __name__ == "__main__":
    plot_jensen_inequality_gap()
    plot_ecological_bias()
    plot_ipcw_zero_division()
