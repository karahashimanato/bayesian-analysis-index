"""
techniques/implementation-hacks.md に埋め込む可視化画像を生成するスクリプト。

「相関の強い状態空間パラメータではmean-field/fullrank ADVIとも分散を誤推定
しうる」の実例として、強く相関したGaussianRandomWalk(ローカルレベル
トレンド)を含む状態空間モデルを、NUTS・mean-field ADVI・fullrank ADVIの
3通りでフィットし、sigma_level(トレンドの変化幅)の事後推定を比較する。

「区間ごとの滞在時間行列で累積ハザードを積算する」の実例として、Piecewise
Exponentialモデルの exposure_matrix を正しく(打ち切り/イベント区間は端数
だけ)計算した場合と、誤って(最後の区間も丸ごと通過したとみなして)計算
した場合とで、各区間のベースラインハザード推定がどれだけズレるかを比較
する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_implementation_hacks_plots.py

出力先: assets/implementation-hacks/*.png
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.optimize as sopt
from scipy import stats

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "implementation-hacks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_scan_sequences_seasonal_beta():
    """`pytensor.scan`の`sequences`引数を使い、季節変動する感染率β(t)を
    SIRモデルのオイラー法数値積分に組み込む。β(t)をscanへ`sequences`として
    渡し、状態(S,I)の逐次遷移をscanの再帰で計算するモデルを実際にPyMCで
    フィットし、季節振幅Aを観測データから復元できることを示す。"""

    rng = np.random.default_rng(21)
    T = 52
    N_pop = 1000.0
    gamma_true = 0.3
    beta0_true = 0.45
    A_true = 0.35
    I0, S0 = 5.0, 1000.0 - 5.0

    t_idx = np.arange(T)
    beta_t_true = beta0_true * (1 + A_true * np.cos(2 * np.pi * t_idx / 52))

    S, I = S0, I0
    inc_true = []
    for t in range(T):
        new_inf = beta_t_true[t] * S * I / N_pop
        new_rec = gamma_true * I
        S = S - new_inf
        I = I + new_inf - new_rec
        inc_true.append(new_inf)
    inc_true = np.array(inc_true)
    y_obs = rng.poisson(np.clip(inc_true, 1e-6, None))

    with pm.Model():
        beta0 = pm.HalfNormal("beta0", 1.0)
        A = pm.Beta("A", 2.0, 2.0)
        t_seq = pt.arange(T)
        beta_seq = beta0 * (1 + A * pt.cos(2 * np.pi * t_seq / 52))

        def step(beta_t, S_prev, I_prev):
            new_inf = beta_t * S_prev * I_prev / N_pop
            new_rec = gamma_true * I_prev
            S_new = S_prev - new_inf
            I_new = I_prev + new_inf - new_rec
            return S_new, I_new, new_inf

        (S_seq, I_seq, inc_seq), _ = pytensor.scan(
            fn=step,
            sequences=[beta_seq],
            outputs_info=[
                pt.constant(S0, dtype="float64"),
                pt.constant(I0, dtype="float64"),
                None,
            ],
            strict=True,
        )
        pm.Deterministic("beta_seq", beta_seq)
        pm.Deterministic("inc_seq", inc_seq)
        pm.Poisson("y", mu=pt.clip(inc_seq, 1e-6, np.inf), observed=y_obs)
        idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                           random_seed=5, progressbar=False,
                           compute_convergence_checks=False)

    beta_post = idata.posterior["beta_seq"].values.reshape(-1, T)
    inc_post = idata.posterior["inc_seq"].values.reshape(-1, T)
    beta_mean = beta_post.mean(axis=0)
    beta_lo, beta_hi = np.percentile(beta_post, [2.5, 97.5], axis=0)
    inc_mean = inc_post.mean(axis=0)
    inc_lo, inc_hi = np.percentile(inc_post, [2.5, 97.5], axis=0)
    A_post_mean = idata.posterior["A"].values.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(t_idx, beta_t_true, color="black", lw=1.5, ls="--", label="真のβ(t)")
    axes[0].plot(t_idx, beta_mean, color=COLOR_OK, lw=1.5, label="事後平均β(t)")
    axes[0].fill_between(t_idx, beta_lo, beta_hi, color=COLOR_OK, alpha=0.25, label="95%区間")
    axes[0].set_xlabel("週")
    axes[0].set_ylabel("β(t)")
    axes[0].set_title(f"季節性β(t) (真の振幅A={A_true}, 事後平均A={A_post_mean:.3f})")
    axes[0].legend(fontsize=8.5)

    axes[1].scatter(t_idx, y_obs, s=14, color="black", alpha=0.6, label="観測(週次新規感染)")
    axes[1].plot(t_idx, inc_mean, color=COLOR_ALT, lw=1.5, label="事後予測平均")
    axes[1].fill_between(t_idx, inc_lo, inc_hi, color=COLOR_ALT, alpha=0.25, label="95%区間")
    axes[1].set_xlabel("週")
    axes[1].set_ylabel("新規感染者数")
    axes[1].set_title("scanのsequencesでβ(t)を渡した\nSIRモデルの事後予測")
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "scan_sequences_seasonal_beta.png")
    plt.close(fig)

    print(f"scan_sequences_seasonal_beta.png saved "
          f"(真のA={A_true}, 事後平均A={A_post_mean:.3f}, "
          f"beta0事後平均={idata.posterior['beta0'].values.mean():.3f}, "
          f"r_hat(A)={float(pm.stats.rhat(idata)['A'].values):.3f})")


def _km_survival(times, event_indicator, grid):
    """Kaplan-Meier推定量を実装する(event_indicator=1がそのイベント種別の発生)。"""
    order = np.argsort(times)
    times_sorted = times[order]
    events_sorted = event_indicator[order]
    unique_times = np.unique(times_sorted)
    current_s = 1.0
    s_at = {0.0: 1.0}
    for ut in unique_times:
        d = np.sum((times_sorted == ut) & (events_sorted == 1))
        n_at_risk = np.sum(times_sorted >= ut)
        if n_at_risk > 0 and d > 0:
            current_s *= (1 - d / n_at_risk)
        s_at[ut] = current_s
    keys = np.array(sorted(s_at.keys()))
    vals = np.array([s_at[k] for k in keys])
    idxs = np.clip(np.searchsorted(keys, grid, side="right") - 1, 0, len(keys) - 1)
    return vals[idxs]


def plot_censoring_clip_zero_division():
    """打ち切り時刻ちょうど(t_max)に生存者が行政打ち切りで集中していると、
    打ち切り生存確率のKaplan-Meier推定G(t_max)が正確にゼロになり、
    IPCW重み1/G(t)がt_max付近で発散することを実際に計算で再現する。
    t_maxをわずかにクリップするとゼロ除算が回避できることを示す。"""

    rng = np.random.default_rng(3)
    n = 300
    t_max = 24.0
    event_time = rng.exponential(15.0 / np.log(2), n)
    dropout_time = rng.uniform(2, t_max, n)
    has_dropout = rng.uniform(0, 1, n) < 0.15
    censor_time = np.where(has_dropout, dropout_time, t_max)
    observed_time = np.minimum(event_time, censor_time)
    event_ind = (event_time <= censor_time).astype(int)
    censor_ind = 1 - event_ind  # G(t)の推定は「打ち切りをイベントとみなした」KM

    grid = np.linspace(0.01, t_max, 500)
    G = _km_survival(observed_time, censor_ind, grid)
    G_at_tmax = _km_survival(observed_time, censor_ind, np.array([t_max]))[0]
    G_at_clipped = _km_survival(observed_time, censor_ind, np.array([t_max * 0.999]))[0]

    with np.errstate(divide="ignore"):
        ipcw_weight = 1.0 / G

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(grid, G, color=COLOR_OK, lw=1.5)
    axes[0].axvline(t_max, color=COLOR_DIVERGENT, ls="--", lw=1.2, label=f"t_max={t_max}")
    axes[0].scatter([t_max], [G_at_tmax], color=COLOR_DIVERGENT, zorder=5,
                     label=f"G(t_max)={G_at_tmax:.4f}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("G(t) = P(打ち切り時刻 > t)")
    axes[0].set_title("打ち切り分布の生存関数G(t)は\nt_maxで正確にゼロへ落ちる")
    axes[0].legend(fontsize=9)

    axes[1].plot(grid, ipcw_weight, color=COLOR_ALT, lw=1.5)
    axes[1].set_ylim(0, min(ipcw_weight[np.isfinite(ipcw_weight)].max() * 1.2, 500))
    axes[1].axvline(t_max, color=COLOR_DIVERGENT, ls="--", lw=1.2)
    axes[1].axvline(t_max * 0.999, color=COLOR_OK, ls=":", lw=1.5,
                     label=f"clip後t_max×0.999\nG={G_at_clipped:.4f} (1/G={1/G_at_clipped:.2f})")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("IPCW重み 1/G(t)")
    axes[1].set_title("1/G(t)はt_max直前まで緩やかだが、\nt_maxちょうどで一気にゼロ除算になる")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "censoring_clip_zero_division.png")
    plt.close(fig)

    n_at_tmax_censored = int(np.sum((observed_time == t_max) & (censor_ind == 1)))
    print(f"censoring_clip_zero_division.png saved "
          f"(t_max={t_max}ちょうどの打ち切り件数={n_at_tmax_censored}, "
          f"G(t_max)={G_at_tmax:.6f}, G(t_max×0.999)={G_at_clipped:.6f}, "
          f"クリップ後1/G={1/G_at_clipped:.2f})")


def plot_seed_consistency_effect():
    """複数notebookにまたがる「独立推定 vs 階層(縮小推定)」の比較のような
    分析で、乱数シードを固定せずに毎回新しいデータを生成すると、
    run(=notebook実行)ごとの指標(独立推定と階層縮小推定のRMSE差)の
    ばらつきが、本質的な差の大きさと同程度になりうることを示す。"""

    def one_run(rng):
        K = 8
        n_trials = rng.integers(10, 25, K)
        true_p = rng.beta(4, 12, K)
        successes = rng.binomial(n_trials, true_p)
        p_hat = successes / n_trials

        # 経験ベイズ縮小推定(James-Stein型)
        grand_mean = p_hat.mean()
        var_p_hat = np.var(p_hat, ddof=1)
        var_sampling = np.mean(p_hat * (1 - p_hat) / n_trials)
        tau2 = max(var_p_hat - var_sampling, 1e-6)
        shrink_w = tau2 / (tau2 + var_sampling / n_trials.mean())
        p_hier = grand_mean + shrink_w * (p_hat - grand_mean)

        rmse_indep = np.sqrt(np.mean((p_hat - true_p) ** 2))
        rmse_hier = np.sqrt(np.mean((p_hier - true_p) ** 2))
        return rmse_indep - rmse_hier

    n_runs = 30
    seeds = np.arange(n_runs)
    metrics = np.array([one_run(np.random.default_rng(int(s))) for s in seeds])
    cum_mean = np.cumsum(metrics) / np.arange(1, n_runs + 1)
    overall_mean = metrics.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = [COLOR_OK if m > 0 else COLOR_DIVERGENT for m in metrics]
    axes[0].scatter(seeds, metrics, color=colors, s=30, zorder=3)
    axes[0].axhline(0, color="black", lw=1.0, ls=":")
    axes[0].axhline(overall_mean, color=COLOR_ALT, lw=1.5,
                     label=f"全{n_runs}run平均={overall_mean:.4f}")
    axes[0].set_xlabel("シード値(= 固定しなければnotebook実行のたびに変わる)")
    axes[0].set_ylabel("RMSE差(独立推定 − 階層縮小推定)")
    axes[0].set_title("シードを固定しないと、run単発では\n階層推定が「悪化」して見えることさえある")
    axes[0].legend(fontsize=9)

    axes[1].plot(np.arange(1, n_runs + 1), cum_mean, color=COLOR_OK, lw=1.5, marker="o", ms=3)
    axes[1].axhline(overall_mean, color=COLOR_ALT, ls="--", lw=1.2)
    axes[1].axhline(0, color="black", lw=1.0, ls=":")
    axes[1].set_xlabel("累積run数")
    axes[1].set_ylabel("累積平均のRMSE差")
    axes[1].set_title("多数のrunを平均して初めて\n「階層推定が優る」という結論が安定する")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_consistency_effect.png")
    plt.close(fig)

    n_negative = int(np.sum(metrics < 0))
    print(f"seed_consistency_effect.png saved "
          f"({n_runs}run中{n_negative}runで階層推定の方が悪化, "
          f"全run平均のRMSE差={overall_mean:.4f}, 標準偏差={metrics.std():.4f})")


def plot_ecological_bias_two_stage():
    """「集計→ローカル推定」だけでは検出できないecological bias(集計由来の
    バイアス)を、「サンプル抽出→検証」の個体レベル推定と突き合わせることで
    検出できることを示す。群間の交絡(集計レベルの傾向)と、真の個体内
    関係(群内の傾向)が逆符号になる設計で、集計回帰と個体レベル回帰の
    傾きがどれだけ乖離するかを比較する。"""

    rng = np.random.default_rng(9)
    n_groups = 40
    n_per_group = 25
    beta_within_true = 0.8

    group_mean_x = rng.uniform(0, 10, n_groups)
    group_level_effect = -0.6 * group_mean_x + rng.normal(0, 1.0, n_groups)

    x_list, y_list, gid_list = [], [], []
    for g in range(n_groups):
        x_g = group_mean_x[g] + rng.normal(0, 0.7, n_per_group)
        y_g = group_level_effect[g] + beta_within_true * (x_g - group_mean_x[g]) + rng.normal(0, 1.0, n_per_group)
        x_list.append(x_g)
        y_list.append(y_g)
        gid_list.append(np.full(n_per_group, g))
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    gid_all = np.concatenate(gid_list)

    # ステージ1: BigQuery側で集計したような「群平均」だけを使った回帰
    group_mean_y = np.array([y_all[gid_all == g].mean() for g in range(n_groups)])
    beta_agg = np.polyfit(group_mean_x, group_mean_y, 1)[0]

    # ステージ2: 層化ランダムサンプリングで抽出した個体レベルデータで、
    # 群内偏差(群平均を引いた値)に対して回帰し、群間の交絡を分離する
    sample_idx = rng.choice(len(x_all), size=300, replace=False)
    x_s, y_s, g_s = x_all[sample_idx], y_all[sample_idx], gid_all[sample_idx]
    x_s_demeaned = x_s - np.array([x_s[g_s == g].mean() for g in g_s])
    y_s_demeaned = y_s - np.array([y_s[g_s == g].mean() for g in g_s])
    beta_individual = np.polyfit(x_s_demeaned, y_s_demeaned, 1)[0]
    beta_individual_naive = np.polyfit(x_s, y_s, 1)[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_groups))
    for g in range(n_groups):
        ax.scatter(x_list[g], y_list[g], s=10, color=cmap[g], alpha=0.35)
    ax.scatter(group_mean_x, group_mean_y, color="black", marker="D", s=40,
               zorder=5, label="群平均(集計データ)")

    xs = np.linspace(0, 10, 100)
    ax.plot(xs, np.poly1d(np.polyfit(group_mean_x, group_mean_y, 1))(xs),
            color=COLOR_DIVERGENT, lw=2.5, zorder=6,
            label=f"ステージ1: 集計回帰(群平均のみ) 傾き={beta_agg:.3f}")
    for g in np.unique(g_s):
        gx, gy = group_mean_x[g], group_mean_y[g]
        xs_local = np.linspace(gx - 1.2, gx + 1.2, 10)
        ax.plot(xs_local, gy + beta_individual * (xs_local - gx), color=COLOR_OK, lw=1.5, alpha=0.8)
    ax.plot([], [], color=COLOR_OK, lw=2.0,
            label=f"ステージ2: 群内偏差回帰(個体レベル検証) 傾き={beta_individual:.3f}\n"
                  f"(真の個体内傾き={beta_within_true}, 群デミーンなしの素朴回帰={beta_individual_naive:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("集計回帰(群平均)は個体内の真の関係と\n符号すら逆転しうる(ecological bias)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ecological_bias_two_stage.png")
    plt.close(fig)

    print(f"ecological_bias_two_stage.png saved "
          f"(真の個体内傾き={beta_within_true}, "
          f"集計回帰の傾き={beta_agg:.3f}, 群内偏差回帰(検証)の傾き={beta_individual:.3f}, "
          f"群デミーンなし素朴回帰={beta_individual_naive:.3f})")


def _build_grid_adjacency(side):
    """side×sideの格子グラフ(4近傍)の隣接行列を作る。"""
    n = side * side

    def idx(r, c):
        return r * side + c

    W = np.zeros((n, n), dtype=int)
    for r in range(side):
        for c in range(side):
            i = idx(r, c)
            if r > 0:
                W[i, idx(r - 1, c)] = 1
                W[idx(r - 1, c), i] = 1
            if c > 0:
                W[i, idx(r, c - 1)] = 1
                W[idx(r, c - 1), i] = 1
    return W


def plot_icar_closed_form_equivalence():
    """`pm.ICAR`はrandom()を持たずsample_prior_predictiveできないが、
    logpを整理すると精度行列Q/σ²+J/(0.001N)²を持つ多変量正規分布と
    数式的に等価である。この閉形式の精度行列から手動でMVNサンプルを
    生成し、そのサンプルにおける`pm.ICAR`のlogpと、閉形式の二次形式から
    計算したlogpが数値的に完全一致することを実際に検証する。"""

    rng = np.random.default_rng(13)
    side = 5
    N = side * side
    W = _build_grid_adjacency(side)
    D = np.diag(W.sum(axis=1))
    L = D - W
    sigma = 1.0
    zero_sum_stdev = 0.001
    Q_full = L / sigma**2 + np.ones((N, N)) / (zero_sum_stdev * N) ** 2
    cov = np.linalg.inv(Q_full)
    chol = np.linalg.cholesky(cov)

    n_draws = 300
    z = rng.normal(size=(n_draws, N))
    samples = z @ chol.T

    with pm.Model() as model:
        pm.ICAR("phi", W=W, sigma=sigma)
    logp_fn = model.compile_logp()
    icar_logp = np.array([logp_fn({"phi": s}) for s in samples])

    closed_form_logp = (
        -0.5 * np.einsum("ij,jk,ik->i", samples, Q_full, samples)
        - np.log(np.sqrt(2 * np.pi) * zero_sum_stdev * N)
    )
    max_abs_diff = float(np.max(np.abs(icar_logp - closed_form_logp)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(closed_form_logp, icar_logp, s=14, color=COLOR_OK, alpha=0.6)
    lims = [min(closed_form_logp.min(), icar_logp.min()), max(closed_form_logp.max(), icar_logp.max())]
    axes[0].plot(lims, lims, color="black", ls="--", lw=1.0, label="y = x")
    axes[0].set_xlabel("閉形式(Q_full二次形式)から計算したlogp")
    axes[0].set_ylabel("pm.ICARのlogp")
    axes[0].set_title(f"両者は最大絶対差{max_abs_diff:.2e}で完全一致")
    axes[0].legend(fontsize=9)

    marginal_var_closed = np.diag(cov)
    marginal_var_empirical = samples.var(axis=0)
    axes[1].scatter(marginal_var_closed, marginal_var_empirical, s=20, color=COLOR_ALT)
    lims2 = [marginal_var_closed.min() * 0.9, marginal_var_closed.max() * 1.1]
    axes[1].plot(lims2, lims2, color="black", ls="--", lw=1.0)
    axes[1].set_xlabel("閉形式の周辺分散 diag(inv(Q_full))")
    axes[1].set_ylabel(f"閉形式MVNから生成した{n_draws}サンプルの経験分散")
    axes[1].set_title("閉形式MVNからの前向きサンプリングは\n理論分散と整合する")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "icar_closed_form_equivalence.png")
    plt.close(fig)

    print(f"icar_closed_form_equivalence.png saved "
          f"(N={N}, サンプル数={n_draws}, pm.ICARのlogpとの最大絶対差={max_abs_diff:.3e})")


def _generalized_inverse_scaling_factor(W):
    """グラフラプラシアンのMoore-Penrose一般化逆行列の対角成分の
    幾何平均として、BYM2のスケーリング係数を計算する。"""
    N = W.shape[0]
    D = np.diag(W.sum(axis=1))
    Q = D - W
    Q_pinv = np.linalg.pinv(Q)
    return float(np.exp(np.mean(np.log(np.diag(Q_pinv)))))


def plot_bym2_scaling_factor_generalized_inverse():
    """BYM2のスケーリング係数は隣接グラフの構造に依存する量であり、
    別のグラフの値を流用すると`rho`(空間分散の割合)の解釈がずれることを
    示す。(1)複数の格子サイズでスケーリング係数を実際に計算し、
    グラフ構造に依存して変わることを確認し、(2)同一データに対して
    正しいスケーリング係数と、別グラフから借用した誤ったスケーリング
    係数の2通りでBYM2モデルをフィットし、rhoの事後分布のズレを見る。"""

    sides = [4, 6, 8, 10]
    scales = {}
    for side in sides:
        W = _build_grid_adjacency(side)
        scales[side] = _generalized_inverse_scaling_factor(W)

    rng = np.random.default_rng(3)
    side_data = 6
    N = side_data * side_data
    W = _build_grid_adjacency(side_data)
    scale_correct = scales[side_data]
    scale_wrong = scales[4]  # 別のグラフ(4x4格子, N=16)から借用した誤った値

    rr, cc = np.meshgrid(np.arange(side_data), np.arange(side_data), indexing="ij")
    phi_true = 0.15 * (rr.flatten() + cc.flatten()) - 0.15 * (side_data - 1)
    phi_true -= phi_true.mean()
    theta_true = rng.normal(0, 0.15, N)
    E = rng.uniform(50, 150, N)
    counts = rng.poisson(E * np.exp(phi_true + theta_true))

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        rho = pm.Beta("rho", 2, 2)
        theta_star = pm.Normal("theta_star", 0, 1, shape=N)
        phi_star = pm.ICAR("phi_star", W=W, sigma=1)
        combined = sigma * (pt.sqrt(1 - rho) * theta_star + pt.sqrt(rho / scale_correct) * phi_star)
        pm.Poisson("y", mu=E * pt.exp(beta0 + combined), observed=counts)
        idata_correct = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                                   random_seed=1, progressbar=False,
                                   compute_convergence_checks=False)
    rho_post = idata_correct.posterior["rho"].values.flatten()
    rho_post_mean = float(rho_post.mean())

    # rhoは「正しいscale」で構築した場合にのみ「空間分散の割合」として解釈できる。
    # 誤ったscaleを流用すると、同じrho値でも実際の空間分散比率は
    # rho * scale_correct / scale_wrong に閉形式でずれる(数式的に決定的な関係)。
    rho_grid = np.linspace(0, 1, 200)
    actual_fraction_correct = rho_grid
    actual_fraction_wrong = np.clip(rho_grid * scale_correct / scale_wrong, 0, 1)
    misinterpreted_at_mean = rho_post_mean * scale_correct / scale_wrong

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ns = [s * s for s in sides]
    axes[0].bar([str(n) for n in ns], [scales[s] for s in sides], color=COLOR_OK)
    axes[0].set_xlabel("格子ノード数 N")
    axes[0].set_ylabel("スケーリング係数(一般化逆行列の対角の幾何平均)")
    axes[0].set_title("スケーリング係数は隣接グラフの\n構造(サイズ)ごとに異なる")

    axes[1].plot(rho_grid, actual_fraction_correct, color=COLOR_OK, lw=2,
                 label=f"正しいscale={scale_correct:.4f}(このN=36格子)\nを使った場合: rho=実際の空間分散比率")
    axes[1].plot(rho_grid, actual_fraction_wrong, color=COLOR_DIVERGENT, lw=2,
                 label=f"誤って借用したscale={scale_wrong:.4f}(N=16格子の値)\nを使った場合: 比率がずれる")
    axes[1].axvline(rho_post_mean, color="black", ls=":", lw=1.2,
                     label=f"実データでの事後平均rho={rho_post_mean:.3f}")
    axes[1].scatter([rho_post_mean], [rho_post_mean], color=COLOR_OK, zorder=5)
    axes[1].scatter([rho_post_mean], [misinterpreted_at_mean], color=COLOR_DIVERGENT, zorder=5)
    axes[1].annotate(f"誤解釈: 実際は{misinterpreted_at_mean:.3f}",
                      xy=(rho_post_mean, misinterpreted_at_mean), xytext=(0.05, 0.75),
                      fontsize=8.5, color=COLOR_DIVERGENT,
                      arrowprops=dict(arrowstyle="->", color=COLOR_DIVERGENT, lw=0.8))
    axes[1].set_xlabel("事後分布から得たrhoの値")
    axes[1].set_ylabel("実際の空間分散比率")
    axes[1].set_title("誤ったscaleを流用すると、rhoの値と\n実際の空間分散比率の対応がずれる")
    axes[1].legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bym2_scaling_factor_generalized_inverse.png")
    plt.close(fig)

    print(f"bym2_scaling_factor_generalized_inverse.png saved (" +
          ", ".join(f"N={s*s}:scale={scales[s]:.4f}" for s in sides) +
          f" | このN=36格子の正しいscale={scale_correct:.4f}, "
          f"誤って借用したN=16格子のscale={scale_wrong:.4f}, "
          f"事後平均rho={rho_post_mean:.3f} → 誤ったscaleでの実際の比率={misinterpreted_at_mean:.3f})")


def _path_graph_laplacian(k):
    """1〜kのパスグラフ(1次元隣接)のグラフラプラシアンを作る。"""
    W = np.zeros((k, k))
    for i in range(k - 1):
        W[i, i + 1] = 1
        W[i + 1, i] = 1
    return np.diag(W.sum(axis=1)) - W


def _build_and_sample_kronecker(Qs, Qt, y_flat, use_explicit, seed):
    m, n = Qs.shape[0], Qt.shape[0]
    with pm.Model():
        tau = pm.HalfNormal("tau", 1.0)
        Psi = pm.Normal("Psi", 0, 1, shape=(m, n))
        if use_explicit:
            Q_full = pt.as_tensor_variable(np.kron(Qs, Qt) + 1e-6 * np.eye(m * n))
            psi_vec = Psi.flatten()
            quad = pt.dot(psi_vec, pt.dot(Q_full, psi_vec))
        else:
            Qs_t = pt.as_tensor_variable(Qs)
            Qt_t = pt.as_tensor_variable(Qt)
            quad = pt.sum(Psi * pt.dot(Qs_t, pt.dot(Psi, Qt_t))) + 1e-6 * pt.sum(Psi ** 2)
        pm.Potential("gmrf_prior", -0.5 * tau * quad)
        pm.Normal("y", mu=Psi.flatten(), sigma=0.5, observed=y_flat)
        t0 = time.perf_counter()
        pm.sample(400, tune=400, chains=2, cores=1, target_accept=0.9,
                  random_seed=seed, progressbar=False,
                  compute_convergence_checks=False)
        elapsed = time.perf_counter() - t0
    return elapsed


def plot_kronecker_quadratic_form_comparison():
    """空間時系列GMRFの交互作用項の二次形式vec(Ψ)^T(Q_space⊗Q_time)vec(Ψ)を、
    (a)郡数×週数の次元を持つ精度行列を明示的に構築するナイーブな実装と、
    (b)sum(Psi*(Q_space@Psi@Q_time))というクロネッカー積の恒等式を使う実装
    の2通りでPyMCモデルとして実際にサンプリングし、数値的な一致と
    壁時計時間の差を比較する。"""

    rng = np.random.default_rng(5)

    Qs_check, Qt_check = _path_graph_laplacian(4), _path_graph_laplacian(5)
    Psi_check = rng.normal(size=(4, 5))
    vec_check = Psi_check.flatten(order="C")
    quad_explicit_check = vec_check @ np.kron(Qs_check, Qt_check) @ vec_check
    quad_trick_check = np.sum(Psi_check * (Qs_check @ Psi_check @ Qt_check))
    identity_diff = abs(quad_explicit_check - quad_trick_check)

    # コンパイルキャッシュを温めておき、初回JITコンパイルのノイズを除く
    dummy_Qs, dummy_Qt = _path_graph_laplacian(4), _path_graph_laplacian(4)
    dummy_y = rng.normal(0, 1, 16)
    _build_and_sample_kronecker(dummy_Qs, dummy_Qt, dummy_y, True, 0)
    _build_and_sample_kronecker(dummy_Qs, dummy_Qt, dummy_y, False, 0)

    sizes = [(8, 10), (14, 16), (20, 24), (28, 32)]
    mn_list, explicit_times, trick_times, explicit_mem_mb = [], [], [], []
    for m, n in sizes:
        Qs, Qt = _path_graph_laplacian(m), _path_graph_laplacian(n)
        y_flat = rng.normal(0, 1, m * n)
        mn_list.append(m * n)
        explicit_mem_mb.append((m * n) ** 2 * 8 / 1e6)
        explicit_times.append(_build_and_sample_kronecker(Qs, Qt, y_flat, True, 1))
        trick_times.append(_build_and_sample_kronecker(Qs, Qt, y_flat, False, 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x_pos = np.arange(len(sizes))
    width_bar = 0.35
    axes[0].bar(x_pos - width_bar / 2, explicit_times, width=width_bar, color=COLOR_DIVERGENT,
                label="明示的なQ_full構築 (O((mn)²))")
    axes[0].bar(x_pos + width_bar / 2, trick_times, width=width_bar, color=COLOR_OK,
                label="クロネッカー積の恒等式 (O(m²n+mn²))")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{m}×{n}\n(mn={m*n})" for m, n in sizes])
    axes[0].set_ylabel("壁時計時間(コンパイル+サンプリング、秒)")
    axes[0].set_title(f"恒等式利用は明示的なQ_full構築より高速\n(2式の数値差={identity_diff:.2e})")
    axes[0].legend(fontsize=8)

    axes[1].plot(mn_list, explicit_mem_mb, color=COLOR_DIVERGENT, lw=1.5, marker="o",
                 label="明示的なQ_full(mn×mn)のメモリ量")
    axes[1].set_xlabel("mn(郡数×週数)")
    axes[1].set_ylabel("Q_fullのメモリ量(MB)")
    axes[1].set_title("明示的な精度行列のメモリはO((mn)²)で増える")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "kronecker_quadratic_form_comparison.png")
    plt.close(fig)

    print("kronecker_quadratic_form_comparison.png saved (" +
          f"恒等式との数値差={identity_diff:.2e}, " +
          ", ".join(f"mn={mn}:explicit={e:.2f}s trick={t:.2f}s"
                     for mn, e, t in zip(mn_list, explicit_times, trick_times)) + ")")


def plot_bo_nuts_vs_map_refit_cost():
    """ベイズ最適化のように反復ごとにGPを再学習する場面で、毎回NUTSで
    フルの事後サンプリングを行うコストと、`pm.find_MAP`による点推定に
    切り替えた場合のコストを、観測点数を増やしながら実際に計測し比較する。
    また同じ初期デザインでNUTSの事後平均とfind_MAPの点推定が近い値を
    与えることを確認し、点推定への切り替えが安全であることを示す。"""

    rng = np.random.default_rng(17)

    def true_obj(x):
        return np.sin(3 * x) + 0.3 * x ** 2 - 0.5 * x

    n_list = [5, 15, 30, 45, 60, 80]
    nuts_times, map_times = [], []
    nuts_ls, map_ls = [], []

    for n in n_list:
        x_obs = rng.uniform(-2, 2, n)
        y_obs = true_obj(x_obs) + rng.normal(0, 0.1, n)

        with pm.Model():
            ls = pm.HalfNormal("ls", 1.0)
            var = pm.HalfNormal("var", 1.0)
            noise = pm.HalfNormal("noise", 0.3)
            cov = var * pm.gp.cov.ExpQuad(1, ls)
            gp = pm.gp.Marginal(cov_func=cov)
            gp.marginal_likelihood("y", X=x_obs[:, None], y=y_obs, sigma=noise)

            t0 = time.perf_counter()
            idata = pm.sample(1000, tune=1000, chains=4, cores=1, target_accept=0.95,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
            nuts_times.append(time.perf_counter() - t0)
            nuts_ls.append(float(idata.posterior["ls"].values.mean()))

            t0 = time.perf_counter()
            map_est = pm.find_MAP(progressbar=False)
            map_times.append(time.perf_counter() - t0)
            map_ls.append(float(map_est["ls"]))

    nuts_cum = np.cumsum(nuts_times)
    map_cum = np.cumsum(map_times)
    ratio_per_n = np.array(nuts_times) / np.array(map_times)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(n_list, nuts_times, color=COLOR_DIVERGENT, lw=1.5, marker="o", label="NUTS(4chains)の1回あたりコスト")
    axes[0].plot(n_list, map_times, color=COLOR_OK, lw=1.5, marker="s", label="find_MAPの1回あたりコスト")
    axes[0].set_xlabel("観測点数(BO反復が進み観測が増えるイメージ)")
    axes[0].set_ylabel("壁時計時間(秒)")
    axes[0].set_title(f"観測点が増えるとNUTSのコストがfind_MAPより\n速く増える(n={n_list[-1]}でratio={ratio_per_n[-1]:.1f}倍)")
    axes[0].legend(fontsize=9)

    axes[1].plot(n_list, nuts_ls, color=COLOR_DIVERGENT, lw=1.5, marker="o", label="NUTS事後平均(lengthscale)")
    axes[1].plot(n_list, map_ls, color=COLOR_OK, lw=1.5, marker="s", ls="--", label="find_MAP点推定(lengthscale)")
    axes[1].set_xlabel("観測点数")
    axes[1].set_ylabel("lengthscale")
    axes[1].set_title("同じデータでのNUTS事後平均とfind_MAP点推定は\n近い値を与える")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bo_nuts_vs_map_refit_cost.png")
    plt.close(fig)

    print(f"bo_nuts_vs_map_refit_cost.png saved (" +
          ", ".join(f"n={n}:NUTS={t1:.2f}s,MAP={t2:.2f}s,ratio={t1/t2:.2f}x"
                     for n, t1, t2 in zip(n_list, nuts_times, map_times)) +
          f" | lengthscale差(最終n) NUTS={nuts_ls[-1]:.3f} vs MAP={map_ls[-1]:.3f})")


def _rbf_kernel_nd(a, b, ls=1.0, var=1.0):
    sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return var * np.exp(-0.5 * sqdist / ls ** 2)


def _gp_posterior_nd(x_train, y_train, x_test, ls=1.0, var=1.0, noise=1e-3):
    K = _rbf_kernel_nd(x_train, x_train, ls, var) + noise * np.eye(len(x_train))
    K_s = _rbf_kernel_nd(x_train, x_test, ls, var)
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    var_diag = var - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
    return mu, np.sqrt(np.clip(var_diag, 1e-12, None))


def _neg_ei_nd(x, x_train, y_train, y_best, D, ls, var, xi=0.01):
    x = np.atleast_2d(x)
    mu, sigma = _gp_posterior_nd(x_train, y_train, x, ls, var)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return -ei[0]


def plot_acquisition_dimensionality_curse():
    """獲得関数(EI)の最大化を、低次元では機能する密なグリッド探索と、
    次元が増えても組合せ爆発しない`scipy.optimize`マルチスタートの
    2通りで実際に実行し、固定の評価予算のもとで次元が増えるにつれて
    グリッド探索が見つける最良値がマルチスタートに劣っていくこと、
    および両者の計算コストの違いを示す。"""

    rng = np.random.default_rng(23)
    budget = 4096
    dims = [1, 2, 4, 6]
    ls, var = 1.5, 1.0

    grid_best, grid_times = [], []
    multistart_best, multistart_times = [], []

    for D in dims:
        n_obs = 20
        x_train = rng.uniform(-3, 3, size=(n_obs, D))
        y_train = np.sum(np.sin(x_train), axis=1) + rng.normal(0, 0.1, n_obs)
        y_best = y_train.max()

        n_per_axis = max(2, int(round(budget ** (1.0 / D))))
        axes_1d = [np.linspace(-3, 3, n_per_axis) for _ in range(D)]
        mesh = np.meshgrid(*axes_1d, indexing="ij")
        grid_points = np.stack([m.flatten() for m in mesh], axis=1)

        t0 = time.perf_counter()
        mu, sigma = _gp_posterior_nd(x_train, y_train, grid_points, ls, var)
        imp = mu - y_best - 0.01
        z = imp / sigma
        ei_grid = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        grid_best.append(float(ei_grid.max()))
        grid_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        starts = rng.uniform(-3, 3, size=(20, D))
        best_val = -np.inf
        for x0 in starts:
            res = sopt.minimize(_neg_ei_nd, x0, args=(x_train, y_train, y_best, D, ls, var),
                                 method="L-BFGS-B", bounds=[(-3, 3)] * D)
            if -res.fun > best_val:
                best_val = -res.fun
        multistart_best.append(float(best_val))
        multistart_times.append(time.perf_counter() - t0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x_pos = np.arange(len(dims))
    width_bar = 0.35
    axes[0].bar(x_pos - width_bar / 2, grid_best, width=width_bar, color=COLOR_DIVERGENT,
                label=f"グリッド探索(総予算約{budget}点)")
    axes[0].bar(x_pos + width_bar / 2, multistart_best, width=width_bar, color=COLOR_OK,
                label="scipy.optimizeマルチスタート(20リスタート)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"D={d}" for d in dims])
    axes[0].set_ylabel("見つかった獲得関数(EI)の最良値")
    axes[0].set_title("次元が増えると固定予算のグリッド探索は\nマルチスタートに劣っていく")
    axes[0].legend(fontsize=8)

    axes[1].plot(dims, grid_times, color=COLOR_DIVERGENT, lw=1.5, marker="o", label="グリッド探索の壁時計時間")
    axes[1].plot(dims, multistart_times, color=COLOR_OK, lw=1.5, marker="s", label="マルチスタートの壁時計時間")
    axes[1].set_xlabel("次元 D")
    axes[1].set_ylabel("壁時計時間(秒)")
    axes[1].set_title("マルチスタートは次元が増えても\nグリッド全列挙のような爆発的な増加はしない")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acquisition_dimensionality_curse.png")
    plt.close(fig)

    print("acquisition_dimensionality_curse.png saved (" +
          ", ".join(f"D={d}:grid_best={gb:.4f}(t={gt:.3f}s),multistart_best={mb:.4f}(t={mt:.3f}s)"
                     for d, gb, gt, mb, mt in zip(dims, grid_best, grid_times, multistart_best, multistart_times)) +
          ")")


def plot_mvnormal_vs_sequential_conditioning():
    """2変数の同時正規分布を`pm.MvNormal`+`LKJCholeskyCov`で直接フィットする
    モデルと、周辺分布×条件付き分布への数式的な分解(y1の周辺正規分布と
    y2|y1の条件付き正規分布の積)を要素ごとの`pm.Normal`だけで実装した
    モデルの、それぞれで相関係数rhoの事後分布を実際にPyMCで比較し、
    両者が同じ分布を与える(=分解が数学的に同値である)ことを示す。"""

    rng = np.random.default_rng(31)
    n = 150
    rho_true = 0.6
    sigma1_true, sigma2_true = 1.2, 0.8
    cov_true = np.array([[sigma1_true ** 2, rho_true * sigma1_true * sigma2_true],
                          [rho_true * sigma1_true * sigma2_true, sigma2_true ** 2]])
    data = rng.multivariate_normal([0, 0], cov_true, size=n)
    y1_obs, y2_obs = data[:, 0], data[:, 1]

    with pm.Model():
        sd_dist = pm.HalfNormal.dist(1.0, shape=2)
        chol, corr, sigmas = pm.LKJCholeskyCov("chol_cov", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        pm.MvNormal("y", mu=[0, 0], chol=chol, observed=data)
        pm.Deterministic("rho_joint", corr[0, 1])
        idata_joint = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                 random_seed=1, progressbar=False,
                                 compute_convergence_checks=False)

    with pm.Model():
        sigma1 = pm.HalfNormal("sigma1", 1.0)
        sigma2_cond = pm.HalfNormal("sigma2_cond", 1.0)
        beta_cross = pm.Normal("beta_cross", 0, 1)
        pm.Normal("y1", 0, sigma1, observed=y1_obs)
        pm.Normal("y2", beta_cross * y1_obs, sigma2_cond, observed=y2_obs)
        sigma2_implied = pt.sqrt(sigma2_cond ** 2 + beta_cross ** 2 * sigma1 ** 2)
        rho_implied = pm.Deterministic("rho_decomposed", beta_cross * sigma1 / sigma2_implied)
        idata_decomp = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                  random_seed=2, progressbar=False,
                                  compute_convergence_checks=False)

    rho_joint = idata_joint.posterior["rho_joint"].values.flatten()
    rho_decomp = idata_decomp.posterior["rho_decomposed"].values.flatten()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(rho_joint, bins=50, density=True, color=COLOR_OK, alpha=0.55,
            label=f"pm.MvNormal+LKJCholeskyCov\n平均={rho_joint.mean():.3f}")
    ax.hist(rho_decomp, bins=50, density=True, color=COLOR_ALT, alpha=0.55,
            label=f"逐次条件付け分解(要素ごとpm.Normal)\n平均={rho_decomp.mean():.3f}")
    ax.axvline(rho_true, color="black", ls="--", lw=1.5, label=f"真の値={rho_true}")
    ax.set_xlabel("rho(相関係数)の事後分布")
    ax.set_ylabel("density")
    ax.set_title("逐次条件付け分解はpm.MvNormal+LKJと\n同じrhoの事後分布を与える")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mvnormal_vs_sequential_conditioning.png")
    plt.close(fig)

    print(f"mvnormal_vs_sequential_conditioning.png saved "
          f"(真のrho={rho_true}, MvNormal+LKJ平均={rho_joint.mean():.3f}, "
          f"逐次条件付け分解平均={rho_decomp.mean():.3f}, "
          f"両者の平均差={abs(rho_joint.mean()-rho_decomp.mean()):.4f})")


def plot_potential_soft_sum_to_zero():
    """他のRVに依存するsoft constraint(sum-to-zero制約)は`observed=`には
    渡せず`pm.Potential`で実装する必要がある。制約の強さeps(小さいほど
    強い制約)を変えながら実際にPyMCでフィットし、sum(phi)の事後標準偏差が
    epsに応じて縮むことを示す。"""

    rng = np.random.default_rng(15)
    K = 15
    true_phi = rng.normal(0, 1.0, K)
    true_phi -= true_phi.mean()
    y_obs = true_phi + rng.normal(0, 0.5, K)

    eps_list = [1.0, 0.3, 0.1, 0.03, 0.01]
    sum_std_list, sum_mean_list = [], []
    posterior_sums = {}

    for eps in eps_list:
        with pm.Model():
            tau = pm.HalfNormal("tau", 2.0)
            phi = pm.Normal("phi", 0, tau, shape=K)
            expr = pt.sum(phi)
            pm.Potential("sum_to_zero", -0.5 * (expr ** 2) / eps ** 2)
            pm.Normal("y", mu=phi, sigma=0.5, observed=y_obs)
            idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
        sum_phi = idata.posterior["phi"].values.sum(axis=-1).flatten()
        sum_std_list.append(float(sum_phi.std()))
        sum_mean_list.append(float(sum_phi.mean()))
        posterior_sums[eps] = sum_phi

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].loglog(eps_list, sum_std_list, color=COLOR_OK, lw=1.5, marker="o")
    axes[0].loglog(eps_list, eps_list, color="black", ls="--", lw=1.0, label="y = eps(参考線)")
    axes[0].set_xlabel("eps(sum_to_zero制約の強さ、小さいほど強い)")
    axes[0].set_ylabel("事後分布での sum(phi) の標準偏差")
    axes[0].set_title("pm.Potentialのepsを絞るほど\nsum(phi)は強制的にゼロへ近づく")
    axes[0].legend(fontsize=9)
    axes[0].invert_xaxis()

    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_list)))
    for eps, color in zip(eps_list, colors):
        axes[1].hist(posterior_sums[eps], bins=40, density=True, alpha=0.5, color=color,
                      label=f"eps={eps}(std={np.std(posterior_sums[eps]):.3f})")
    axes[1].set_xlabel("sum(phi)の事後分布")
    axes[1].set_ylabel("density")
    axes[1].set_title("epsを絞るとsum(phi)の事後分布は\n急速に0へ集中する")
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "potential_soft_sum_to_zero.png")
    plt.close(fig)

    print("potential_soft_sum_to_zero.png saved (" +
          ", ".join(f"eps={e}:sum_std={s:.4f}" for e, s in zip(eps_list, sum_std_list)) + ")")


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


def plot_piecewise_exponential_exposure():
    """Piecewise Exponentialモデルで、区間ごとの滞在時間(exposure)を
    正しく計算した場合(打ち切り/イベントが起きた最後の区間は端数だけ)と、
    誤って計算した場合(最後の区間も丸ごと通過したとみなす)とで、
    各区間のベースラインハザード推定がどれだけズレるかを示す。"""

    rng = np.random.default_rng(11)
    n = 600
    breaks = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    widths = np.diff(breaks)
    K = len(widths)
    true_h0 = np.array([0.05, 0.10, 0.20, 0.35])
    beta_true = 0.4

    x = rng.normal(0, 1, n)
    hazard_ratio = np.exp(beta_true * x)

    cum_H0_at_break = np.concatenate([[0.0], np.cumsum(true_h0 * widths)])

    u = rng.exponential(1.0, n)
    target = u / hazard_ratio
    event_time = np.empty(n)
    for i in range(n):
        j = np.searchsorted(cum_H0_at_break, target[i], side="right") - 1
        if j >= K:
            event_time[i] = np.inf  # ベースラインハザードの範囲を超えて生存
        else:
            event_time[i] = breaks[j] + (target[i] - cum_H0_at_break[j]) / true_h0[j]

    dropout = rng.uniform(0, 8, n)
    has_dropout = rng.uniform(0, 1, n) < 0.25
    censor_time = np.where(has_dropout, dropout, 8.0)

    observed_time = np.minimum(event_time, censor_time)
    event_ind = (event_time <= censor_time).astype(float)
    last_interval_idx = np.clip(np.searchsorted(breaks, observed_time, side="right") - 1, 0, K - 1)

    correct_exposure = np.zeros((n, K))
    naive_exposure = np.zeros((n, K))
    for j in range(K):
        full_pass = observed_time >= breaks[j + 1]
        in_interval = (observed_time > breaks[j]) & ~full_pass
        correct_exposure[:, j] = np.where(full_pass, widths[j], 0.0)
        correct_exposure[:, j] = np.where(in_interval, observed_time - breaks[j], correct_exposure[:, j])
        # 誤り: 最後(打ち切り/イベント)の区間も丸ごと通過したとみなす
        naive_exposure[:, j] = np.where(full_pass | in_interval, widths[j], 0.0)

    def fit_model(exposure_matrix):
        with pm.Model():
            h0 = pm.Gamma("h0", alpha=2.0, beta=10.0, shape=K)
            beta = pm.Normal("beta", 0, 1)
            hr = pt.exp(beta * x)
            H_i = pt.sum(exposure_matrix * h0[None, :], axis=1) * hr
            h_at_event = h0[last_interval_idx] * hr
            loglik = event_ind * pt.log(h_at_event) - H_i
            pm.Potential("loglik", pt.sum(loglik))
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=3, progressbar=False,
                               compute_convergence_checks=False)
        return idata

    idata_correct = fit_model(correct_exposure)
    idata_naive = fit_model(naive_exposure)

    h0_correct = idata_correct.posterior["h0"].values.reshape(-1, K)
    h0_naive = idata_naive.posterior["h0"].values.reshape(-1, K)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(K)
    width_bar = 0.25
    ax.bar(x_pos - width_bar, true_h0, width=width_bar, color="black", alpha=0.7, label="真値")
    ax.bar(x_pos, h0_correct.mean(axis=0), width=width_bar, color=COLOR_OK,
           yerr=h0_correct.std(axis=0), capsize=3, label="正しいexposure_matrix")
    ax.bar(x_pos + width_bar, h0_naive.mean(axis=0), width=width_bar, color=COLOR_DIVERGENT,
           yerr=h0_naive.std(axis=0), capsize=3, label="誤ったexposure_matrix(最後の区間も丸ごと通過扱い)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"区間{j}\n[{breaks[j]:.0f},{breaks[j+1]:.0f})" for j in range(K)])
    ax.set_ylabel("ベースラインハザード h0")
    ax.set_title("exposure_matrixの誤り(最後の区間を丸ごと通過扱い)は\n区間ごとのハザード推定を過小評価させる")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "piecewise_exponential_exposure.png")
    plt.close(fig)

    print("piecewise_exponential_exposure.png saved "
          f"(真値={true_h0.tolist()}, "
          f"correct={h0_correct.mean(axis=0).round(3).tolist()}, "
          f"naive={h0_naive.mean(axis=0).round(3).tolist()})")


def _simulate_hawkes(rng, T_horizon, mu, kappa, beta):
    """Ogataの間引き法で自己励起点過程(Hawkes過程)のイベント列を生成する。"""
    t = 0.0
    events = []
    while True:
        past = np.array(events)
        lam_bar = mu + (kappa * np.exp(-beta * (t - past))).sum() if events else mu
        t_candidate = t + rng.exponential(1.0 / lam_bar)
        if t_candidate > T_horizon:
            break
        past = np.array(events)
        lam_t = mu + (kappa * np.exp(-beta * (t_candidate - past))).sum() if events else mu
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)


def _hawkes_loglik(t_obs, T_horizon, mu, kappa, beta, use_scan):
    """Hawkes過程の対数尤度を、scan版(逐次再帰)とベクトル化版(全ペア行列)の
    どちらでも数学的に同じ値になるように計算する。"""
    if use_scan:
        dt_consec = t_obs[1:] - t_obs[:-1]

        def step(dt_i, S_prev, beta_ns):
            return pt.exp(-beta_ns * dt_i) * (S_prev + 1.0)

        S_rest, _ = pytensor.scan(fn=step, sequences=[dt_consec],
                                   outputs_info=[pt.constant(0.0, dtype="float64")],
                                   non_sequences=[beta], strict=True)
        S = pt.concatenate([[0.0], S_rest])
    else:
        dt_mat = t_obs[:, None] - t_obs[None, :]
        excitation = pt.switch(dt_mat > 0, pt.exp(-beta * dt_mat), 0.0)
        S = pt.sum(excitation, axis=1)

    lam = mu + kappa * S
    loglik_events = pt.sum(pt.log(lam))
    compensator = mu * T_horizon + (kappa / beta) * pt.sum(1 - pt.exp(-beta * (T_horizon - t_obs)))
    return loglik_events - compensator


def _build_and_sample_hawkes(t_obs, T_horizon, use_scan, seed):
    t_obs_tensor = pt.as_tensor_variable(t_obs)
    with pm.Model():
        mu = pm.HalfNormal("mu", 1.0)
        kappa = pm.HalfNormal("kappa", 1.0)
        beta = pm.HalfNormal("beta", 2.0)
        ll = _hawkes_loglik(t_obs_tensor, T_horizon, mu, kappa, beta, use_scan)
        pm.Potential("loglik", ll)
        t0 = time.perf_counter()
        pm.sample(500, tune=500, chains=2, cores=1, target_accept=0.9,
                  random_seed=seed, progressbar=False,
                  compute_convergence_checks=False)
        elapsed = time.perf_counter() - t0
    return elapsed


def plot_scan_vs_vectorized_hawkes():
    """Hawkes過程の対数尤度を、pytensor.scanによる逐次再帰実装と、
    全イベントペアの時間差行列によるベクトル化実装の2通りでPyMCモデルとして
    実装し、同じデータ・同じサンプリング設定での実測の壁時計時間(コンパイル
    +サンプリング)をイベント数を変えながら比較する。"""

    rng = np.random.default_rng(5)
    mu_true, kappa_true, beta_true = 0.3, 0.5, 1.0

    # コンパイルキャッシュを温めておき、初回JITコンパイルのノイズを除いた
    # 定常状態の比較にする
    dummy = np.sort(rng.uniform(0, 10, 5))
    _build_and_sample_hawkes(dummy, 10.0, True, 0)
    _build_and_sample_hawkes(dummy, 10.0, False, 0)

    horizons = [30, 100, 300, 600]
    n_events_list = []
    scan_times = []
    vec_times = []
    for T_horizon in horizons:
        t_obs = _simulate_hawkes(rng, T_horizon, mu_true, kappa_true, beta_true)
        n_events_list.append(len(t_obs))
        scan_times.append(_build_and_sample_hawkes(t_obs, T_horizon, True, 1))
        vec_times.append(_build_and_sample_hawkes(t_obs, T_horizon, False, 1))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(len(horizons))
    width_bar = 0.35
    ax.bar(x_pos - width_bar / 2, scan_times, width=width_bar, color=COLOR_ALT, label="scan(逐次再帰, O(n))")
    ax.bar(x_pos + width_bar / 2, vec_times, width=width_bar, color=COLOR_OK, label="ベクトル化(全ペア行列, O(n²))")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n={n}" for n in n_events_list])
    ax.set_ylabel("壁時計時間(コンパイル+サンプリング、秒)")
    ax.set_title("イベント数が少ないうちはベクトル化が優位だが、\n件数が増えるとO(n²)の負荷が逆転させる")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scan_vs_vectorized_hawkes.png")
    plt.close(fig)

    print("scan_vs_vectorized_hawkes.png saved (" +
          ", ".join(f"n={n}: scan={s:.2f}s vec={v:.2f}s"
                     for n, s, v in zip(n_events_list, scan_times, vec_times)) + ")")


if __name__ == "__main__":
    plot_scan_sequences_seasonal_beta()
    plot_censoring_clip_zero_division()
    plot_seed_consistency_effect()
    plot_ecological_bias_two_stage()
    plot_advi_variance_inflation()
    plot_piecewise_exponential_exposure()
    plot_scan_vs_vectorized_hawkes()
    plot_icar_closed_form_equivalence()
    plot_bym2_scaling_factor_generalized_inverse()
    plot_kronecker_quadratic_form_comparison()
    plot_bo_nuts_vs_map_refit_cost()
    plot_acquisition_dimensionality_curse()
    plot_mvnormal_vs_sequential_conditioning()
    plot_potential_soft_sum_to_zero()
