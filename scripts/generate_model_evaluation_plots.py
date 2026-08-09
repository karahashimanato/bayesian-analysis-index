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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import torch
import torch.nn as nn

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

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


def c_index(risk_score, event_time):
    """打ち切りなしの単純なconcordance index(全ペアのtotal order)。"""
    n = len(event_time)
    concordant = 0.0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if event_time[i] == event_time[j]:
                continue
            comparable += 1
            earlier = i if event_time[i] < event_time[j] else j
            later = j if earlier == i else i
            if risk_score[earlier] > risk_score[later]:
                concordant += 1
            elif risk_score[earlier] == risk_score[later]:
                concordant += 0.5
    return concordant / comparable


def brier_score(surv_prob, event_time, t_eval):
    observed = (event_time > t_eval).astype(float)
    return np.mean((surv_prob - observed) ** 2)


def plot_cindex_brier_independence():
    """共変量なし(基準ハザードのみ)/共変量ありの2つのベイズ指数分布ハザード
    モデルをフィットし、共変量追加でC-index(順位付け)は明確に改善するが
    Brier Score(較正)はほとんど変わらないことを示す。"""

    rng = np.random.default_rng(5)
    n_total = 320
    lambda0 = 0.05
    beta_true = 0.25

    x = rng.normal(0, 1, n_total)
    lam = lambda0 * np.exp(beta_true * x)
    T = rng.exponential(1 / lam)

    n_train = 220
    x_train, x_test = x[:n_train], x[n_train:]
    T_train, T_test = T[:n_train], T[n_train:]

    with pm.Model():
        log_lam0 = pm.Normal("log_lam0", np.log(0.05), 1.0)
        lam_i = pm.math.exp(log_lam0)
        pm.Exponential("T", lam=lam_i, observed=T_train)
        idata_a = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                             random_seed=1, progressbar=False,
                             compute_convergence_checks=False)

    with pm.Model():
        log_lam0 = pm.Normal("log_lam0", np.log(0.05), 1.0)
        beta = pm.Normal("beta", 0, 1.0)
        lam_i = pm.math.exp(log_lam0 + beta * x_train)
        pm.Exponential("T", lam=lam_i, observed=T_train)
        idata_b = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                             random_seed=1, progressbar=False,
                             compute_convergence_checks=False)

    lam0_a = float(idata_a.posterior["log_lam0"].mean())
    lam0_b = float(idata_b.posterior["log_lam0"].mean())
    beta_b = float(idata_b.posterior["beta"].mean())

    risk_a = np.full(len(x_test), np.exp(lam0_a))
    risk_b = np.exp(lam0_b + beta_b * x_test)

    t_eval = np.median(T_train)
    surv_a = np.exp(-risk_a * t_eval)
    surv_b = np.exp(-risk_b * t_eval)

    cidx_a = c_index(risk_a, T_test)
    cidx_b = c_index(risk_b, T_test)
    brier_a = brier_score(surv_a, T_test, t_eval)
    brier_b = brier_score(surv_b, T_test, t_eval)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["共変量なし\n(基準ハザードのみ)", "共変量あり\n(x込みのハザード比)"]
    x_pos = np.arange(2)

    b1 = axes[0].bar(x_pos, [cidx_a, cidx_b], color=[COLOR_DIVERGENT, COLOR_OK], width=0.5)
    for rect, val in zip(b1, [cidx_a, cidx_b]):
        axes[0].annotate(f"{val:.3f}", (rect.get_x() + rect.get_width() / 2, val),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    axes[0].axhline(0.5, color="black", lw=1, ls="--", label="ランダム(0.5)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("C-index")
    axes[0].set_ylim(0.4, max(cidx_a, cidx_b) + 0.1)
    axes[0].set_title("順位付け精度は共変量追加で明確に改善")
    axes[0].legend(fontsize=8)

    b2 = axes[1].bar(x_pos, [brier_a, brier_b], color=[COLOR_DIVERGENT, COLOR_OK], width=0.5)
    for rect, val in zip(b2, [brier_a, brier_b]):
        axes[1].annotate(f"{val:.4f}", (rect.get_x() + rect.get_width() / 2, val),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel(f"Brier Score (t={t_eval:.1f})")
    axes[1].set_ylim(0, max(brier_a, brier_b) * 1.3)
    axes[1].set_title("Brierの改善幅はC-indexほど劇的ではない")

    fig.suptitle("C-indexとBrier Scoreは独立の問題を測る", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "cindex_brier_independence.png")
    plt.close(fig)

    print(f"cindex_brier_independence.png saved "
          f"(C-index: {cidx_a:.3f}->{cidx_b:.3f}, Brier: {brier_a:.4f}->{brier_b:.4f})")


def plot_placebo_false_positive_rate():
    """真の効果がないデータに対し、架空の介入日(プラセボ)を多数設定して
    ベイズ的な検定(95%信用区間が0を含まないか)を繰り返し、単一のプラセボ
    試行だけでは偽陽性率(理論値5%)を判断できないことを示す。"""

    rng = np.random.default_rng(9)
    n_placebo = 60
    n_per_test = 40
    true_effect = 0.0

    detections = np.zeros(n_placebo, dtype=bool)
    lo_all = np.zeros(n_placebo)
    hi_all = np.zeros(n_placebo)
    mean_all = np.zeros(n_placebo)
    for k in range(n_placebo):
        y = rng.normal(true_effect, 1.0, n_per_test)
        with pm.Model():
            mu = pm.Normal("mu", 0, 2)
            sigma = pm.HalfNormal("sigma", 2)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(1000, tune=800, chains=2, target_accept=0.9,
                               random_seed=100 + k, progressbar=False,
                               compute_convergence_checks=False)
        mu_draws = idata.posterior["mu"].values.flatten()
        lo, hi = np.percentile(mu_draws, [2.5, 97.5])
        lo_all[k], hi_all[k], mean_all[k] = lo, hi, mu_draws.mean()
        detections[k] = not (lo <= 0 <= hi)  # 95%信用区間が0を含まない = 誤検出

    observed_fp_count = detections.sum()
    observed_fp_rate = observed_fp_count / n_placebo

    # 理論上の偽陽性率5%の下で、n_placebo回中k回検出される確率(二項分布)
    from scipy import stats
    k_range = np.arange(0, n_placebo + 1)
    binom_pmf = stats.binom.pmf(k_range, n_placebo, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    order = np.arange(n_placebo)
    colors = np.where(detections, COLOR_DIVERGENT, COLOR_OK)
    for k in order:
        axes[0].plot([lo_all[k], hi_all[k]], [k, k], color=colors[k], lw=1.3, alpha=0.8)
        axes[0].plot(mean_all[k], k, "o", color=colors[k], ms=2.5)
    axes[0].axvline(0, color="black", lw=1, ls="--", label="効果なし(真値=0)")
    axes[0].axhline(-0.5, color=COLOR_DIVERGENT, lw=0, label="")  # ダミー(凡例間隔調整用)
    axes[0].plot([], [], color=COLOR_DIVERGENT, lw=1.5, label="誤検出(95%CIが0を含まない)")
    axes[0].plot([], [], color=COLOR_OK, lw=1.5, label="正しく非有意")
    axes[0].set_xlabel("推定された効果(muの95%信用区間)")
    axes[0].set_ylabel("プラセボ試行 番号")
    axes[0].set_title(f"1回目の試行だけを見ても正しく非有意({'誤検出' if detections[0] else '正しい'})\n"
                       f"だが、それが偶然か理論通りかは複数回見ないと分からない")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].bar(k_range, binom_pmf, color=COLOR_ALT, alpha=0.6,
                label=f"理論分布 Binomial(n={n_placebo}, p=0.05)")
    axes[1].axvline(observed_fp_count, color=COLOR_DIVERGENT, lw=2,
                     label=f"実測: {n_placebo}回中{observed_fp_count}回誤検出 ({observed_fp_rate:.1%})")
    axes[1].set_xlabel("誤検出(有意)となった回数")
    axes[1].set_ylabel("確率")
    axes[1].set_xlim(-0.5, 12)
    axes[1].set_title(f"{n_placebo}回繰り返して初めて\n偽陽性率が理論値と整合するか判断できる")
    axes[1].legend(fontsize=8)

    fig.suptitle("プラセボ検定は1回では偽陽性率を測れない", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "placebo_false_positive_rate.png")
    plt.close(fig)

    print(f"placebo_false_positive_rate.png saved "
          f"(n_placebo={n_placebo}, 観測誤検出={observed_fp_count}回={observed_fp_rate:.1%}, "
          f"1回目の結果={'誤検出' if detections[0] else '正しく非有意'})")


def plot_loo_dse_comparison():
    """3つのベイズ線形回帰モデル(切片のみ/真の説明変数xあり/xと無関係な
    ノイズ変数zも追加)を実際にサンプリングし、az.compareでelpd_diffと
    その標準誤差dseを比較する。xの有無は明確に有意(dseの約12倍)だが、
    無関係なzの追加はelpd_diffがdseの範囲内(不確実性の中)に収まり
    有意でないことを示す。"""

    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)  # yと無関係なノイズ変数
    true_a, true_b = 2.0, 3.0
    y = true_a + true_b * x + rng.normal(0, 1.5, n)

    def fit(use_x, use_z):
        with pm.Model():
            a = pm.Normal("a", 0, 10)
            mu = a
            if use_x:
                b = pm.Normal("b", 0, 10)
                mu = mu + b * x
            if use_z:
                g = pm.Normal("g", 0, 10)
                mu = mu + g * z
            sigma = pm.HalfNormal("sigma", 5)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
            pm.compute_log_likelihood(idata)
        return idata

    idata_intercept = fit(False, False)
    idata_x = fit(True, False)
    idata_xz = fit(True, True)

    cmp_xvs0 = az.compare({"intercept_only": idata_intercept, "with_x": idata_x}, round_to=6)
    cmp_xzvs_x = az.compare({"with_x": idata_x, "with_x_and_z": idata_xz}, round_to=6)

    diff1 = abs(cmp_xvs0.loc["intercept_only", "elpd_diff"])
    dse1 = cmp_xvs0.loc["intercept_only", "dse"]
    diff2 = abs(cmp_xzvs_x.loc["with_x_and_z", "elpd_diff"])
    dse2 = cmp_xzvs_x.loc["with_x_and_z", "dse"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["切片のみ vs\n真の説明変数xあり", "xあり vs\nxと無関係なzも追加"]
    x_pos = np.arange(2)
    diffs = [diff1, diff2]
    dses = [dse1, dse2]
    colors = [COLOR_OK, COLOR_DIVERGENT]
    ax.bar(x_pos, diffs, yerr=dses, capsize=6, color=colors, width=0.5)
    for i, (d, se) in enumerate(zip(diffs, dses)):
        ax.annotate(f"|elpd_diff|={d:.1f}\ndse={se:.1f}\n({d/se:.1f}倍)",
                    (i, d + se), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|elpd_diff| (誤差棒はdse)")
    ax.set_title("elpd_diffの絶対値だけでなくdseとの比で有意性を判断する\n"
                  "(xの追加は明確に有意、無関係なzの追加は誤差の範囲内)")
    ax.set_ylim(0, max(d + se for d, se in zip(diffs, dses)) * 1.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loo_dse_comparison.png")
    plt.close(fig)

    print(f"loo_dse_comparison.png saved "
          f"(intercept_only vs with_x: |elpd_diff|={diff1:.1f}, dse={dse1:.1f}, {diff1/dse1:.1f}x; "
          f"with_x vs with_x_and_z: |elpd_diff|={diff2:.1f}, dse={dse2:.1f}, {diff2/dse2:.1f}x)")


def plot_mde_power_curve():
    """既知の効果量を人為的に注入した半合成データを、残差プールのブート
    ストラップ再サンプリングで反復ごとに変えながら生成し、ベイズ的な
    平均差検定(95%信用区間が0を含まないか)の検出率を効果量ごとに実測して
    検出力曲線(power curve)を描き、80%検出力に対応する最小検出可能効果
    (MDE)を求める。"""

    rng = np.random.default_rng(7)
    n_pre, n_post = 30, 30
    true_mu = 10.0
    obs_sigma = 2.0

    base_pre = rng.normal(true_mu, obs_sigma, n_pre)
    base_post = rng.normal(true_mu, obs_sigma, n_post)
    residual_pool = np.concatenate([base_pre - base_pre.mean(), base_post - base_post.mean()])

    def fit_effect(y_pre, y_post, seed):
        with pm.Model():
            mu_pre = pm.Normal("mu_pre", y_pre.mean(), 5)
            delta = pm.Normal("delta", 0, 5)
            sigma = pm.HalfNormal("sigma", 5)
            pm.Normal("y_pre", mu=mu_pre, sigma=sigma, observed=y_pre)
            pm.Normal("y_post", mu=mu_pre + delta, sigma=sigma, observed=y_post)
            idata = pm.sample(400, tune=400, chains=2, cores=1, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        d = idata.posterior["delta"].values.flatten()
        lo, hi = np.percentile(d, [2.5, 97.5])
        return not (lo <= 0 <= hi)

    effect_sizes = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    n_rep = 20
    detect_rates = []
    for delta_true in effect_sizes:
        detections = 0
        for r in range(n_rep):
            boot_pre = rng.choice(residual_pool, n_pre, replace=True) + true_mu
            boot_post = rng.choice(residual_pool, n_post, replace=True) + true_mu + delta_true
            detections += fit_effect(boot_pre, boot_post, seed=r)
        detect_rates.append(detections / n_rep)

    detect_rates = np.array(detect_rates)
    effect_sizes_arr = np.array(effect_sizes)
    # 80%検出力の効果量(MDE)を線形補間で求める
    if (detect_rates >= 0.8).any():
        mde = float(np.interp(0.8, detect_rates, effect_sizes_arr))
    else:
        mde = float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(effect_sizes_arr, detect_rates, "o-", color=COLOR_OK, lw=2, ms=7)
    ax.axhline(0.8, color=COLOR_ALT, ls="--", lw=1.3, label="検出力80%ライン")
    if not np.isnan(mde):
        ax.axvline(mde, color=COLOR_DIVERGENT, ls="--", lw=1.3, label=f"MDE ≈ {mde:.2f}")
    ax.set_xlabel("注入した効果量(真値)")
    ax.set_ylabel("検出率(95%信用区間が0を含まない割合)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"半合成データへの効果量注入から検出力曲線を実測し\nMDE(80%検出力の効果量)を較正する(反復{n_rep}回/効果量)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mde_power_curve.png")
    plt.close(fig)

    print("mde_power_curve.png saved (" +
          ", ".join(f"delta={d}: rate={r:.2f}" for d, r in zip(effect_sizes, detect_rates)) +
          f", MDE≈{mde:.2f})")


def _kalman_loglik(y_obs, mu, phi, sigma_level, obs_sigma, P0=25.0):
    """AR(1)型local levelモデルの周辺尤度をKalmanフィルタで計算する。
    潜在状態(level)を解析的に周辺化するため、非中心化パラメータ化を
    使わずとも divergence を起こさない(funnelを構造的に回避できる)。"""
    sigma_level2 = sigma_level ** 2
    obs_sigma2 = obs_sigma ** 2

    def step(y_t, level_prev, P_prev, mu_ns, phi_ns, sl2_ns, os2_ns):
        level_pred = mu_ns + phi_ns * (level_prev - mu_ns)
        P_pred = phi_ns ** 2 * P_prev + sl2_ns
        F = P_pred + os2_ns
        v = y_t - level_pred
        loglik_t = -0.5 * pt.log(2 * np.pi * F) - 0.5 * v ** 2 / F
        K = P_pred / F
        level_filt = level_pred + K * v
        P_filt = P_pred * (1 - K)
        return level_filt, P_filt, loglik_t

    (_, _, loglik_seq), _ = pytensor.scan(
        fn=step,
        sequences=[y_obs],
        outputs_info=[mu, pt.constant(P0, dtype="float64"), None],
        non_sequences=[mu, phi, sigma_level2, obs_sigma2],
        strict=True,
    )
    return pt.sum(loglik_seq)


def plot_ar1_phi_persistence():
    """真のデータ生成過程が純粋なランダムウォーク(平均回帰なし)である
    データに対し、平均回帰を許すAR(1)型local levelモデル(Kalmanフィルタで
    周辺化、divergenceなしの健全な事後分布)を評価期間n=60とn=150の両方で
    実際にサンプリングする。事前分布Beta(2,2)は0.5を中心に置いているにも
    かかわらず、phi(平均回帰の速さ、1に近いほどランダムウォークに近い)の
    事後分布はどちらの評価期間でも1側に寄り、期間を伸ばすほど1により強く
    張り付く(n=60: 平均0.761→n=150: 平均0.952)ことを示す。"""

    rng = np.random.default_rng(21)
    true_sigma_level = 0.5
    obs_sigma_true = 1.0

    def fit(n_pre, seed):
        level = np.cumsum(rng.normal(0, true_sigma_level, n_pre))
        y = level + rng.normal(0, obs_sigma_true, n_pre)
        with pm.Model():
            mu_level = pm.Normal("mu_level", y.mean(), 5.0)
            phi = pm.Beta("phi", 2, 2)
            sigma_level = pm.HalfNormal("sigma_level", 1.0)
            obs_sigma = pm.HalfNormal("obs_sigma", 1.0)
            ll = _kalman_loglik(pt.as_tensor_variable(y), mu_level, phi, sigma_level, obs_sigma)
            pm.Potential("loglik", ll)
            idata = pm.sample(1500, tune=2000, chains=4, target_accept=0.95,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["phi"].values.flatten(), int(idata.sample_stats["diverging"].sum())

    phi_60, div_60 = fit(60, 2)
    phi_150, div_150 = fit(150, 3)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(phi_60, bins=50, density=True, color=COLOR_ALT, alpha=0.55,
            label=f"評価期間n=60日(事後平均={phi_60.mean():.3f})")
    ax.hist(phi_150, bins=50, density=True, color=COLOR_OK, alpha=0.55,
            label=f"評価期間n=150日(事後平均={phi_150.mean():.3f})")
    ax.axvline(0.5, color="black", lw=1.5, ls="--", label="事前分布の中心(Beta(2,2))")
    ax.axvline(1.0, color=COLOR_DIVERGENT, lw=1.5, ls=":", label="phi=1(純粋なランダムウォーク)")
    ax.set_xlabel("phi(平均回帰の速さ)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1.05)
    ax.set_title("真のDGPがランダムウォークだと、評価期間を伸ばすほど\n"
                 "事前分布の中心(0.5)によらずphiの事後分布が1側へ強く張り付く")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ar1_phi_persistence.png")
    plt.close(fig)

    print(f"ar1_phi_persistence.png saved "
          f"(n=60: phi平均={phi_60.mean():.3f} divergence={div_60}, "
          f"n=150: phi平均={phi_150.mean():.3f} divergence={div_150})")


def plot_gp_extrapolation_behavior():
    """訓練域内では健全に収束する2種類のGP回帰(RBF-exact / HSGP)が、
    訓練域外への外挿では全く異なる挙動を示すことを示す。RBF-exact(ガウス
    尤度)は距離が離れるほど事前平均(0)へ回帰する一方、HSGP(Poisson尤度、
    有限個の基底関数による近似)は基底関数の有効域境界を超えると不安定になり、
    観測スケールを大きく超えて跳ね上がる。"""

    rng = np.random.default_rng(14)

    # ---- RBF-exact GP(ガウス尤度): 訓練域外で事前平均(0)へ回帰 ----
    x_train = np.linspace(0, 10, 80)
    y_train = 3.0 * np.sin(0.6 * x_train) + rng.normal(0, 0.3, 80)
    x_ext = np.linspace(-5, 20, 200)

    with pm.Model() as model_rbf:
        ell = pm.Gamma("ell", alpha=3, beta=1)
        eta = pm.HalfNormal("eta", 3.0)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ell)
        gp_rbf = pm.gp.Marginal(cov_func=cov)
        sigma = pm.HalfNormal("sigma", 1.0)
        gp_rbf.marginal_likelihood("y", X=x_train[:, None], y=y_train, sigma=sigma)
        idata_rbf = pm.sample(500, tune=1000, chains=2, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
    with model_rbf:
        f_ext = gp_rbf.conditional("f_ext", Xnew=x_ext[:, None])
        ppc_rbf = pm.sample_posterior_predictive(idata_rbf, var_names=["f_ext"],
                                                   random_seed=1, progressbar=False)
    f_ext_draws = ppc_rbf.posterior_predictive["f_ext"].values.reshape(-1, len(x_ext))
    f_ext_mean = f_ext_draws.mean(axis=0)
    f_ext_lo, f_ext_hi = np.percentile(f_ext_draws, [2.5, 97.5], axis=0)

    # ---- HSGP(Poisson尤度): 学習域の基底関数境界を超えると不安定になる ----
    x_train2 = np.linspace(0, 10, 100)
    true_rate = np.exp(np.log(20.0) + 0.8 * np.sin(0.6 * x_train2))
    counts = rng.poisson(true_rate)
    x_ext2 = np.linspace(0, 30, 300)

    with pm.Model() as model_hsgp:
        mu0 = np.log(counts.mean() + 1)
        eta2 = pm.HalfNormal("eta2", 2.0)
        ell2 = pm.Gamma("ell2", alpha=3, beta=1)
        cov2 = eta2 ** 2 * pm.gp.cov.ExpQuad(1, ls=ell2)
        gp_hsgp = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov2)
        f2 = gp_hsgp.prior("f2", X=x_train2[:, None])
        pm.Poisson("y2", mu=pm.math.exp(mu0 + f2), observed=counts)
        idata_hsgp = pm.sample(500, tune=1500, chains=2, target_accept=0.95,
                                random_seed=2, progressbar=False,
                                compute_convergence_checks=False)
    with model_hsgp:
        f2_ext = gp_hsgp.conditional("f2_ext", Xnew=x_ext2[:, None])
        ppc_hsgp = pm.sample_posterior_predictive(idata_hsgp, var_names=["f2_ext"],
                                                    random_seed=2, progressbar=False)
    f2_ext_draws = ppc_hsgp.posterior_predictive["f2_ext"].values.reshape(-1, len(x_ext2))
    rate_ext_draws = np.exp(mu0 + f2_ext_draws)
    rate_ext_mean = rate_ext_draws.mean(axis=0)
    rate_lo, rate_hi = np.percentile(rate_ext_draws, [2.5, 97.5], axis=0)

    rate_at_edge = float(np.interp(10, x_ext2, rate_ext_mean))
    rate_at_peak = float(rate_ext_mean[(x_ext2 > 10)].max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    axes[0].fill_between(x_ext, f_ext_lo, f_ext_hi, color=COLOR_OK, alpha=0.2)
    axes[0].plot(x_ext, f_ext_mean, color=COLOR_OK, lw=2, label="事後平均")
    axes[0].scatter(x_train, y_train, color="black", s=8, alpha=0.4, label="訓練データ")
    axes[0].axvspan(0, 10, color="gray", alpha=0.08, label="訓練域")
    axes[0].axhline(0, color=COLOR_DIVERGENT, lw=1, ls="--", label="事前平均(0)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("RBF-exact(ガウス尤度):\n訓練域外で事前平均へ回帰")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].fill_between(x_ext2, rate_lo, rate_hi, color=COLOR_ALT, alpha=0.2)
    axes[1].plot(x_ext2, rate_ext_mean, color=COLOR_ALT, lw=2, label="事後平均レート")
    axes[1].scatter(x_train2, counts, color="black", s=6, alpha=0.3, label="訓練データ(カウント)")
    axes[1].axvspan(0, 10, color="gray", alpha=0.08, label="訓練域")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("rate = exp(mu+f)")
    axes[1].set_title(f"HSGP(Poisson尤度):\n訓練域境界(rate={rate_at_edge:.0f})を超えると"
                       f"\n最大rate={rate_at_peak:.0f}まで不安定に跳ね上がる")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("GP回帰: 訓練域内で健全でも、外挿の挙動は手法によって全く異なる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "gp_extrapolation_behavior.png")
    plt.close(fig)

    print(f"gp_extrapolation_behavior.png saved "
          f"(RBF-exact: f(x=20)={np.interp(20, x_ext, f_ext_mean):.2f}(事前平均0に回帰), "
          f"HSGP: rate at訓練端={rate_at_edge:.0f} -> 域外最大={rate_at_peak:.0f})")


def _build_grid_adjacency(side):
    def idx(r, c):
        return r * side + c

    n = side * side
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


def plot_loo_variance_separation():
    """ICAR/BYM/BYM2の3モデルを6x6格子の疾病マッピング風データに実際に
    フィットし、az.compareのelpd_diffがいずれもdseの範囲内に収まり
    LOOでは予測性能を区別できないことと、それとは独立にBYM2のrho
    (空間/非構造の分散配分)は狭い信用区間で安定推定できることを示す。"""

    side = 6
    n = side * side
    rng = np.random.default_rng(11)
    W = _build_grid_adjacency(side)
    rr, cc = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    phi_true = 0.2 * (rr.flatten() + cc.flatten()) - 0.2 * (side - 1)
    phi_true -= phi_true.mean()
    theta_true = rng.normal(0, 0.25, n)
    E = rng.uniform(50, 150, n)
    counts = rng.poisson(E * np.exp(phi_true + theta_true))

    def fit(use_theta, use_phi, seed):
        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 2)
            log_rr = beta0
            if use_theta:
                sigma_theta = pm.HalfNormal("sigma_theta", 1)
                theta = pm.Normal("theta", 0, sigma_theta, shape=n)
                log_rr = log_rr + theta
            if use_phi:
                sigma_phi = pm.HalfNormal("sigma_phi", 1)
                phi = pm.ICAR("phi", W=W, sigma=sigma_phi)
                log_rr = log_rr + phi
            pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
            idata = pm.sample(800, tune=1200, chains=2, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
            pm.compute_log_likelihood(idata)
        return idata

    idata_icar = fit(False, True, 1)
    idata_bym = fit(True, True, 2)

    D = W.sum(axis=1)
    Q = np.diag(D) - W
    Q_inv_diag = np.diag(np.linalg.inv(Q + np.eye(n) * 1e-6))
    scale = float(np.exp(np.mean(np.log(Q_inv_diag))))

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        rho = pm.Beta("rho", 2, 2)
        theta_star = pm.Normal("theta_star", 0, 1, shape=n)
        phi_star = pm.ICAR("phi_star", W=W, sigma=1)
        combined = sigma * (pt.sqrt(1 - rho) * theta_star + pt.sqrt(rho / scale) * phi_star)
        log_rr = beta0 + combined
        pm.Poisson("y", mu=E * pm.math.exp(log_rr), observed=counts)
        idata_bym2 = pm.sample(800, tune=1200, chains=2, target_accept=0.9,
                                random_seed=3, progressbar=False,
                                compute_convergence_checks=False)
        pm.compute_log_likelihood(idata_bym2)

    cmp = az.compare({"ICAR": idata_icar, "BYM": idata_bym, "BYM2": idata_bym2}, round_to=6)
    max_ratio = float((cmp["elpd_diff"].abs() / cmp["dse"].replace(0, np.nan)).max())

    rho_draws = idata_bym2.posterior["rho"].values.flatten()
    rho_mean = float(rho_draws.mean())
    rho_lo, rho_hi = np.percentile(rho_draws, [2.5, 97.5])

    rhat_bym = max(float(az.rhat(idata_bym, var_names=["sigma_theta"])["sigma_theta"].values),
                   float(az.rhat(idata_bym, var_names=["sigma_phi"])["sigma_phi"].values))
    rhat_bym2 = max(float(az.rhat(idata_bym2, var_names=["sigma"])["sigma"].values),
                     float(az.rhat(idata_bym2, var_names=["rho"])["rho"].values))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    names = list(cmp.index)
    elpd_diff = cmp["elpd_diff"].values
    dse = cmp["dse"].values
    colors = [COLOR_OK if abs(d) <= s * 1.5 or s == 0 else COLOR_DIVERGENT for d, s in zip(elpd_diff, dse)]
    axes[0].bar(names, elpd_diff, yerr=dse, capsize=6, color=colors, width=0.5)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("elpd_diff(基準=最良モデル、誤差棒はdse)")
    axes[0].set_title(f"LOO(az.compare)では3モデルを\n区別できない(最大|elpd_diff|/dse={max_ratio:.2f}倍)")

    labels2 = ["BYM\n(sigma_theta/sigma_phi)", "BYM2\n(sigma/rho)"]
    rhats = [rhat_bym, rhat_bym2]
    colors2 = [COLOR_DIVERGENT if r > 1.01 else COLOR_OK for r in rhats]
    axes[1].bar(labels2, rhats, color=colors2, width=0.5)
    axes[1].axhline(1.01, color="black", lw=1, ls="--", label="r_hat=1.01の目安")
    for i, v in enumerate(rhats):
        axes[1].annotate(f"{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=10)
    axes[1].set_ylabel("分散成分パラメータの最大r_hat")
    axes[1].set_title(f"BYM2再パラメータ化はLOOの結果とは独立に\nサンプリングの健全性(r_hat)を改善する\n(BYM2のrho事後平均={rho_mean:.2f}, 95%CI=[{rho_lo:.2f},{rho_hi:.2f}])")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("LOOで差がつかなくても分散成分の推論の健全性は別軸で評価できる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "loo_variance_separation.png")
    plt.close(fig)

    print(f"loo_variance_separation.png saved "
          f"(elpd_diff/dse比 最大={max_ratio:.2f}, BYM r_hat={rhat_bym:.3f}, BYM2 r_hat={rhat_bym2:.3f}, "
          f"BYM2 rho事後平均={rho_mean:.3f} 95%CI=[{rho_lo:.3f}, {rho_hi:.3f}])")
    print(cmp[["elpd_diff", "dse"]])


def plot_credible_interval_edge_instability():
    """データが疎な領域(x>1)を含む3次多項式回帰を実際にPyMCでフィットし、
    信用区間の幅がデータの疎な端で急拡大する一方、真のピーク(データが密な
    領域内)の位置は事後サンプル間で安定して推定できることを示す。"""

    rng = np.random.default_rng(41)
    n_dense = 70
    x_dense = rng.uniform(-3, 1, n_dense)
    n_sparse = 7
    x_sparse = rng.uniform(1, 3.2, n_sparse)
    x = np.concatenate([x_dense, x_sparse])
    y_true = 5 - 0.35 * x ** 2
    y = y_true + rng.normal(0, 0.5, len(x))

    with pm.Model():
        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 5)
        c = pm.Normal("c", 0, 5)
        d = pm.Normal("d", 0, 2)
        mu = a + b * x + c * x ** 2 + d * x ** 3
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    post = idata.posterior
    a_d = post["a"].values.flatten()
    b_d = post["b"].values.flatten()
    c_d = post["c"].values.flatten()
    d_d = post["d"].values.flatten()

    xg = np.linspace(-3, 3.2, 300)
    curves = (a_d[:, None] + b_d[:, None] * xg[None, :]
              + c_d[:, None] * xg[None, :] ** 2 + d_d[:, None] * xg[None, :] ** 3)
    mean_curve = curves.mean(axis=0)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    width = hi - lo

    width_dense = float(np.interp(0, xg, width))
    width_sparse = float(np.interp(3, xg, width))

    argmax_x = xg[curves.argmax(axis=1)]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    axes[0].axvspan(-3, 1, color=COLOR_OK, alpha=0.08, label="データが密な領域")
    axes[0].axvspan(1, 3.2, color=COLOR_DIVERGENT, alpha=0.08, label="データが疎な領域")
    axes[0].fill_between(xg, lo, hi, color=COLOR_ALT, alpha=0.25, label="95%信用区間")
    axes[0].plot(xg, mean_curve, color=COLOR_ALT, lw=2, label="事後平均")
    axes[0].scatter(x, y, color="black", s=12, alpha=0.4, label="観測データ")
    axes[0].annotate(f"幅={width_dense:.2f}", (0, hi[np.argmin(np.abs(xg))] + 0.3), ha="center", fontsize=9)
    axes[0].annotate(f"幅={width_sparse:.2f}", (3, hi[np.argmin(np.abs(xg - 3))] + 0.3), ha="center", fontsize=9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"信用区間の幅は疎な端で急拡大\n(x=0: 幅{width_dense:.2f} → x=3: 幅{width_sparse:.2f}, "
                       f"{width_sparse / width_dense:.1f}倍)")
    axes[0].legend(fontsize=7.5, loc="lower left")

    axes[1].hist(argmax_x, bins=40, color=COLOR_OK, alpha=0.75)
    axes[1].axvline(0, color="black", lw=1, ls="--", label="真のピーク(x=0)")
    axes[1].axvspan(1, 3.2, color=COLOR_DIVERGENT, alpha=0.08, label="データが疎な領域")
    axes[1].set_xlim(-3, 3.2)
    axes[1].set_xlabel("事後サンプルごとの曲線の最大値の位置")
    axes[1].set_ylabel("サンプル数")
    axes[1].set_title(f"ピーク位置はデータが密な領域内で\n安定して推定できる"
                       f"(平均={argmax_x.mean():.2f}, 標準偏差={argmax_x.std():.2f})")
    axes[1].legend(fontsize=8)

    fig.suptitle("意思決定に使う推定は信用区間が安定している範囲に限定する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "credible_interval_edge_instability.png")
    plt.close(fig)

    print(f"credible_interval_edge_instability.png saved "
          f"(CI幅 x=0:{width_dense:.3f} x=3:{width_sparse:.3f} 比={width_sparse/width_dense:.2f}x, "
          f"argmax位置 平均={argmax_x.mean():.3f} 標準偏差={argmax_x.std():.3f}, "
          f"疎な領域にargmaxが入った割合={float((argmax_x>1).mean()):.1%})")


def plot_hierarchical_shrinkage_bias():
    """K=20腕のCTRを階層Beta-Binomialモデル(部分プーリング)で実際に
    PyMCでフィットし、単純なMLE(腕ごとの観測比率)と比較する。階層推定は
    全体のMAEをMLEより下げる一方、真のCTRが高い腕を系統的に過小評価し、
    低い腕を過大評価する収縮バイアスを持つことを示す。"""

    rng = np.random.default_rng(3)
    K = 20
    true_ctr = np.linspace(0.01, 0.40, K)
    n_trials = rng.integers(20, 60, K)
    successes = rng.binomial(n_trials, true_ctr)
    mle = successes / n_trials

    with pm.Model():
        mu = pm.Beta("mu", 2, 2)
        kappa = pm.Gamma("kappa", 2, 0.1)
        p = pm.Beta("p", mu * kappa, (1 - mu) * kappa, shape=K)
        pm.Binomial("y", n=n_trials, p=p, observed=successes)
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    p_est = idata.posterior["p"].values.reshape(-1, K).mean(axis=0)
    mae_hier = float(np.mean(np.abs(p_est - true_ctr)))
    mae_mle = float(np.mean(np.abs(mle - true_ctr)))

    top_idx = true_ctr > np.percentile(true_ctr, 75)
    bot_idx = true_ctr < np.percentile(true_ctr, 25)
    bias_hier_top = float((p_est[top_idx] - true_ctr[top_idx]).mean())
    bias_hier_bot = float((p_est[bot_idx] - true_ctr[bot_idx]).mean())
    bias_mle_top = float((mle[top_idx] - true_ctr[top_idx]).mean())
    bias_mle_bot = float((mle[bot_idx] - true_ctr[bot_idx]).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    lims = [0, 0.45]
    axes[0].plot(lims, lims, color="black", lw=1, ls="--", label="y=x(完全一致)")
    axes[0].scatter(true_ctr, mle, color=COLOR_ALT, s=45, alpha=0.8, label="MLE(腕ごとの観測比率)")
    axes[0].scatter(true_ctr, p_est, color=COLOR_DIVERGENT, s=45, alpha=0.8, label="階層モデル(部分プーリング)")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("真のCTR")
    axes[0].set_ylabel("推定CTR")
    axes[0].set_title("階層モデルは高CTR腕を過小評価、\n低CTR腕を過大評価する収縮バイアスを持つ")
    axes[0].legend(fontsize=8.5, loc="upper left")

    labels = ["全体MAE", "高CTR群\n(上位25%)の平均バイアス", "低CTR群\n(下位25%)の平均バイアス"]
    x_pos = np.arange(3)
    width_bar = 0.35
    mle_vals = [mae_mle, bias_mle_top, bias_mle_bot]
    hier_vals = [mae_hier, bias_hier_top, bias_hier_bot]
    axes[1].bar(x_pos - width_bar / 2, mle_vals, width=width_bar, color=COLOR_ALT, label="MLE")
    axes[1].bar(x_pos + width_bar / 2, hier_vals, width=width_bar, color=COLOR_DIVERGENT, label="階層モデル")
    axes[1].axhline(0, color="black", lw=0.8)
    for i, (m, h) in enumerate(zip(mle_vals, hier_vals)):
        axes[1].annotate(f"{m:+.3f}", (i - width_bar / 2, m), xytext=(0, 4 if m >= 0 else -14),
                          textcoords="offset points", ha="center", fontsize=8)
        axes[1].annotate(f"{h:+.3f}", (i + width_bar / 2, h), xytext=(0, 4 if h >= 0 else -14),
                          textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_title(f"全体MAEは階層モデルの方が小さい\n(MLE={mae_mle:.3f} vs 階層={mae_hier:.3f})が\n"
                       f"腕ごとの系統バイアスは階層モデルの方が大きい")
    axes[1].legend(fontsize=9)

    fig.suptitle("階層モデルの収縮バイアスは事前分布の調整だけでは消えない", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "hierarchical_shrinkage_bias.png")
    plt.close(fig)

    print(f"hierarchical_shrinkage_bias.png saved "
          f"(MAE mle={mae_mle:.4f} hier={mae_hier:.4f} 比={mae_hier/mae_mle:.2f}, "
          f"高CTR群バイアス mle={bias_mle_top:+.4f} hier={bias_hier_top:+.4f}, "
          f"低CTR群バイアス mle={bias_mle_bot:+.4f} hier={bias_hier_bot:+.4f})")


def _ope_estimators(actions, rewards, pi_b_probs, pi_e_probs, K):
    q_hat = np.zeros(K)
    for a in range(K):
        mask = actions == a
        q_hat[a] = rewards[mask].mean() if mask.sum() > 0 else 0.0
    dm = float(np.sum(pi_e_probs * q_hat))
    w = pi_e_probs[actions] / pi_b_probs[actions]
    ips = float(np.mean(w * rewards))
    snips = float(np.sum(w * rewards) / np.sum(w))
    return dm, ips, snips


def plot_ope_special_case_identity():
    """K=4腕のオフライン方策評価で、ログ収集方策と評価方策がどちらも一様
    ランダムという特殊構造の下ではDM・IPS・SNIPSがほぼ完全に一致するが、
    ログ収集方策が偏っている(評価方策と異なる)場合は3推定量の間のばらつきが
    約5倍に拡大することを、実際に多数回のシミュレーションで確認する。"""

    rng = np.random.default_rng(61)
    K = 4
    mu_true = np.array([0.10, 0.15, 0.20, 0.25])
    pi_uniform = np.full(K, 0.25)
    pi_skewed = np.array([0.60, 0.25, 0.10, 0.05])
    n = 1500
    reps = 300

    def simulate(pi_b_probs, n):
        actions = rng.choice(K, size=n, p=pi_b_probs)
        rewards = rng.binomial(1, mu_true[actions])
        return actions, rewards

    spread_u, spread_s = [], []
    vals_u_example, vals_s_example = None, None
    for r in range(reps):
        a, rw = simulate(pi_uniform, n)
        dm, ips, snips = _ope_estimators(a, rw, pi_uniform, pi_uniform, K)
        spread_u.append(max(dm, ips, snips) - min(dm, ips, snips))
        if r == 0:
            vals_u_example = (dm, ips, snips)

        a2, rw2 = simulate(pi_skewed, n)
        dm2, ips2, snips2 = _ope_estimators(a2, rw2, pi_skewed, pi_uniform, K)
        spread_s.append(max(dm2, ips2, snips2) - min(dm2, ips2, snips2))
        if r == 0:
            vals_s_example = (dm2, ips2, snips2)

    spread_u = np.array(spread_u)
    spread_s = np.array(spread_s)
    true_value = float(mu_true.mean())  # 評価方策(一様)の下での真の期待報酬

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    labels3 = ["DM", "IPS", "SNIPS"]
    xp = np.arange(3)
    axes[0].axhline(true_value, color="black", lw=1, ls="--", label=f"真の評価方策価値={true_value:.3f}")
    axes[0].plot(xp - 0.08, vals_u_example, "o", color=COLOR_OK, ms=10,
                 label="ログ収集方策=一様(評価方策と同一)")
    axes[0].plot(xp + 0.08, vals_s_example, "s", color=COLOR_DIVERGENT, ms=10,
                 label="ログ収集方策=偏り(評価方策と別)")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(labels3)
    axes[0].set_ylabel("推定値")
    axes[0].set_title("ログ収集方策が一様(評価方策と同一)だと\n3推定量がほぼ完全に一致する")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].boxplot([spread_u, spread_s], tick_labels=["ログ収集方策=一様", "ログ収集方策=偏り"], widths=0.5)
    axes[1].set_ylabel("3推定量の最大差(max-min)")
    axes[1].set_title(f"{reps}回のシミュレーションでの3推定量の\nばらつきの比較(偏り時は一様時の約"
                       f"{spread_s.mean()/spread_u.mean():.1f}倍)")

    fig.suptitle("特殊構造(一様な傾向スコア)の下での推定量一致を実装検証に使う", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "ope_special_case_identity.png")
    plt.close(fig)

    print(f"ope_special_case_identity.png saved "
          f"(真の価値={true_value:.4f}, 一様時の例: DM={vals_u_example[0]:.4f} IPS={vals_u_example[1]:.4f} "
          f"SNIPS={vals_u_example[2]:.4f}, 偏り時の例: DM={vals_s_example[0]:.4f} IPS={vals_s_example[1]:.4f} "
          f"SNIPS={vals_s_example[2]:.4f}, 平均spread 一様={spread_u.mean():.5f} 偏り={spread_s.mean():.5f} "
          f"比={spread_s.mean()/spread_u.mean():.2f}x)")


def plot_external_validation_correlation():
    """K=8腕のログデータからDM推定量で算出したOPEの価値ランキングと、
    同じ真の報酬構造の下で独立にThompson Samplingを実際に走らせた
    「本番方策」が収束した腕選択頻度のランキングを、実際にシミュレーション
    して順位相関(Spearman)で突き合わせる。"""

    rng = np.random.default_rng(71)
    K = 8
    mu_true = rng.uniform(0.05, 0.35, K)

    n_log = 4000
    actions = rng.integers(0, K, n_log)
    rewards = rng.binomial(1, mu_true[actions])
    ope_est = np.array([rewards[actions == a].mean() if (actions == a).sum() > 0 else 0.0 for a in range(K)])

    T = 6000
    alpha_ts = np.ones(K)
    beta_ts = np.ones(K)
    choice_hist = np.zeros(T, dtype=int)
    for t in range(T):
        samples = rng.beta(alpha_ts, beta_ts)
        a = int(np.argmax(samples))
        r = rng.binomial(1, mu_true[a])
        alpha_ts[a] += r
        beta_ts[a] += 1 - r
        choice_hist[t] = a
    last = choice_hist[-2000:]
    freq = np.array([(last == a).mean() for a in range(K)])

    from scipy import stats as sstats
    rho, pval = sstats.spearmanr(ope_est, freq)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = np.argsort(-ope_est)
    for rank, a in enumerate(order):
        ax.scatter(ope_est[a], max(freq[a], 3e-4), color=COLOR_OK, s=60, zorder=3)
        ax.annotate(f"腕{a}", (ope_est[a], max(freq[a], 3e-4)), xytext=(5, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("ログのみから算出したOPE推定値(DM)")
    ax.set_ylabel("本番Thompson Samplingが収束した選択頻度(対数軸)")
    ax.set_title(f"ログのみのOPE推定値と本番方策の収束先の\n順位相関: Spearman rho={rho:.2f}(p={pval:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_validation_correlation.png")
    plt.close(fig)

    print(f"external_validation_correlation.png saved "
          f"(spearman rho={rho:.3f}, p={pval:.4f}, mu_true={np.round(mu_true, 3)}, "
          f"ope_est={np.round(ope_est, 3)}, freq={np.round(freq, 3)})")


def plot_ppc_good_mechanism_wrong():
    """真の周期(25ステップ)を持つ合成データに対し、それとは異なる周期
    (42ステップ、意図的な機構の誤指定)のcos/sin構造項と、自由に調整可能な
    GaussianRandomWalk(潜在レベル)を持つモデルを実際にPyMCでフィットする。
    事後予測(PPC、レベル込み)はデータとほぼ完全に一致する(相関~1.0)一方、
    構造パラメータ(beta, gamma)だけで決定論的に走らせたforward simulationは
    データの周期性をほとんど再現できない(相関~0.2)ことを示す。"""

    rng = np.random.default_rng(81)
    T = 100
    true_period = 25
    A_true = 1.2
    t = np.arange(T)
    y_true_signal = A_true * np.sin(2 * np.pi * t / true_period)
    y = y_true_signal + rng.normal(0, 0.3, T)

    wrong_period = 42  # 意図的に真の周期とずらした構造パラメータの周期
    omega = 2 * np.pi / wrong_period

    with pm.Model():
        beta = pm.Normal("beta", 0, 2)
        gamma = pm.Normal("gamma", 0, 2)
        sigma_level = pm.HalfNormal("sigma_level", 0.5)
        level = pm.GaussianRandomWalk("level", sigma=sigma_level, shape=T)
        obs_sigma = pm.HalfNormal("obs_sigma", 0.5)
        structural = beta * np.cos(omega * t) + gamma * np.sin(omega * t)
        mu = pm.Deterministic("mu", structural + level)
        pm.Normal("y_obs", mu=mu, sigma=obs_sigma, observed=y)
        idata = pm.sample(800, tune=1200, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    mu_draws = idata.posterior["mu"].values.reshape(-1, T)
    mu_mean = mu_draws.mean(axis=0)
    mu_lo, mu_hi = np.percentile(mu_draws, [2.5, 97.5], axis=0)
    ppc_corr = float(np.corrcoef(mu_mean, y)[0, 1])

    beta_mean = float(idata.posterior["beta"].mean())
    gamma_mean = float(idata.posterior["gamma"].mean())
    structural_only = beta_mean * np.cos(omega * t) + gamma_mean * np.sin(omega * t)
    forward_corr = float(np.corrcoef(structural_only, y)[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    axes[0].fill_between(t, mu_lo, mu_hi, color=COLOR_OK, alpha=0.25, label="事後予測95%区間(潜在レベル込み)")
    axes[0].plot(t, mu_mean, color=COLOR_OK, lw=1.8, label="事後予測平均")
    axes[0].scatter(t, y, color="black", s=12, alpha=0.5, label="観測データ")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"PPC(潜在レベル込み)はデータとほぼ完全に一致\n(相関={ppc_corr:.3f}) — しかし機構が正しい保証にはならない")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(t, structural_only, color=COLOR_DIVERGENT, lw=1.8,
                 label=f"構造パラメータのみのforward-sim(周期{wrong_period}のcos/sin項)")
    axes[1].scatter(t, y, color="black", s=12, alpha=0.5, label="観測データ(真の周期25)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"潜在レベルを外し構造パラメータだけで走らせると\n"
                       f"データの周期性をほとんど再現できない(相関={forward_corr:.3f})")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("PPCが良好でも、モデルの機構が現象を説明しているとは限らない", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "ppc_good_mechanism_wrong.png")
    plt.close(fig)

    print(f"ppc_good_mechanism_wrong.png saved "
          f"(PPC相関={ppc_corr:.4f}, forward-sim相関={forward_corr:.4f}, "
          f"beta={beta_mean:.3f}, gamma={gamma_mean:.3f})")


class _SmallMLP(nn.Module):
    """last-layer Laplace / Deep Ensembleの比較に使う小さなMLP(平均のみ出力)。"""

    def __init__(self, d_in, hidden=50):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        h = torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(h))

    def forward(self, x):
        return self.head(self.features(x))


def _train_small_mlp(d_in, x, y, epochs=800, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    m = _SmallMLP(d_in)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return m


def _nll(mu, var, y):
    return float((0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mu) ** 2 / var).mean())


def _laplace_last_layer_nll(m, x_train, y_train, x_test, y_test, prior_precision=5.0):
    with torch.no_grad():
        phi_train = m.features(x_train).numpy()
        phi_test = m.features(x_test).numpy()
        mu_train = m(x_train).numpy()
        w_map = m.head.weight.detach().numpy().flatten()
        b_map = float(m.head.bias.detach())
    resid_var = float(np.mean((mu_train - y_train.numpy()) ** 2))
    H = phi_train.shape[1]
    A = phi_train.T @ phi_train / resid_var + prior_precision * np.eye(H)
    A_inv = np.linalg.inv(A)
    mu_test = phi_test @ w_map + b_map
    var_epi = np.einsum("ij,jk,ik->i", phi_test, A_inv, phi_test)
    var_test = var_epi + resid_var
    return _nll(mu_test, var_test, y_test.numpy().flatten())


def _deep_ensemble_nll(d_in, x_train, y_train, x_test, y_test, M=5, seed0=100, epochs=800):
    mus = []
    resid_vars = []
    for s in range(M):
        m = _train_small_mlp(d_in, x_train, y_train, epochs=epochs, seed=seed0 + s)
        with torch.no_grad():
            mu_tr = m(x_train).numpy()
            mu_te = m(x_test).numpy()
        mus.append(mu_te.flatten())
        resid_vars.append(np.mean((mu_tr.flatten() - y_train.numpy().flatten()) ** 2))
    mus = np.array(mus)
    mu_mean = mus.mean(axis=0)
    var_test = mus.var(axis=0) + np.mean(resid_vars)
    return _nll(mu_mean, var_test, y_test.numpy().flatten())


def plot_dimension_dependent_conclusions():
    """last-layer Laplace近似とDeep Ensemblesを実際にPyTorchで学習し、
    (A) 1次元・訓練分布内のテストでは両手法のtest NLLが近い一方、
    (B) 6次元・訓練域から離れた分布シフトのあるテストでは、特徴抽出器を
    MAP解に固定するlast-layer Laplaceのtest NLLがDeep Ensemblesより
    大幅に悪化することを示す。低次元での手法比較を高次元・分布シフト下に
    単純に外挿できない実例。"""

    # ---- タスクA: 1次元、訓練分布内のテスト ----
    rng = np.random.default_rng(1)
    n_train, n_test = 300, 150
    x_train_a = rng.uniform(-4, 4, n_train)
    y_train_a = np.sin(x_train_a) + rng.normal(0, 0.15, n_train)
    x_test_a = rng.uniform(-4, 4, n_test)
    y_test_a = np.sin(x_test_a) + rng.normal(0, 0.15, n_test)
    xt_a = torch.tensor(x_train_a[:, None], dtype=torch.float32)
    yt_a = torch.tensor(y_train_a[:, None], dtype=torch.float32)
    xte_a = torch.tensor(x_test_a[:, None], dtype=torch.float32)
    yte_a = torch.tensor(y_test_a[:, None], dtype=torch.float32)

    m_a = _train_small_mlp(1, xt_a, yt_a, seed=1)
    nll_ll_a = _laplace_last_layer_nll(m_a, xt_a, yt_a, xte_a, yte_a)
    nll_de_a = _deep_ensemble_nll(1, xt_a, yt_a, xte_a, yte_a)

    # ---- タスクB: 6次元、訓練域から離れたOODテスト ----
    rng = np.random.default_rng(2)
    D = 6

    def f6(x):
        return np.sin(x[:, 0] * 1.5) + 0.4 * x[:, 1] ** 2 - 0.5 * x[:, 2] * x[:, 3] + 0.3 * x[:, 4] - 0.2 * x[:, 5]

    n_train_b = 500
    x_train_b = rng.normal(0, 0.7, (n_train_b, D))
    y_train_b = f6(x_train_b) + rng.normal(0, 0.15, n_train_b)
    n_test_b = 200
    direction = rng.normal(0, 1, D)
    direction /= np.linalg.norm(direction)
    shift = direction * 3.0
    x_test_b = rng.normal(0, 0.7, (n_test_b, D)) + shift[None, :]
    y_test_b = f6(x_test_b) + rng.normal(0, 0.15, n_test_b)

    xt_b = torch.tensor(x_train_b, dtype=torch.float32)
    yt_b = torch.tensor(y_train_b[:, None], dtype=torch.float32)
    xte_b = torch.tensor(x_test_b, dtype=torch.float32)
    yte_b = torch.tensor(y_test_b[:, None], dtype=torch.float32)

    m_b = _train_small_mlp(D, xt_b, yt_b, seed=1)
    nll_ll_b = _laplace_last_layer_nll(m_b, xt_b, yt_b, xte_b, yte_b)
    nll_de_b = _deep_ensemble_nll(D, xt_b, yt_b, xte_b, yte_b)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    labels = ["last-layer\nLaplace", "Deep\nEnsembles"]
    axes[0].bar(labels, [nll_ll_a, nll_de_a], color=[COLOR_DIVERGENT, COLOR_OK], width=0.5)
    for i, v in enumerate([nll_ll_a, nll_de_a]):
        axes[0].annotate(f"{v:.2f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    axes[0].set_ylabel("test NLL(低いほど良い)")
    axes[0].set_title("タスクA: 1次元・訓練分布内のテスト\n両手法のtest NLLは近い")

    axes[1].bar(labels, [nll_ll_b, nll_de_b], color=[COLOR_DIVERGENT, COLOR_OK], width=0.5)
    for i, v in enumerate([nll_ll_b, nll_de_b]):
        axes[1].annotate(f"{v:.1f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    axes[1].set_ylabel("test NLL(低いほど良い)")
    axes[1].set_title("タスクB: 6次元・訓練域から離れた\n分布シフトのあるテスト\nlast-layerが大幅に悪化")

    fig.suptitle("手法比較の結論はタスクの次元・複雑さに依存し、単純に外挿できない", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "dimension_dependent_conclusions.png")
    plt.close(fig)

    print(f"dimension_dependent_conclusions.png saved "
          f"(タスクA 1D in-domain: last-layer NLL={nll_ll_a:.3f} deep-ensemble NLL={nll_de_a:.3f}; "
          f"タスクB 6D OOD: last-layer NLL={nll_ll_b:.2f} deep-ensemble NLL={nll_de_b:.2f})")


def plot_missing_mechanism_asymmetric_bias():
    """観測済み共変量(GDP相当)にのみ依存するMAR欠測を実際に300回反復
    シミュレーションし、完全ケース分析(CC)の回帰係数バイアスと周辺平均
    バイアスを比較する。回帰係数はMCAR・MARどちらの下でもほぼゼロバイアス
    (反復平均で理論通り)だが、周辺平均のバイアスはMCARでは無視できるほど
    小さい一方、MARでは大きく残ることを示す。単回の試行では有効サンプル数の
    縮小によるばらつきで係数バイアスが見かけ上大きく出ることがあるため、
    反復平均でバイアスとばらつきを切り分けている。"""

    n = 300
    true_beta0, true_beta1 = 3.0, -0.6
    true_mean_y = true_beta0  # E[y] = beta0 (gdpの平均は0)

    def one_rep(seed):
        r = np.random.default_rng(seed)
        gdp = r.normal(0, 1, n)
        y = true_beta0 + true_beta1 * gdp + r.normal(0, 1.0, n)
        p_mar = 1 / (1 + np.exp(2.0 * (gdp - 0.3)))  # 低GDPほど欠測しやすい(観測済み共変量にのみ依存=MAR)
        miss_mar = r.uniform(0, 1, n) < p_mar
        miss_mcar = r.uniform(0, 1, n) < miss_mar.mean()  # 欠測率をMARと揃えたMCAR

        def cc(miss):
            x, yy = gdp[~miss], y[~miss]
            b1 = np.polyfit(x, yy, 1)[0]
            return b1, yy.mean()

        b1_mcar, mean_mcar = cc(miss_mcar)
        b1_mar, mean_mar = cc(miss_mar)
        return b1_mcar, mean_mcar, b1_mar, mean_mar

    reps = 300
    res = np.array([one_rep(1000 + i) for i in range(reps)])
    b1_mcar_avg, mean_mcar_avg = res[:, 0].mean(), res[:, 1].mean()
    b1_mar_avg, mean_mar_avg = res[:, 2].mean(), res[:, 3].mean()

    bias_beta1 = [b1_mcar_avg - true_beta1, b1_mar_avg - true_beta1]
    bias_mean = [mean_mcar_avg - true_mean_y, mean_mar_avg - true_mean_y]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["MCAR", "MAR\n(観測済みGDPに依存)"]
    x_pos = np.arange(2)
    width = 0.35
    ax.bar(x_pos - width / 2, bias_beta1, width=width, color=COLOR_OK, label="回帰係数beta1のバイアス")
    ax.bar(x_pos + width / 2, bias_mean, width=width, color=COLOR_DIVERGENT, label="周辺平均のバイアス")
    ax.axhline(0, color="black", lw=0.8)
    for i, (b, m) in enumerate(zip(bias_beta1, bias_mean)):
        ax.annotate(f"{b:+.3f}", (i - width / 2, b), xytext=(0, 4 if b >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{m:+.3f}", (i + width / 2, m), xytext=(0, 4 if m >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"完全ケース分析(CC)のバイアス({reps}回反復平均)")
    ax.set_title("回帰係数はMCAR/MARどちらでもほぼ不偏だが、\n周辺平均のバイアスはMARで大きく残る")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "missing_mechanism_asymmetric_bias.png")
    plt.close(fig)

    print(f"missing_mechanism_asymmetric_bias.png saved "
          f"({reps}回反復平均: beta1バイアス MCAR={bias_beta1[0]:+.4f} MAR={bias_beta1[1]:+.4f}, "
          f"周辺平均バイアス MCAR={bias_mean[0]:+.4f} MAR={bias_mean[1]:+.4f})")


def plot_pareto_khat_summary_vs_pointwise():
    """外れ値混入データにNormal尤度の線形回帰をフィットし、PSIS-LOOの
    Pareto k_hat診断で多数の観測点が警告閾値0.7を超える一方、要約統計量
    elpd_diff(頑健なStudentT尤度モデルとの比較)はdseに対し明確に有意で
    あり続けることを示す。要約レベルの判断と個票レベルの判断は別物である。"""

    rng = np.random.default_rng(23)
    n = 40
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 0.25, n)
    out_idx = rng.choice(n, size=5, replace=False)
    y[out_idx] += rng.choice([-1, 1], 5) * rng.uniform(25, 40, 5)  # 少数の極端な外れ値

    def fit_normal():
        with pm.Model():
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 5)
            sigma = pm.HalfNormal("sigma", 0.4)  # 事前分布で分散を膨らませて外れ値を吸収させない
            pm.Normal("y", mu=a + b * x, sigma=sigma, observed=y)
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.97,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
            pm.compute_log_likelihood(idata)
        return idata

    def fit_studentt():
        with pm.Model():
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 5)
            sigma = pm.HalfNormal("sigma", 0.4)
            nu = pm.Gamma("nu", 2, 0.5)
            pm.StudentT("y", nu=nu, mu=a + b * x, sigma=sigma, observed=y)
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.97,
                               random_seed=2, progressbar=False,
                               compute_convergence_checks=False)
            pm.compute_log_likelihood(idata)
        return idata

    idata_normal = fit_normal()
    idata_studentt = fit_studentt()

    loo_normal = az.loo(idata_normal, pointwise=True)
    khat = loo_normal.pareto_k.values
    n_warn = int((khat > 0.7).sum())

    cmp = az.compare({"Normal尤度": idata_normal, "StudentT尤度": idata_studentt}, round_to=6)
    diff = abs(float(cmp.loc["Normal尤度", "elpd_diff"])) if "Normal尤度" in cmp.index and cmp.loc["Normal尤度", "elpd_diff"] != 0 else abs(float(cmp.loc["StudentT尤度", "elpd_diff"]))
    dse = float(cmp.loc["Normal尤度", "dse"]) if cmp.loc["Normal尤度", "dse"] != 0 else float(cmp.loc["StudentT尤度", "dse"])
    ratio = diff / dse if dse > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].hist(khat, bins=30, color=COLOR_ALT, alpha=0.75)
    axes[0].axvline(0.7, color=COLOR_DIVERGENT, lw=1.5, ls="--", label="警告閾値k_hat=0.7")
    axes[0].set_xlabel("Pareto k_hat(観測点ごと)")
    axes[0].set_ylabel("観測点数")
    axes[0].set_title(f"{n}点中{n_warn}点({n_warn/n:.0%})がk_hat>0.7の警告\n(個票レベルでは信頼性に留保が必要)")
    axes[0].legend(fontsize=9)

    axes[1].bar(["Normal vs StudentT"], [diff], yerr=[dse], capsize=8, color=COLOR_OK, width=0.4)
    axes[1].annotate(f"|elpd_diff|={diff:.1f}\ndse={dse:.1f}\n({ratio:.1f}倍)",
                      (0, diff + dse), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=10)
    axes[1].set_ylabel("|elpd_diff|(誤差棒はdse)")
    axes[1].set_ylim(0, (diff + dse) * 1.4)
    axes[1].set_title("要約統計量(elpd_diff)は\ndseに対し明確に有意なまま")

    fig.suptitle("Pareto k_hatの警告が多くても、要約統計量の有意性判断は別に成立しうる", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "pareto_khat_summary_vs_pointwise.png")
    plt.close(fig)

    print(f"pareto_khat_summary_vs_pointwise.png saved "
          f"(n={n}, k_hat>0.7: {n_warn}件({n_warn/n:.1%}), "
          f"Normal vs StudentT |elpd_diff|={diff:.1f} dse={dse:.1f} 比={ratio:.1f}x)")


def _rbf_kernel_nd(a, b, ls=1.0, var=1.0):
    sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return var * np.exp(-0.5 * sq / ls ** 2)


def _gp_posterior_nd(x_train, y_train, x_test, ls=1.0, var=1.0, noise=1e-3):
    K = _rbf_kernel_nd(x_train, x_train, ls, var) + noise * np.eye(len(x_train))
    K_s = _rbf_kernel_nd(x_train, x_test, ls, var)
    K_ss_diag = var * np.ones(len(x_test))
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    v = K_ss_diag - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
    return mu, np.sqrt(np.clip(v, 1e-12, None))


def _expected_improvement_nd(mu, sigma, y_best, xi=0.01):
    from scipy import stats as sstats
    imp = mu - y_best - xi
    z = imp / np.maximum(sigma, 1e-9)
    return imp * sstats.norm.cdf(z) + sigma * sstats.norm.pdf(z)


def _run_bo_nd(objective, true_objective, D, bounds, n_init, n_iter, rng, n_candidates=2000, ls=1.5):
    """乱数候補点集合上でEIを最大化するnext-point選択で、次元Dに依らず
    動くシンプルなGP-EIベイズ最適化(グリッド探索ではなくランダム候補点)。"""
    lo, hi = bounds
    x_train = rng.uniform(lo, hi, size=(n_init, D))
    y_train = objective(x_train)

    def cur_regret():
        best_idx = np.argmax(y_train)
        return float(0.0 - true_objective(x_train[best_idx:best_idx + 1])[0])

    regrets = [cur_regret()]
    for _ in range(n_iter):
        cand = rng.uniform(lo, hi, size=(n_candidates, D))
        mu, sigma = _gp_posterior_nd(x_train, y_train, cand, ls=ls)
        ei = _expected_improvement_nd(mu, sigma, y_train.max())
        x_next = cand[np.argmax(ei)]
        y_next = objective(x_next[None, :])
        x_train = np.vstack([x_train, x_next])
        y_train = np.append(y_train, y_next)
        regrets.append(cur_regret())
    return np.array(regrets)


def plot_dimension_curse_iteration_count():
    """1次元と4次元の単峰の目的関数に対し、同じGP-EIベイズ最適化を実際に
    走らせ、真の最適点に対するregret(真の目的関数値での劣化)の推移を
    比較する。1次元はごく少数の反復でregretがほぼ0まで下がるが、4次元は
    同程度のregretに到達するまでに大幅に多くの反復数を要することを示す。
    次元の呪いは「BOが機能しなくなること」ではなく「必要な評価回数の増加」
    として現れるという実例。"""

    rng = np.random.default_rng(111)
    n_iter = 40
    thresh = 0.05

    def make_objective(center, noise=0.02):
        def obj(X):
            return -np.sum((X - center[None, :]) ** 2, axis=1) + rng.normal(0, noise, size=X.shape[0])
        return obj

    def make_true_objective(center):
        def obj(X):
            return -np.sum((X - center[None, :]) ** 2, axis=1)
        return obj

    D1 = 1
    center1 = rng.uniform(-2, 2, D1)
    regret1 = _run_bo_nd(make_objective(center1), make_true_objective(center1), D1, (-3, 3), 3, n_iter, rng)

    D4 = 4
    center4 = rng.uniform(-2, 2, D4)
    regret4 = _run_bo_nd(make_objective(center4), make_true_objective(center4), D4, (-3, 3), 3, n_iter, rng)

    def iters_to_threshold(regret, thresh):
        idx = np.where(regret < thresh)[0]
        return int(idx[0]) if len(idx) > 0 else None

    it1 = iters_to_threshold(regret1, thresh)
    it4 = iters_to_threshold(regret4, thresh)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.arange(len(regret1))
    ax.plot(xs, np.maximum(regret1, 1e-4), "o-", color=COLOR_OK, ms=4, label=f"1次元(regret<{thresh}到達: {it1}反復目)")
    ax.plot(xs, np.maximum(regret4, 1e-4), "s-", color=COLOR_DIVERGENT, ms=4,
            label=f"4次元(regret<{thresh}到達: {'到達せず' if it4 is None else str(it4)+'反復目'})")
    ax.axhline(thresh, color="black", lw=1, ls="--", label=f"regret={thresh}の目安")
    ax.set_yscale("log")
    ax.set_xlabel("評価回数(初期点+反復)")
    ax.set_ylabel("regret(真の最適値との差、対数軸)")
    ax.set_title("次元の呪いは「必要な評価回数の増加」として現れる\n"
                  f"(1次元は数回で収束、4次元は最終regret={regret4[-1]:.3f}まで多くの反復を要する)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dimension_curse_iteration_count.png")
    plt.close(fig)

    print(f"dimension_curse_iteration_count.png saved "
          f"(1D: regret<{thresh}到達={it1}反復目, 最終regret={regret1[-1]:.5f}; "
          f"4D: regret<{thresh}到達={it4}, 最終regret={regret4[-1]:.5f})")


def _acf_abs(x, lag):
    x = np.abs(x) - np.abs(x).mean()
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x ** 2))


def plot_significant_param_vs_solved_problem():
    """真の負のleverage効果(rho_true=-0.7)を持つが、周期8ステップの
    季節変動(モデルには含めない)も併せ持つ合成リターンに対し、leverage項
    付きSVモデル(rho)を実際にPyMCでフィットする(pytensor.scanで再帰を
    実装し実用的な速度を確保)。rhoの事後は0を含まず統計的に明確な値を
    示す(真のleverage効果を正しく検出)一方、モデルが仮定していない季節的な
    |リターン|の自己相関(ACF、遅れ8)は事後予測レプリカの範囲外にあり、
    rhoが有意でも当初の診断問題(ACFのギャップ)は未解決のままであることを示す。"""

    rng = np.random.default_rng(91)
    T = 400
    phi_true = 0.9
    sigma_eta_true = 0.2
    mu_h_base = -1.0
    rho_true = -0.7
    seasonal_period = 8
    seasonal_amp = 0.9

    seasonal = seasonal_amp * np.sin(2 * np.pi * np.arange(T) / seasonal_period)
    h = np.zeros(T)
    eps = np.zeros(T)
    returns = np.zeros(T)
    h_ar_val = mu_h_base
    h[0] = h_ar_val + seasonal[0]
    eps[0] = rng.normal(0, 1)
    returns[0] = eps[0] * np.exp(h[0] / 2)
    for t in range(1, T):
        z = rng.normal(0, 1)
        eta_t = rho_true * eps[t - 1] + np.sqrt(1 - rho_true ** 2) * z
        h_ar_val = mu_h_base + phi_true * (h_ar_val - mu_h_base) + sigma_eta_true * eta_t
        h[t] = h_ar_val + seasonal[t]
        eps[t] = rng.normal(0, 1)
        returns[t] = eps[t] * np.exp(h[t] / 2)

    returns_t = pt.as_tensor_variable(returns)
    with pm.Model():
        mu_h = pm.Normal("mu_h", -1.0, 2.0)
        phi_raw = pm.Beta("phi_raw", 20, 1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma_eta = pm.HalfNormal("sigma_eta", 0.5)
        rho = pm.Uniform("rho", -0.99, 0.99)
        h_raw = pm.Normal("h_raw", 0, 1, shape=T)
        h0 = mu_h + sigma_eta * h_raw[0] / pt.sqrt(1 - phi ** 2)

        def step(r_prev, h_raw_t, h_prev, mu_h_, phi_, sigma_eta_, rho_):
            eps_prev = r_prev * pt.exp(-h_prev / 2)
            eta_t = rho_ * eps_prev + pt.sqrt(1 - rho_ ** 2) * h_raw_t
            return mu_h_ + phi_ * (h_prev - mu_h_) + sigma_eta_ * eta_t

        h_rest, _ = pytensor.scan(fn=step, sequences=[returns_t[:-1], h_raw[1:]], outputs_info=[h0],
                                   non_sequences=[mu_h, phi, sigma_eta, rho], strict=True)
        h_expr = pt.concatenate([[h0], h_rest])
        pm.Deterministic("h", h_expr)
        pm.Normal("returns", mu=0, sigma=pt.exp(h_expr / 2), observed=returns)
        idata = pm.sample(600, tune=1000, chains=2, target_accept=0.9,
                           random_seed=4, progressbar=False,
                           compute_convergence_checks=False)

    rho_draws = idata.posterior["rho"].values.flatten()
    rho_mean = float(rho_draws.mean())
    rho_lo, rho_hi = np.percentile(rho_draws, [2.5, 97.5])
    rho_significant = not (rho_lo <= 0 <= rho_hi)

    lag = seasonal_period
    data_acf = _acf_abs(returns, lag)
    h_draws = idata.posterior["h"].values.reshape(-1, T)
    rng2 = np.random.default_rng(50)
    rep_acfs = np.array([_acf_abs(rng2.normal(0, np.exp(h_draws[i] / 2)), lag)
                          for i in range(0, h_draws.shape[0], 10)])
    rep_lo, rep_hi = np.percentile(rep_acfs, [2.5, 97.5])
    acf_resolved = rep_lo <= data_acf <= rep_hi

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].hist(rho_draws, bins=40, color=COLOR_ALT, alpha=0.75, density=True)
    axes[0].axvline(0, color="black", lw=1, ls="--", label="rho=0(効果なし)")
    axes[0].axvline(rho_mean, color=COLOR_DIVERGENT, lw=1.5, label=f"事後平均={rho_mean:.2f}")
    axes[0].axvspan(rho_lo, rho_hi, color=COLOR_DIVERGENT, alpha=0.12, label=f"95%CI=[{rho_lo:.2f},{rho_hi:.2f}]")
    axes[0].set_xlabel("leverage効果 rho")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"rhoの事後は0を含まず統計的に明確\n(真値={rho_true}、正しく検出できている)")
    axes[0].legend(fontsize=8)

    axes[1].hist(rep_acfs, bins=30, color=COLOR_OK, alpha=0.7, density=True,
                 label="事後予測レプリカのACF分布")
    axes[1].axvline(data_acf, color=COLOR_DIVERGENT, lw=2, label=f"実データのACF={data_acf:.3f}")
    axes[1].axvspan(rep_lo, rep_hi, color=COLOR_OK, alpha=0.12, label=f"95%範囲=[{rep_lo:.2f},{rep_hi:.2f}]")
    axes[1].set_xlabel(f"ACF(|リターン|, 遅れ{lag})")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"季節的な|リターン|の自己相関は\nモデルの再現範囲外のまま(未解決)")
    axes[1].legend(fontsize=8)

    fig.suptitle("「有意なパラメータ」と「狙っていた問題の解決」は別軸で検証する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "significant_param_vs_solved_problem.png")
    plt.close(fig)

    print(f"significant_param_vs_solved_problem.png saved "
          f"(rho事後平均={rho_mean:.3f} 95%CI=[{rho_lo:.3f},{rho_hi:.3f}] 有意={rho_significant}, "
          f"データACF={data_acf:.3f} モデル95%範囲=[{rep_lo:.3f},{rep_hi:.3f}] 解決={acf_resolved})")


def plot_noisy_objective_insearch_vs_final():
    """ノイズの大きい1次元目的関数に対し、GP-EIベイズ最適化とランダムサーチを
    実際に走らせ、探索中の観測値(ノイズ込み)による running-best比較と、
    探索終了後に真の(ノイズなし)目的関数で評価するclean最終比較を対比する。
    探索中の指標ではランダムサーチの方がノイズの当たりで長く優勢に見える
    局面があるが、探索終了後のclean評価ではベイズ最適化が選んだ点の方が
    真の値で明確に優れていることを示す。"""

    rng = np.random.default_rng(3)
    noise_sigma = 0.8
    bounds = (-4, 8)
    lo, hi = bounds
    n_init, n_iter = 3, 25

    def f_true(x):
        return -(x[:, 0] - 2.0) ** 2

    def f_noisy(x):
        return f_true(x) + rng.normal(0, noise_sigma, x.shape[0])

    x_train = rng.uniform(lo, hi, (n_init, 1))
    y_train = f_noisy(x_train)
    best_noisy_bo = [float(y_train.max())]
    for _ in range(n_iter):
        cand = rng.uniform(lo, hi, (1500, 1))
        mu, sigma = _gp_posterior_nd(x_train, y_train, cand, ls=1.5, noise=noise_sigma ** 2)
        ei = _expected_improvement_nd(mu, sigma, y_train.max())
        x_next = cand[np.argmax(ei)]
        y_next = f_noisy(x_next[None, :])
        x_train = np.vstack([x_train, x_next])
        y_train = np.append(y_train, y_next)
        best_noisy_bo.append(float(y_train.max()))
    x_bo_final = x_train[np.argmax(y_train)]
    clean_bo = float(f_true(x_bo_final[None, :])[0])

    x_rand = rng.uniform(lo, hi, (n_init, 1))
    y_rand = f_noisy(x_rand)
    best_noisy_rand = [float(y_rand.max())]
    for _ in range(n_iter):
        x_next = rng.uniform(lo, hi, (1, 1))
        y_next = f_noisy(x_next)
        x_rand = np.vstack([x_rand, x_next])
        y_rand = np.append(y_rand, y_next)
        best_noisy_rand.append(float(y_rand.max()))
    x_rand_final = x_rand[np.argmax(y_rand)]
    clean_rand = float(f_true(x_rand_final[None, :])[0])

    best_noisy_bo = np.array(best_noisy_bo)
    best_noisy_rand = np.array(best_noisy_rand)
    rand_ahead = int((best_noisy_rand > best_noisy_bo).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    it = np.arange(len(best_noisy_bo))
    axes[0].plot(it, best_noisy_bo, "o-", color=COLOR_OK, ms=4, label="ベイズ最適化(GP-EI)")
    axes[0].plot(it, best_noisy_rand, "s-", color=COLOR_DIVERGENT, ms=4, label="ランダムサーチ")
    axes[0].set_xlabel("評価回数(初期点+反復)")
    axes[0].set_ylabel("running-best(ノイズ込みの観測値)")
    axes[0].set_title(f"探索中の指標(ノイズ込み)では10回目以降\nほぼ同水準で並ぶ({rand_ahead}/{len(it)}回はランダムサーチが上回る)")
    axes[0].legend(fontsize=9, loc="lower right")

    inset = axes[0].inset_axes([0.42, 0.12, 0.5, 0.42])
    inset.plot(it[9:], best_noisy_bo[9:], "o-", color=COLOR_OK, ms=3)
    inset.plot(it[9:], best_noisy_rand[9:], "s-", color=COLOR_DIVERGENT, ms=3)
    inset.set_title("拡大(9回目以降)", fontsize=8)
    inset.tick_params(labelsize=7)

    labels = ["ベイズ最適化", "ランダムサーチ"]
    axes[1].bar(labels, [clean_bo, clean_rand], color=[COLOR_OK, COLOR_DIVERGENT], width=0.5)
    for i, v in enumerate([clean_bo, clean_rand]):
        axes[1].annotate(f"{v:.3f}", (i, v), xytext=(0, 4 if v >= 0 else -14),
                          textcoords="offset points", ha="center", fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("探索終了後にノイズなしの真の目的関数で再評価")
    axes[1].set_title("cleanな最終評価では\nベイズ最適化が選んだ点が明確に優れる")

    fig.suptitle("ノイズのある目的関数では、探索中の指標と最終指標を区別する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "noisy_objective_insearch_vs_final.png")
    plt.close(fig)

    print(f"noisy_objective_insearch_vs_final.png saved "
          f"(探索中: ランダムサーチが{rand_ahead}/{len(it)}回優勢, "
          f"clean最終評価: BO={clean_bo:.4f} Random={clean_rand:.4f}, "
          f"x_bo_final={float(x_bo_final[0]):.3f} x_rand_final={float(x_rand_final[0]):.3f})")


if __name__ == "__main__":
    plot_cumulative_effect_variance_growth()
    plot_cindex_brier_independence()
    plot_placebo_false_positive_rate()
    plot_loo_dse_comparison()
    plot_mde_power_curve()
    plot_ar1_phi_persistence()
    plot_gp_extrapolation_behavior()
    plot_loo_variance_separation()
    plot_pareto_khat_summary_vs_pointwise()
