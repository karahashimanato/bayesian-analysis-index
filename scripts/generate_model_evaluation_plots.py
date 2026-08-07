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


if __name__ == "__main__":
    plot_cumulative_effect_variance_growth()
    plot_cindex_brier_independence()
    plot_placebo_false_positive_rate()
    plot_loo_dse_comparison()
    plot_mde_power_curve()
    plot_ar1_phi_persistence()
