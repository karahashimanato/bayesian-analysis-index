"""
techniques/data-pitfalls.md に埋め込む可視化画像を生成するスクリプト。

「ランダム化されていない観測変数から因果を主張しない」の実例として、
交絡変数が観測変数(X)と結果(Y)の両方に影響する状況で、
素朴な回帰(Xのみ)・交絡変数で調整した回帰・ランダム化されたX、の
3通りのベイズ線形回帰の事後分布を比較する。

「データソースの限界(非公式・小サンプル)は分析結果とセットで明示する」の
実例として、同じ真の割合から生成した小サンプル(n=8)と大サンプル(n=200)を
それぞれベイズ二項モデルでフィットし、事後分布の95%信用区間の幅がどれだけ
異なるかを比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_data_pitfalls_plots.py

出力先: assets/data-pitfalls/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "data-pitfalls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_confounding_bias():
    """交絡変数Zが観測変数X(広告接触回数)と結果Y(CVR)の両方に影響する場合、
    Xのみの素朴な回帰は真の効果を過大評価するが、Zで調整するかXをランダム化
    すれば真の効果を回復できることを示す。"""

    rng = np.random.default_rng(0)
    n = 400
    true_beta_x = 0.2   # Xの真の因果効果(小さい)
    beta_z = 1.5         # 交絡変数Zの効果(大きい)
    noise_sd = 0.5

    # 観測データ: Z(顧客の関心度)がXの割り当て(ターゲティング)にも影響する
    z_obs = rng.normal(0, 1, n)
    x_obs = 0.8 * z_obs + rng.normal(0, 0.6, n)  # Xはランダム化されていない
    y_obs = true_beta_x * x_obs + beta_z * z_obs + rng.normal(0, noise_sd, n)

    # ランダム化データ: Xの割り当てがZと独立
    z_rand = rng.normal(0, 1, n)
    x_rand = rng.normal(0, 1, n)  # Zと無相関にランダム割り当て
    y_rand = true_beta_x * x_rand + beta_z * z_rand + rng.normal(0, noise_sd, n)

    def fit_naive(x, y):
        with pm.Model():
            b0 = pm.Normal("b0", 0, 5)
            bx = pm.Normal("bx", 0, 5)
            sigma = pm.HalfNormal("sigma", 2)
            pm.Normal("y", mu=b0 + bx * x, sigma=sigma, observed=y)
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["bx"].values.flatten()

    def fit_adjusted(x, z, y):
        with pm.Model():
            b0 = pm.Normal("b0", 0, 5)
            bx = pm.Normal("bx", 0, 5)
            bz = pm.Normal("bz", 0, 5)
            sigma = pm.HalfNormal("sigma", 2)
            pm.Normal("y", mu=b0 + bx * x + bz * z, sigma=sigma, observed=y)
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["bx"].values.flatten()

    bx_naive = fit_naive(x_obs, y_obs)
    bx_adjusted = fit_adjusted(x_obs, z_obs, y_obs)
    bx_randomized = fit_naive(x_rand, y_rand)  # ランダム化済みなのでXのみでよい

    fig, ax = plt.subplots(figsize=(8, 5.5))
    datasets = [
        ("観測変数(素朴にXのみ回帰)\n= 交絡Zで歪んだ因果主張", bx_naive, COLOR_DIVERGENT),
        ("観測変数(Zで調整して回帰)", bx_adjusted, COLOR_ALT),
        ("ランダム化変数(Xのみ回帰)", bx_randomized, COLOR_OK),
    ]
    positions = [2, 1, 0]
    for (label, samples, color), pos in zip(datasets, positions):
        parts = ax.violinplot([samples], positions=[pos], orientation="horizontal", widths=0.7,
                               showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color(color)
        mean = samples.mean()
        ax.annotate(f"事後平均={mean:.3f}", (mean, pos), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, color=color)

    ax.axvline(true_beta_x, color="black", lw=1.2, ls="--", label=f"真の因果効果={true_beta_x}")
    ax.set_yticks(positions)
    ax.set_yticklabels([d[0] for d in datasets], fontsize=9.5)
    ax.set_xlabel("Xの回帰係数の事後分布")
    ax.set_title("交絡変数のあるXを素朴に回帰すると真の効果を過大評価する\n"
                  "(Zで調整するか、Xをランダム化すれば真の効果を回復できる)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confounding_bias.png")
    plt.close(fig)

    print(f"confounding_bias.png saved "
          f"(真の効果={true_beta_x}, 素朴={bx_naive.mean():.3f}, "
          f"Z調整={bx_adjusted.mean():.3f}, ランダム化={bx_randomized.mean():.3f})")


def plot_small_sample_uncertainty():
    """同じ真の割合から生成した小サンプル(n=8)と大サンプル(n=200)で、
    事後分布の95%信用区間の幅がどれだけ異なるかを示す。"""

    rng = np.random.default_rng(42)
    p_true = 0.15
    n_small, n_large = 8, 200
    k_small = rng.binomial(n_small, p_true)
    k_large = rng.binomial(n_large, p_true)

    def fit(n, k):
        with pm.Model():
            p = pm.Beta("p", 1, 1)
            pm.Binomial("y", n=n, p=p, observed=k)
            idata = pm.sample(3000, tune=1500, chains=4, target_accept=0.9,
                               random_seed=2, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["p"].values.flatten()

    p_small = fit(n_small, k_small)
    p_large = fit(n_large, k_large)

    ci_small = np.percentile(p_small, [2.5, 97.5])
    ci_large = np.percentile(p_large, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xg = np.linspace(0, 1, 300)
    for samples, ci, label, color, n, k in [
        (p_small, ci_small, f"非公式・小サンプル(n={n_small}, k={k_small})", COLOR_DIVERGENT, n_small, k_small),
        (p_large, ci_large, f"公式・大サンプル(n={n_large}, k={k_large})", COLOR_OK, n_large, k_large),
    ]:
        ax.hist(samples, bins=60, density=True, color=color, alpha=0.45,
                label=f"{label}\n事後平均={samples.mean():.3f}, 95%区間=[{ci[0]:.3f},{ci[1]:.3f}](幅{ci[1]-ci[0]:.3f})")
    ax.axvline(p_true, color="black", lw=1.5, ls="--", label=f"真の割合={p_true}")
    ax.set_xlabel("割合 p の事後分布")
    ax.set_ylabel("density")
    ax.set_title("同じ真の値でも、小サンプルの事後分布は\n信用区間が大幅に広い(点推定だけでは区別できない)")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "small_sample_uncertainty.png")
    plt.close(fig)

    print(f"small_sample_uncertainty.png saved "
          f"(真値={p_true}, 小サンプル: 平均={p_small.mean():.3f} 95%区間幅={ci_small[1]-ci_small[0]:.3f}, "
          f"大サンプル: 平均={p_large.mean():.3f} 95%区間幅={ci_large[1]-ci_large[0]:.3f})")


def plot_rate_period_confusion():
    """γ(回復率)と1/γ(平均感染期間)を取り違えると、離散時間SISモデルの
    平衡有病率が生物学的にありえない水準まで歪むことを示す。"""

    beta = 6.0            # 年あたりの実効接触率
    period_years = 0.2    # 真の平均感染期間(年)
    gamma_correct = 1 / period_years            # 正しいgamma = 5.0/年
    gamma_wrong = period_years                  # 誤り: 期間の値をそのままgammaに使う = 0.2/年

    t = np.linspace(0, 5, 500)
    dt = t[1] - t[0]

    def simulate_sis(gamma, i0=0.01):
        i = np.empty_like(t)
        i[0] = i0
        for k in range(1, len(t)):
            di = beta * i[k - 1] * (1 - i[k - 1]) - gamma * i[k - 1]
            i[k] = np.clip(i[k - 1] + di * dt, 0, 1)
        return i

    i_correct = simulate_sis(gamma_correct)
    i_wrong = simulate_sis(gamma_wrong)
    eq_correct = max(0.0, 1 - gamma_correct / beta)
    eq_wrong = max(0.0, 1 - gamma_wrong / beta)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(t, i_correct * 100, color=COLOR_OK, lw=2,
            label=f"正: γ=1/期間={gamma_correct:.1f}/年 → 平衡有病率={eq_correct * 100:.1f}%")
    ax.plot(t, i_wrong * 100, color=COLOR_DIVERGENT, lw=2,
            label=f"誤: γ=期間の値をそのまま使用={gamma_wrong:.1f}/年 → 平衡有病率={eq_wrong * 100:.1f}%")
    ax.set_xlabel("時間(年)")
    ax.set_ylabel("感染割合 I(t) [%]")
    ax.set_title("γ(回復率)と1/γ(感染期間)の取り違えは\nSISモデルの平衡有病率を大きく歪める")
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rate_period_confusion.png")
    plt.close(fig)

    print(f"rate_period_confusion.png saved "
          f"(gamma_correct={gamma_correct:.2f}, eq={eq_correct:.3f}, "
          f"gamma_wrong={gamma_wrong:.2f}, eq={eq_wrong:.3f})")


def plot_population_denominator_error():
    """指標の母数(人口1,000人あたり)を見落としてN=1,000をそのまま使うと、
    観測された報告件数を説明するために有病率パラメータが1(=全員感染)に
    張り付く、生物学的にありえない事後分布になることをPyMCで実際に示す。"""

    rng = np.random.default_rng(9)
    N_correct = 2e8       # 真のリスク人口
    N_wrong = 1000        # 誤って指標の母数をそのまま使った場合
    reporting_rate = 0.02  # 外部文献から得た既知の報告率(固定)
    I_true = 0.05          # 真の有病率(5%、生物学的に妥当な水準)

    lam_true = reporting_rate * N_correct * I_true
    reported_cases = rng.poisson(lam_true)

    def fit(N):
        with pm.Model():
            I = pm.Beta("I", 1, 1)
            lam = reporting_rate * N * I
            pm.Poisson("y", mu=lam, observed=reported_cases)
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=3, progressbar=False,
                               compute_convergence_checks=False)
        return idata.posterior["I"].values.flatten()

    I_post_correct = fit(N_correct)
    I_post_wrong = fit(N_wrong)

    # 報告件数の情報量が大きいため、どちらの事後分布も極端に幅が狭い
    # (ほぼ点推定に近いスパイク)。ヒストグラムでは可視化できないため、
    # 事後平均±95%区間の点推定プロットで表現する。
    ci_correct = np.percentile(I_post_correct, [2.5, 97.5])
    ci_wrong = np.percentile(I_post_wrong, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = [f"正: N={N_correct:.0e}\n(真のリスク人口)", f"誤: N={N_wrong}\n(指標の母数をそのまま使用)"]
    means = [I_post_correct.mean(), I_post_wrong.mean()]
    errs = [[means[0] - ci_correct[0], means[1] - ci_wrong[0]],
            [ci_correct[1] - means[0], ci_wrong[1] - means[1]]]
    colors = [COLOR_OK, COLOR_DIVERGENT]
    for i, (label, mean, color) in enumerate(zip(labels, means, colors)):
        ax.errorbar([i], [mean], yerr=[[errs[0][i]], [errs[1][i]]], fmt="o", color=color,
                    markersize=12, capsize=8, lw=2.5)
        ax.annotate(f"事後平均={mean:.3f}\n95%区間=[{ci_correct[0] if i == 0 else ci_wrong[0]:.3f}, "
                    f"{ci_correct[1] if i == 0 else ci_wrong[1]:.3f}]",
                    (i, mean), xytext=(15, 0), textcoords="offset points", fontsize=9,
                    va="center", color=color)
    ax.axhline(I_true, color="black", lw=1.2, ls="--", label=f"真の有病率={I_true}")
    ax.set_xlim(-0.5, 1.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("有病率 I の事後分布(平均±95%区間)")
    ax.set_title("報告件数は同じでも、母数Nを誤ると有病率の事後分布が\n生物学的にありえない水準(境界のI=1)に張り付く")
    ax.legend(fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "population_denominator_error.png")
    plt.close(fig)

    print(f"population_denominator_error.png saved "
          f"(reported_cases={reported_cases}, I_correct={I_post_correct.mean():.4f}, "
          f"I_wrong={I_post_wrong.mean():.4f})")


def plot_outlier_business_value():
    """接触回数の多い上位1%を機械的に「外れ値」として除外すると、
    実際にはCVRが平均の6倍あるビジネス上の高価値セグメントを
    切り捨ててしまうことを示す。"""

    rng = np.random.default_rng(21)
    n = 20000
    contacts = np.round(rng.lognormal(mean=1.0, sigma=1.0, size=n)).astype(int) + 1
    p99 = np.percentile(contacts, 99)
    is_top = contacts >= p99

    base_cvr = 0.03
    top_cvr = base_cvr * 6
    p_convert = np.where(is_top, top_cvr, base_cvr)
    converted = rng.binomial(1, p_convert)

    # 値の重複(離散カウントデータ)があっても各ビンの件数が均等になるよう、
    # 値でなく順位でビン分割する(百分位が変わらない範囲で上位ほど細かく刻む)
    edge_pcts = [0, 25, 50, 75, 90, 95, 98, 99, 100]
    order = np.argsort(contacts, kind="stable")
    edge_idx = [int(round(p / 100 * n)) for p in edge_pcts]
    labels = [f"{edge_pcts[i]}-{edge_pcts[i + 1]}%" for i in range(len(edge_pcts) - 1)]
    cvr_by_bin = np.array([
        converted[order[edge_idx[i]:edge_idx[i + 1]]].mean()
        for i in range(len(edge_idx) - 1)
    ])

    conversions_lost_fraction = converted[is_top].sum() / converted.sum()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [COLOR_DIVERGENT if lbl == "99-100%" else COLOR_OK for lbl in labels]
    ax.bar(np.arange(len(labels)), cvr_by_bin * 100, color=colors, alpha=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(base_cvr * 100, color="black", lw=1, ls="--", label=f"全体平均CVR={base_cvr * 100:.1f}%")
    ax.set_xlabel("接触回数の百分位ビン(下位→上位)")
    ax.set_ylabel("CVR [%]")
    ax.set_title(f"上位1%(接触回数の多いセグメント)は\nCVRが平均の約6倍(除外すると全コンバージョンの{conversions_lost_fraction * 100:.0f}%を喪失)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "outlier_business_value.png")
    plt.close(fig)

    print(f"outlier_business_value.png saved "
          f"(top1%_cvr={cvr_by_bin[-1]:.4f}, base_cvr={base_cvr:.4f}, "
          f"conversions_lost_if_excluded={conversions_lost_fraction:.3f})")


def plot_stratified_split_variance():
    """単純ランダム分割と層別分割(イベント有無で層別)で、訓練・テスト間の
    イベント発生率の差(ギャップ)のばらつきがどれだけ異なるかを示す。"""

    rng = np.random.default_rng(31)
    n = 2000
    event_rate_true = 0.12
    event = rng.binomial(1, event_rate_true, n)

    idx_all = np.arange(n)
    idx_event = idx_all[event == 1]
    idx_nonevent = idx_all[event == 0]
    test_frac = 0.2
    n_test = int(n * test_frac)
    n_test_event = int(len(idx_event) * test_frac)
    n_test_nonevent = n_test - n_test_event

    n_resamples = 1000
    gap_random = np.empty(n_resamples)
    gap_stratified = np.empty(n_resamples)
    for i in range(n_resamples):
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        gap_random[i] = event[train_idx].mean() - event[test_idx].mean()

        test_event = rng.choice(idx_event, n_test_event, replace=False)
        test_nonevent = rng.choice(idx_nonevent, n_test_nonevent, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_event] = True
        test_mask[test_nonevent] = True
        gap_stratified[i] = event[~test_mask].mean() - event[test_mask].mean()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(gap_random * 100, bins=50, density=True, color=COLOR_DIVERGENT, alpha=0.55,
            label=f"単純ランダム分割(標準偏差={gap_random.std() * 100:.2f}pt、1,000回の再分割)")
    strat_val = gap_stratified.mean() * 100
    ax.axvline(strat_val, color=COLOR_OK, lw=2.5,
               label=f"イベント有無で層別分割(常に差={strat_val:.2f}pt、標準偏差={gap_stratified.std() * 100:.2f}pt)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("訓練 - テストのイベント発生率の差 [pt]")
    ax.set_ylabel("density(ランダム分割)")
    ax.set_title(f"層別分割は訓練・テスト間のイベント発生率の差を\nほぼ一定値に固定する(ランダム分割は再分割のたびに大きくばらつく)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stratified_split_variance.png")
    plt.close(fig)

    print(f"stratified_split_variance.png saved "
          f"(random_std={gap_random.std():.4f}, stratified_std={gap_stratified.std():.4f})")


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


def plot_informative_censoring_exclusion():
    """短い持続時間ほど「ノイズだろう」という恣意的なヒューリスティックで
    除外すると、除外基準そのものが持続時間と相関し、Kaplan-Meier曲線が
    系統的に歪む(informative censoring)ことを示す。客観的なフラグに
    基づく除外はこの歪みを持ち込まない。"""

    rng = np.random.default_rng(41)
    n = 3000
    true_rate = 0.2
    t = rng.exponential(1 / true_rate, n)
    cens_time = 20.0
    event = (t <= cens_time).astype(int)
    t_obs = np.minimum(t, cens_time)

    objective_flag = rng.binomial(1, 0.05, n).astype(bool)
    heuristic_prob = np.clip(0.4 - 0.05 * t_obs, 0.01, 0.5)
    heuristic_flag = rng.binomial(1, heuristic_prob).astype(bool)

    km_t_true, km_s_true = _kaplan_meier(t_obs, event)
    km_t_obj, km_s_obj = _kaplan_meier(t_obs[~objective_flag], event[~objective_flag])
    km_t_heu, km_s_heu = _kaplan_meier(t_obs[~heuristic_flag], event[~heuristic_flag])

    mean_t_true = t_obs.mean()
    mean_t_heu = t_obs[~heuristic_flag].mean()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.step(km_t_true, km_s_true, where="post", color="black", lw=1.8,
            label="除外なし(全データ)")
    ax.step(km_t_obj, km_s_obj, where="post", color=COLOR_OK, lw=1.8, ls="--",
            label="客観的フラグで除外(5%, 持続時間と無関係)")
    ax.step(km_t_heu, km_s_heu, where="post", color=COLOR_DIVERGENT, lw=1.8, ls="--",
            label="恣意的ヒューリスティックで除外(短時間ほど除外されやすい)")
    ax.set_xlabel("持続時間")
    ax.set_ylabel("生存確率")
    ax.set_title(f"持続時間と相関する恣意的な除外基準は生存曲線を上方に歪める\n"
                 f"(平均持続時間: 全データ{mean_t_true:.2f} → ヒューリスティック除外後{mean_t_heu:.2f})")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "informative_censoring_exclusion.png")
    plt.close(fig)

    print(f"informative_censoring_exclusion.png saved "
          f"(mean_t_true={mean_t_true:.3f}, mean_t_objective_excluded={t_obs[~objective_flag].mean():.3f}, "
          f"mean_t_heuristic_excluded={mean_t_heu:.3f})")


def plot_partition_column_illustration():
    """DAY粒度で宣言されたパーティション列の実際の値が月初日に丸められて
    いる場合、日単位のフィルタでは絞り込めず、素朴な期間フィルタが
    フルスキャンになることを示す。"""

    rng = np.random.default_rng(51)
    n = 5000
    day_offsets = rng.uniform(0, 31, n)
    real_day_of_month = np.floor(day_offsets).astype(int) + 1

    scanned_full_gb = 796.0
    n_months = 12
    scanned_pruned_gb = scanned_full_gb / n_months

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.hist(real_day_of_month, bins=np.arange(0.5, 32.5, 1), color=COLOR_OK, alpha=0.8)
    ax.set_xlabel("実際のイベント発生日(月内)")
    ax.set_ylabel("件数")
    ax.set_title("実データは月内の全31日に分散している")

    ax = axes[1]
    bars = ax.bar(["列名を鵜呑みにした\n日単位フィルタ\n(実質フルスキャン)", "パーティション列で\n月単位にプルーニング"],
                   [scanned_full_gb, scanned_pruned_gb], color=[COLOR_DIVERGENT, COLOR_OK], alpha=0.85)
    for b, v in zip(bars, [scanned_full_gb, scanned_pruned_gb]):
        ax.annotate(f"{v:.0f}GB", (b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("スキャン対象データ量 [GB]")
    ax.set_title("パーティション列は実際には月初日に丸められており\n日単位では絞り込めない(粒度は実質月単位)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "partition_column_illustration.png")
    plt.close(fig)

    print(f"partition_column_illustration.png saved "
          f"(scanned_full_gb={scanned_full_gb}, scanned_pruned_gb={scanned_pruned_gb:.1f})")


def plot_propensity_score_conditional():
    """傾向スコア(表示位置に依存して割り当て確率が変わる)を条件によらず
    一様に扱うと、IPS推定量が真の因果効果からズレることを示す。"""

    rng = np.random.default_rng(61)
    positions = np.arange(1, 6)
    true_propensity_by_position = np.array([0.8, 0.5, 0.3, 0.15, 0.08])
    base_ctr_by_position = np.array([0.20, 0.15, 0.10, 0.07, 0.05])
    n_per_position = 4000
    true_uplift = 0.05

    pos_list, treat_list, click_list, e_true_list = [], [], [], []
    for base_ctr, e in zip(base_ctr_by_position, true_propensity_by_position):
        treat = rng.binomial(1, e, n_per_position)
        ctr = base_ctr + true_uplift * treat
        click = rng.binomial(1, ctr)
        treat_list.append(treat)
        click_list.append(click)
        e_true_list.append(np.full(n_per_position, e))
    treat_arr = np.concatenate(treat_list)
    click_arr = np.concatenate(click_list)
    e_true_arr = np.concatenate(e_true_list)

    naive_diff = click_arr[treat_arr == 1].mean() - click_arr[treat_arr == 0].mean()

    e_pooled = treat_arr.mean()
    ips_pooled = (np.mean(treat_arr * click_arr / e_pooled)
                  - np.mean((1 - treat_arr) * click_arr / (1 - e_pooled)))

    ips_correct = (np.mean(treat_arr * click_arr / e_true_arr)
                   - np.mean((1 - treat_arr) * click_arr / (1 - e_true_arr)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["素朴な差分\n(傾向スコア未使用)", "IPS\n(全体で一様な傾向スコア)", "IPS\n(表示位置ごとの傾向スコア)"]
    values = [naive_diff, ips_pooled, ips_correct]
    colors = [COLOR_DIVERGENT, COLOR_ALT, COLOR_OK]
    bars = ax.bar(labels, np.array(values) * 100, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        ax.annotate(f"{v * 100:.2f}pt", (b.get_x() + b.get_width() / 2, v * 100), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.axhline(true_uplift * 100, color="black", lw=1.2, ls="--", label=f"真の効果={true_uplift * 100:.1f}pt")
    ax.set_ylabel("推定された効果 [pt]")
    ax.set_title("表示位置ごとに傾向スコアの分布が異なる状況で、\n条件を無視した補正は真の効果から系統的にズレる")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "propensity_score_conditional.png")
    plt.close(fig)

    print(f"propensity_score_conditional.png saved "
          f"(true={true_uplift:.4f}, naive={naive_diff:.4f}, "
          f"ips_pooled={ips_pooled:.4f}, ips_correct={ips_correct:.4f})")


def plot_mean_imputation_danger():
    """平均補完は欠測値を単一の定数(平均)で埋めるため、その変数が本来
    持っていた分散だけでなく、他の説明変数との相関構造も機械的に破壊する。
    単一変数のみの回帰では平均補完の点推定は完全ケース分析(CC)とほぼ一致
    してしまうため(教科書的な既知の結果)、GDP的な変数X2が別の説明変数X1と
    相関している多変量の設定で、平均補完がX2の回帰係数を過小評価すること
    を示す。"""

    rng = np.random.default_rng(71)
    n = 1000
    x1 = rng.normal(0, 1, n)
    x2_true = 0.6 * x1 + rng.normal(0, 0.8, n)  # X1と相関するX2(GDP相当)
    true_beta1, true_beta2 = 0.4, 0.6
    y = 2 + true_beta1 * x1 + true_beta2 * x2_true + rng.normal(0, 1.0, n)

    missing = rng.binomial(1, 0.35, n).astype(bool)
    x2_obs = x2_true.copy()
    x2_obs[missing] = np.nan

    x2_mean_imputed = x2_obs.copy()
    x2_mean_imputed[missing] = np.nanmean(x2_obs)

    def fit_ols(x1_, x2_, y_):
        X = np.column_stack([np.ones(len(y_)), x1_, x2_])
        coef, *_ = np.linalg.lstsq(X, y_, rcond=None)
        return coef  # [intercept, beta1, beta2]

    beta_full = fit_ols(x1, x2_true, y)
    beta_mean_imputed = fit_ols(x1, x2_mean_imputed, y)
    beta_cc = fit_ols(x1[~missing], x2_obs[~missing], y[~missing])

    var_full = x2_true.var()
    var_mean_imputed = x2_mean_imputed.var()
    var_cc = x2_obs[~missing].var()
    corr_full = np.corrcoef(x1, x2_true)[0, 1]
    corr_mean_imputed = np.corrcoef(x1, x2_mean_imputed)[0, 1]
    corr_cc = np.corrcoef(x1[~missing], x2_obs[~missing])[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    labels = ["完全データ", "平均補完後", "完全ケース(CC)"]
    vars_ = [var_full, var_mean_imputed, var_cc]
    colors = [COLOR_OK, COLOR_DIVERGENT, COLOR_ALT]
    bars = ax.bar(labels, vars_, color=colors, alpha=0.85)
    for b, v, c in zip(bars, vars_, [corr_full, corr_mean_imputed, corr_cc]):
        ax.annotate(f"分散={v:.2f}\ncorr(X1,X2)={c:.2f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("X2(GDP相当)の分散")
    ax.set_title("平均補完はX2の分散と、X1との相関構造を\n機械的に押しつぶす")

    ax = axes[1]
    labels2 = ["真値", "完全データで回帰", "平均補完後に回帰", "CCで回帰"]
    betas = [true_beta2, beta_full[2], beta_mean_imputed[2], beta_cc[2]]
    colors2 = ["black", COLOR_OK, COLOR_DIVERGENT, COLOR_ALT]
    bars2 = ax.bar(labels2, betas, color=colors2, alpha=0.85)
    for b, v in zip(bars2, betas):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("X2の回帰係数 β2")
    ax.set_title("X1と相関するX2を平均補完すると、\n相関構造の破壊によりβ2が真値・CCより過小推定される")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_imputation_danger.png")
    plt.close(fig)

    beta_full, beta_mean_imputed, beta_cc = beta_full[2], beta_mean_imputed[2], beta_cc[2]
    true_beta = true_beta2
    print(f"mean_imputation_danger.png saved "
          f"(beta_true={true_beta}, beta_full={beta_full:.3f}, "
          f"beta_mean_imputed={beta_mean_imputed:.3f}, beta_cc={beta_cc:.3f})")


def _rbf_kernel(a, b, length_scale=1.0, variance=1.0):
    sqdist = (a[:, None] - b[None, :]) ** 2
    return variance * np.exp(-0.5 * sqdist / length_scale ** 2)


def _gp_posterior(x_train, y_train, x_test, length_scale=1.0, variance=1.0, noise=1e-4):
    K = _rbf_kernel(x_train, x_train, length_scale, variance) + noise * np.eye(len(x_train))
    K_s = _rbf_kernel(x_train, x_test, length_scale, variance)
    K_ss_diag = variance * np.ones(len(x_test))
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    var = K_ss_diag - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
    return mu, np.sqrt(np.clip(var, 1e-12, None))


def _expected_improvement(mu, sigma, y_best, xi=0.01):
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return ei


def _run_bo(objective, bounds, n_init, n_iter, rng):
    lo, hi = bounds
    x_train = rng.uniform(lo, hi, n_init)
    y_train = objective(x_train)
    x_grid = np.linspace(lo, hi, 400)
    for _ in range(n_iter):
        mu, sigma = _gp_posterior(x_train, y_train, x_grid)
        ei = _expected_improvement(mu, sigma, y_train.max())
        x_next = x_grid[np.argmax(ei)]
        y_next = objective(np.array([x_next]))[0]
        x_train = np.append(x_train, x_next)
        y_train = np.append(y_train, y_next)
    best_idx = np.argmax(y_train)
    return x_train, y_train, x_train[best_idx], y_train[best_idx]


def plot_bo_boundary_effect():
    """探索範囲を狭く決め打ちしたベイズ最適化(GP代理モデル+期待改善量)は、
    真の最適点が範囲外にある場合でも範囲の端に張り付いた点を「最良」として
    報告してしまうことを示す。範囲を広げると真の最適点付近を発見できる。"""

    rng = np.random.default_rng(81)
    true_optimum_log = np.log10(25.0)

    def objective(u):
        return -2.0 * (u - true_optimum_log) ** 2 + rng.normal(0, 0.03, size=np.shape(u))

    narrow_bounds = (np.log10(1e-3), np.log10(10.0))
    wide_bounds = (np.log10(1e-3), np.log10(50.0))

    x_narrow, y_narrow, xbest_narrow, ybest_narrow = _run_bo(objective, narrow_bounds, 4, 12, rng)
    x_wide, y_wide, xbest_wide, ybest_wide = _run_bo(objective, wide_bounds, 4, 12, rng)

    u_grid = np.linspace(wide_bounds[0], wide_bounds[1], 400)
    f_grid = -2.0 * (u_grid - true_optimum_log) ** 2

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(10 ** u_grid, f_grid, color="black", lw=1.2, alpha=0.5, label="真の目的関数")
    ax.axvline(10 ** narrow_bounds[1], color=COLOR_DIVERGENT, lw=1, ls=":", label="狭い探索範囲の上限(reg_lambda=10)")
    ax.scatter(10 ** x_narrow, y_narrow, color=COLOR_DIVERGENT, alpha=0.5, s=25, label="狭い範囲での評価点")
    ax.scatter([10 ** xbest_narrow], [ybest_narrow], color=COLOR_DIVERGENT, marker="*", s=250,
               edgecolor="black", label=f"狭い範囲での最良点(reg_lambda={10 ** xbest_narrow:.2f}, 範囲の端)")
    ax.scatter(10 ** x_wide, y_wide, color=COLOR_OK, alpha=0.5, s=25, label="広い範囲での評価点")
    ax.scatter([10 ** xbest_wide], [ybest_wide], color=COLOR_OK, marker="*", s=250,
               edgecolor="black", label=f"広い範囲での最良点(reg_lambda={10 ** xbest_wide:.2f}, 内部)")
    ax.set_xscale("log")
    ax.set_xlabel("reg_lambda(log軸)")
    ax.set_ylabel("目的関数(バリデーションスコア相当)")
    ax.set_title("探索範囲の端に最良点が張り付く場合、\n真の最適点が範囲外にある可能性を疑う")
    ax.legend(fontsize=7.5, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bo_boundary_effect.png")
    plt.close(fig)

    print(f"bo_boundary_effect.png saved "
          f"(narrow_best_reg_lambda={10 ** xbest_narrow:.3f}, wide_best_reg_lambda={10 ** xbest_wide:.3f}, "
          f"true_optimum_reg_lambda={10 ** true_optimum_log:.3f})")


def plot_equirectangular_projection_error():
    """等長方形図法近似による緯度経度→平面座標(km)変換の誤差が、対象領域の
    中心からの距離と緯度に応じてどう増加するかを解析的に示す。"""

    R = 6371.0  # 地球半径(km)

    def equirect_dist_km(lat0, lon0, lat2, lon2):
        x = R * np.radians(lon2 - lon0) * np.cos(np.radians(lat0))
        y = R * np.radians(lat2 - lat0)
        return np.hypot(x, y)

    def haversine_km(lat1, lon1, lat2, lon2):
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    distances_km = np.linspace(1, 1200, 200)
    lon0 = 137.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lat0, color, label in [(0.0, COLOR_ALT, "赤道付近(緯度0°)"),
                                (37.5, COLOR_OK, "能登半島付近(緯度37.5°)"),
                                (65.0, COLOR_DIVERGENT, "高緯度(緯度65°)")]:
        rel_errors = []
        for d in distances_km:
            dlon = np.degrees(d / (R * np.cos(np.radians(lat0))))
            lat2, lon2 = lat0, lon0 + dlon
            approx = equirect_dist_km(lat0, lon0, lat2, lon2)
            true = haversine_km(lat0, lon0, lat2, lon2)
            rel_errors.append((approx - true) / true * 100)
        ax.plot(distances_km, rel_errors, color=color, lw=2, label=label)

    ax.axvline(80, color="black", lw=1, ls=":", label="能登半島地震の解析範囲(約80km四方)")
    ax.set_xlabel("領域中心からの距離 [km]")
    ax.set_ylabel("等長方形図法近似の相対誤差 [%]")
    ax.set_title("等長方形図法近似の誤差は対象領域の広がりと緯度に応じて増加する\n(数十〜百km規模の狭い領域では無視できる)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equirectangular_projection_error.png")
    plt.close(fig)

    print("equirectangular_projection_error.png saved")


if __name__ == "__main__":
    plot_confounding_bias()
    plot_small_sample_uncertainty()
    plot_rate_period_confusion()
    plot_population_denominator_error()
    plot_outlier_business_value()
    plot_stratified_split_variance()
    plot_informative_censoring_exclusion()
    plot_partition_column_illustration()
    plot_propensity_score_conditional()
    plot_mean_imputation_danger()
    plot_bo_boundary_effect()
    plot_equirectangular_projection_error()
