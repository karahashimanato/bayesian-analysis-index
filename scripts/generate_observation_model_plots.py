"""
techniques/observation-model.md に埋め込む可視化画像を生成するスクリプト。

1. 「右側打ち切りを尤度に直接組み込む」の実例として、打ち切りを正しく
   尤度に組み込んだモデルと、打ち切りを無視してイベントとして扱った
   ナイーブなモデルの生存曲線を比較する。
2. 「離散潜在状態はforward algorithmで周辺化する」の実例として、
   2レジームのMarkov-Switchingモデルをforward algorithmで周辺化した
   尤度でフィットし、推定パラメータから復元したレジーム確率と真のレジームを
   比較する。
3. 「観測変数がモデルのどの量に対応するかを明示的に検討する」の実例として、
   SEIRのE(t)に対応するはずの観測をI(t)に誤って対応付けると、係数を
   どう調整しても波形が一致しないことを示す。
4. 「同一の観測プロセスなら分布族を統一する」の実例として、同じPoisson
   記録過程から生まれた2つの計数変数に根拠なくNormal尤度を割り当てると
   予測区間の較正が崩れることを示す。
5. 「点過程の対数尤度はpm.Potentialで直接記述する」の実例として、
   Hawkes過程の対数尤度をpm.Potentialで直接書き下し、真の分岐比・減衰率を
   復元できることを示す。
6. 「MNARが疑われる場合は欠測メカニズム自体を尤度に組み込む」の実例として、
   観測のみに基づくナイーブな推定と、Heckman型Selection modelの推定を
   MNARデータで比較する。
7. 「非ガウス尤度でGPを使う場合は潜在関数を明示的にサンプリングする」の
   実例として、pm.gp.Latent+Poisson尤度による正しいアプローチと、
   log(count+1)をガウス近似する簡便法を比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_observation_model_plots.py

出力先: assets/observation-model/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats
from scipy.optimize import minimize

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "observation-model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_censoring_bias():
    """打ち切りを尤度に正しく組み込んだモデルと、打ち切りを無視して
    (打ち切り時刻をイベント時刻として)扱ったナイーブなモデルの
    生存曲線を比較し、ナイーブなモデルが生存確率を過小評価することを示す。"""

    rng = np.random.default_rng(3)
    n = 300
    true_lambda = 0.05  # 定数ハザード(指数分布)

    event_time = rng.exponential(1 / true_lambda, n)
    censor_time = rng.exponential(1 / 0.03, n)  # 独立な打ち切り過程
    obs_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    with pm.Model():
        lam = pm.HalfNormal("lam", 0.2)
        log_lik = event * pt.log(lam) - lam * obs_time  # event*log h(t) + log S(t)
        pm.Potential("lik", log_lik)
        idata_correct = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                                   random_seed=1, progressbar=False,
                                   compute_convergence_checks=False)

    with pm.Model():
        lam = pm.HalfNormal("lam", 0.2)
        # 打ち切りを無視し、全観測を「その時刻に死亡した」として扱う
        log_lik_naive = pt.log(lam) - lam * obs_time
        pm.Potential("lik", log_lik_naive)
        idata_naive = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                                 random_seed=1, progressbar=False,
                                 compute_convergence_checks=False)

    lam_correct = idata_correct.posterior["lam"].values.flatten()
    lam_naive = idata_naive.posterior["lam"].values.flatten()

    t_grid = np.linspace(0, 60, 200)
    surv_true = np.exp(-true_lambda * t_grid)
    surv_correct = np.exp(-lam_correct.mean() * t_grid)
    surv_naive = np.exp(-lam_naive.mean() * t_grid)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(t_grid, surv_true, color="black", lw=1.5, ls="--", label=f"真の生存曲線(λ={true_lambda})")
    ax.plot(t_grid, surv_correct, color=COLOR_OK, lw=2,
             label=f"打ち切りを正しく組み込み(事後平均λ={lam_correct.mean():.4f})")
    ax.plot(t_grid, surv_naive, color=COLOR_DIVERGENT, lw=2,
             label=f"打ち切りを無視(事後平均λ={lam_naive.mean():.4f})")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("生存確率 S(t)")
    ax.set_title(f"打ち切り({(1-event.mean())*100:.0f}%が打ち切り)を無視すると\n生存確率を過小評価する(ハザードを過大評価する)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "censoring_bias.png")
    plt.close(fig)

    print(f"censoring_bias.png saved "
          f"(censored率={(1-event.mean())*100:.1f}%, 真のλ={true_lambda}, "
          f"正しいモデルλ={lam_correct.mean():.4f}, ナイーブλ={lam_naive.mean():.4f})")


def plot_markov_switching_forward_algorithm():
    """2レジームのMarkov-Switchingモデルをforward algorithmで周辺化した
    尤度でフィットし、推定パラメータから復元したレジーム確率と真のレジームを
    比較する。"""

    rng = np.random.default_rng(4)
    n_steps = 150
    p_stay = np.array([0.95, 0.90])  # 各レジームの自己遷移確率
    mu_true = np.array([0.0, 3.0])
    sigma_true = np.array([1.0, 1.0])

    true_state = np.zeros(n_steps, dtype=int)
    for t in range(1, n_steps):
        stay_p = p_stay[true_state[t - 1]]
        true_state[t] = true_state[t - 1] if rng.random() < stay_p else 1 - true_state[t - 1]
    y = rng.normal(mu_true[true_state], sigma_true[true_state])

    def forward_log_marginal(y, p00, p11, mu0, mu1, sigma0, sigma1):
        trans = pt.stack([pt.stack([p00, 1 - p00]), pt.stack([1 - p11, p11])])
        emit0 = pm.logp(pm.Normal.dist(mu=mu0, sigma=sigma0), y)
        emit1 = pm.logp(pm.Normal.dist(mu=mu1, sigma=sigma1), y)
        log_emit = pt.stack([emit0, emit1], axis=1)  # (n_steps, 2)

        log_pi0 = pt.log(pt.stack([0.5, 0.5]))

        def step(log_emit_t, log_alpha_prev, log_trans):
            log_alpha_t = pt.logsumexp(
                log_alpha_prev[:, None] + log_trans, axis=0
            ) + log_emit_t
            return log_alpha_t

        log_alpha_seq, _ = pytensor_scan_wrapper(step, log_emit, log_pi0 + log_emit[0], pt.log(trans))
        log_lik = pt.logsumexp(log_alpha_seq[-1])
        return log_lik

    def pytensor_scan_wrapper(step, log_emit, init, log_trans):
        import pytensor
        result, updates = pytensor.scan(
            fn=step,
            sequences=[log_emit[1:]],
            outputs_info=[init],
            non_sequences=[log_trans],
        )
        return result, updates

    with pm.Model():
        p00 = pm.Beta("p00", 8, 2)
        p11 = pm.Beta("p11", 8, 2)
        mu0 = pm.Normal("mu0", 0, 3)
        mu1 = pm.Normal("mu1", 0, 3)
        sigma0 = pm.HalfNormal("sigma0", 2)
        sigma1 = pm.HalfNormal("sigma1", 2)
        pm.Potential("lik", forward_log_marginal(y, p00, p11, mu0, mu1, sigma0, sigma1))
        idata = pm.sample(2000, tune=1500, chains=4, target_accept=0.95,
                           random_seed=2, progressbar=False,
                           compute_convergence_checks=False)

    post_mean = idata.posterior.mean(dim=("chain", "draw"))
    p00_hat = float(post_mean["p00"])
    p11_hat = float(post_mean["p11"])
    mu_hat = np.array([float(post_mean["mu0"]), float(post_mean["mu1"])])
    sigma_hat = np.array([float(post_mean["sigma0"]), float(post_mean["sigma1"])])
    # ラベルの入れ替わりを補正(mu0<mu1になるよう並べ替え)
    if mu_hat[0] > mu_hat[1]:
        mu_hat = mu_hat[::-1]
        sigma_hat = sigma_hat[::-1]
        p00_hat, p11_hat = p11_hat, p00_hat

    trans_hat = np.array([[p00_hat, 1 - p00_hat], [1 - p11_hat, p11_hat]])
    emit_hat = np.stack([
        (1 / (np.sqrt(2 * np.pi) * sigma_hat[0])) * np.exp(-0.5 * ((y - mu_hat[0]) / sigma_hat[0]) ** 2),
        (1 / (np.sqrt(2 * np.pi) * sigma_hat[1])) * np.exp(-0.5 * ((y - mu_hat[1]) / sigma_hat[1]) ** 2),
    ], axis=1)

    alpha = np.zeros((n_steps, 2))
    alpha[0] = np.array([0.5, 0.5]) * emit_hat[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, n_steps):
        alpha[t] = (alpha[t - 1] @ trans_hat) * emit_hat[t]
        alpha[t] /= alpha[t].sum()
    prob_state1 = alpha[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(y, color=COLOR_CHAIN[0], lw=1, alpha=0.8, label="観測値 y")
    axes[0].fill_between(np.arange(n_steps), y.min() - 1, y.max() + 1,
                          where=true_state == 1, color=COLOR_ALT, alpha=0.15, step="mid",
                          label="真のレジーム1")
    axes[0].set_ylabel("観測値")
    axes[0].set_title("2レジームMarkov-Switchingモデル: forward algorithmで周辺化した尤度でフィット")
    axes[0].legend(fontsize=9, loc="upper right")

    axes[1].plot(prob_state1, color=COLOR_DIVERGENT, lw=1.5, label="推定 P(レジーム1)")
    axes[1].fill_between(np.arange(n_steps), 0, 1, where=true_state == 1,
                          color=COLOR_ALT, alpha=0.15, step="mid", label="真のレジーム1")
    axes[1].set_ylabel("P(レジーム1)")
    axes[1].set_xlabel("時刻 t")
    axes[1].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "markov_switching_forward_algorithm.png")
    plt.close(fig)

    accuracy = ((prob_state1 > 0.5).astype(int) == true_state).mean()
    print(f"markov_switching_forward_algorithm.png saved "
          f"(推定 mu={np.round(mu_hat,3)}, sigma={np.round(sigma_hat,3)}, "
          f"レジーム判定accuracy={accuracy:.3f})")


def plot_seir_observation_mapping():
    """SEIRモデルで観測(日次新規報告数)が本来E(t)からI(t)への遷移
    (sigma*E(t))に対応するはずのところを、誤ってI(t)からR(t)への遷移
    (gamma*I(t))に対応付けてしまうと、比例係数をどう最適化しても
    観測データの時間的な形(ピークのタイミング)に一致しないことを示す。"""

    beta, sigma, gamma = 0.35, 0.20, 0.10
    N = 10_000
    dt = 0.1
    n_steps = 3000

    S, E, I, R = N - 10, 10.0, 0.0, 0.0
    incidence_E = np.zeros(n_steps)  # sigma*E(t): E->Iへの真の遷移量(観測が対応すべき量)
    I_traj = np.zeros(n_steps)
    for t in range(n_steps):
        new_exposed = beta * S * I / N * dt
        new_infectious = sigma * E * dt
        new_recovered = gamma * I * dt
        S += -new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered
        R += new_recovered
        incidence_E[t] = new_infectious / dt
        I_traj[t] = I

    # 1日ごとにサブサンプリング(dt=0.1のため10ステップ=1日)
    days = np.arange(0, n_steps, 10)
    true_daily_new = incidence_E[days]  # 真の観測対応: sigma*E(t)
    I_daily = I_traj[days]

    rng = np.random.default_rng(17)
    obs_daily_new = rng.poisson(true_daily_new)  # 観測ノイズ(Poisson)

    t_days = np.arange(len(days))

    def best_scale(candidate):
        # 最小二乗で最適な比例係数cを求める(候補系列のどんな倍率でも一致するかを検証)
        c = np.sum(candidate * obs_daily_new) / np.sum(candidate ** 2)
        fitted = c * candidate
        rss = np.sum((obs_daily_new - fitted) ** 2)
        return c, fitted, rss

    c_correct, fitted_correct, rss_correct = best_scale(true_daily_new)
    c_wrong, fitted_wrong, rss_wrong = best_scale(I_daily)

    peak_obs = t_days[np.argmax(obs_daily_new)]
    peak_correct = t_days[np.argmax(fitted_correct)]
    peak_wrong = t_days[np.argmax(fitted_wrong)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(t_days, obs_daily_new, s=10, color="black", alpha=0.5, label="観測データ(sigma*E(t)+Poissonノイズ)")
    ax.plot(t_days, fitted_correct, color=COLOR_OK, lw=2,
             label=f"正しい対応付け: c×E(t)(最適c={c_correct:.3f}, RSS={rss_correct:.0f})")
    ax.plot(t_days, fitted_wrong, color=COLOR_DIVERGENT, lw=2,
             label=f"誤った対応付け: c×I(t)(最適c={c_wrong:.3f}, RSS={rss_wrong:.0f})")
    ax.axvline(peak_obs, color="black", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("日数")
    ax.set_ylabel("日次新規報告数")
    ax.set_title(f"観測ピーク={peak_obs}日目に対し、E(t)対応付けのピーク={peak_correct}日目 vs "
                 f"I(t)対応付けのピーク={peak_wrong}日目\n比例係数をどう最適化しても誤った対応付けは波形が一致しない")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seir_observation_mapping.png")
    plt.close(fig)

    print(f"seir_observation_mapping.png saved (RSS: 正しい対応付け={rss_correct:.1f} "
          f"誤った対応付け={rss_wrong:.1f}, 比={rss_wrong/rss_correct:.1f}倍, "
          f"ピークずれ: 正しい={peak_correct-peak_obs}日 誤り={peak_wrong-peak_obs}日)")


def plot_unified_distribution_family():
    """同一のPoisson記録過程から生まれた2つの計数変数(平均が大きく異なる)
    に対し、統一してPoisson尤度を使った場合と、根拠なく一方にNormal尤度を
    割り当てた場合とで、90%予測区間の較正(的中率)を比較する。"""

    rng = np.random.default_rng(19)
    lam_low, lam_high = 2.0, 45.0  # 同じ村の記録過程からの、平均が異なる2変数
    n_train = 300
    n_test = 2000

    y_low_train = rng.poisson(lam_low, n_train)
    y_high_train = rng.poisson(lam_high, n_train)

    lam_low_hat = y_low_train.mean()
    lam_high_hat = y_high_train.mean()
    normal_sigma_low = y_low_train.std(ddof=1)
    normal_sigma_high = y_high_train.std(ddof=1)

    y_low_test = rng.poisson(lam_low, n_test)
    y_high_test = rng.poisson(lam_high, n_test)

    def poisson_interval(lam_hat, alpha=0.10):
        lo = stats.poisson.ppf(alpha / 2, lam_hat)
        hi = stats.poisson.ppf(1 - alpha / 2, lam_hat)
        return lo, hi

    def normal_interval(mu_hat, sigma_hat, alpha=0.10):
        z = stats.norm.ppf(1 - alpha / 2)
        return mu_hat - z * sigma_hat, mu_hat + z * sigma_hat

    def coverage(y, lo, hi):
        return float(np.mean((y >= lo) & (y <= hi)))

    lo_p_low, hi_p_low = poisson_interval(lam_low_hat)
    lo_p_high, hi_p_high = poisson_interval(lam_high_hat)
    lo_n_low, hi_n_low = normal_interval(lam_low_hat, normal_sigma_low)
    lo_n_high, hi_n_high = normal_interval(lam_high_hat, normal_sigma_high)

    cov_p_low = coverage(y_low_test, lo_p_low, hi_p_low)
    cov_p_high = coverage(y_high_test, lo_p_high, hi_p_high)
    cov_n_low = coverage(y_low_test, lo_n_low, hi_n_low)
    cov_n_high = coverage(y_high_test, lo_n_high, hi_n_high)

    # Normal尤度が「あり得ない負の計数」に割り当てる確率質量(構造的な不整合の直接的な証拠)
    p_negative_low = float(stats.norm.cdf(0, lam_low_hat, normal_sigma_low))
    p_negative_high = float(stats.norm.cdf(0, lam_high_hat, normal_sigma_high))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bins_low = np.arange(0, y_low_test.max() + 2) - 0.5
    axes[0].hist(y_low_test, bins=bins_low, density=True, color="lightgray", alpha=0.8,
                 label=f"実データ(平均{lam_low})")
    axes[0].axvspan(lo_p_low, hi_p_low, color=COLOR_OK, alpha=0.25, label=f"Poisson 90%区間(的中率{cov_p_low:.1%})")
    axes[0].axvspan(lo_n_low, hi_n_low, color=COLOR_DIVERGENT, alpha=0.25, label=f"Normal 90%区間(的中率{cov_n_low:.1%})")
    axes[0].axvspan(lo_n_low, 0, color=COLOR_DIVERGENT, alpha=0.5,
                     label=f"Normalがあり得ない負の値に割く確率={p_negative_low:.1%}")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title(f"平均が小さい変数(真のλ={lam_low})")
    axes[0].set_xlabel("計数値")
    axes[0].legend(fontsize=7.5)

    bins_high = np.arange(max(0, y_high_test.min() - 2), y_high_test.max() + 2) - 0.5
    axes[1].hist(y_high_test, bins=bins_high, density=True, color="lightgray", alpha=0.8,
                 label=f"実データ(平均{lam_high})")
    axes[1].axvspan(lo_p_high, hi_p_high, color=COLOR_OK, alpha=0.25, label=f"Poisson 90%区間(的中率{cov_p_high:.1%})")
    axes[1].axvspan(lo_n_high, hi_n_high, color=COLOR_DIVERGENT, alpha=0.25, label=f"Normal 90%区間(的中率{cov_n_high:.1%})")
    axes[1].set_title(f"平均が大きい変数(真のλ={lam_high})")
    axes[1].set_xlabel("計数値")
    axes[1].legend(fontsize=8)

    fig.suptitle("同じ記録過程の2変数に根拠なく異なる分布族を割り当てると較正が崩れる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "unified_distribution_family.png")
    plt.close(fig)

    print(f"unified_distribution_family.png saved (名目90%に対する的中率: "
          f"低平均側 Poisson={cov_p_low:.3f} Normal={cov_n_low:.3f}, "
          f"高平均側 Poisson={cov_p_high:.3f} Normal={cov_n_high:.3f}; "
          f"Normalが負の値に割く確率: 低平均側={p_negative_low:.3f} 高平均側={p_negative_high:.5f})")


def _simulate_hawkes(mu, alpha, beta, T, rng):
    """Ogata's thinning algorithmによる指数核Hawkes過程のシミュレーション。"""
    events = []
    t = 0.0
    while t < T:
        lam = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in events)
        if lam <= 0:
            break
        w = rng.exponential(1.0 / lam)
        t += w
        if t >= T:
            break
        lam_new = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in events)
        if rng.uniform() <= lam_new / lam:
            events.append(t)
    return np.array(events)


def plot_point_process_potential():
    """Hawkes過程(自己励起点過程)のシミュレーションデータに対し、既製の確率
    分布に対応しない対数尤度をpm.Potentialで直接記述してNUTSでフィットし、
    真の分岐比(alpha/beta)・ベースライン強度・減衰率を復元できることを示す。"""

    rng = np.random.default_rng(53)
    mu_true, alpha_true, beta_true = 0.5, 1.2, 2.0  # 分岐比 n=alpha/beta=0.6(安定)
    T = 300.0
    events = _simulate_hawkes(mu_true, alpha_true, beta_true, T, rng)
    n_events = len(events)
    gaps = np.diff(events)

    with pm.Model():
        mu = pm.HalfNormal("mu", 1.0)
        alpha = pm.HalfNormal("alpha", 2.0)
        beta = pm.HalfNormal("beta", 3.0)

        def step(gap, a_prev, beta_):
            return pt.exp(-beta_ * gap) * (1 + a_prev)

        a_seq, _ = pytensor.scan(fn=step, sequences=[gaps],
                                  outputs_info=[pt.constant(0.0, dtype="float64")],
                                  non_sequences=[beta])
        a_full = pt.concatenate([pt.constant([0.0], dtype="float64"), a_seq])

        log_lambda_at_events = pt.log(mu + alpha * a_full)
        compensator = mu * T + (alpha / beta) * pt.sum(1 - pt.exp(-beta * (T - events)))
        log_lik = pt.sum(log_lambda_at_events) - compensator
        pm.Potential("hawkes_loglik", log_lik)

        idata = pm.sample(2000, tune=1500, chains=4, target_accept=0.9,
                           random_seed=5, progressbar=False,
                           compute_convergence_checks=False)

    post = idata.posterior
    mu_hat = float(post["mu"].mean())
    alpha_hat = float(post["alpha"].mean())
    beta_hat = float(post["beta"].mean())
    branching_true = alpha_true / beta_true
    branching_hat_samples = (post["alpha"] / post["beta"]).values.flatten()

    t_grid = np.linspace(0, T, 3000)

    def intensity(t_grid, events, mu, alpha, beta):
        lam = np.full_like(t_grid, mu)
        for ti in events:
            mask = t_grid > ti
            lam[mask] += alpha * np.exp(-beta * (t_grid[mask] - ti))
        return lam

    lam_true = intensity(t_grid, events, mu_true, alpha_true, beta_true)
    lam_hat = intensity(t_grid, events, mu_hat, alpha_hat, beta_hat)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2]})
    axes[0].eventplot(events, color="black", lw=0.6, linelengths=0.8)
    axes[0].set_yticks([])
    axes[0].set_ylabel(f"イベント\n(n={n_events})")
    axes[0].set_title(f"真の分岐比={branching_true:.2f} → 事後平均の分岐比={branching_hat_samples.mean():.3f} "
                       f"(95%区間 [{np.percentile(branching_hat_samples,2.5):.3f}, "
                       f"{np.percentile(branching_hat_samples,97.5):.3f}])")

    axes[1].plot(t_grid, lam_true, color="black", lw=1.2, ls="--", label="真の強度関数 λ(t)")
    axes[1].plot(t_grid, lam_hat, color=COLOR_OK, lw=1.2, alpha=0.9, label="事後平均パラメータでの λ(t)")
    axes[1].set_xlabel("時刻 t")
    axes[1].set_ylabel("強度 λ(t)")
    axes[1].set_xlim(0, 60)  # 見やすさのため先頭60単位時間を拡大
    axes[1].legend(fontsize=9)

    fig.suptitle("Hawkes過程の対数尤度をpm.Potentialで直接記述して真のパラメータを復元", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "point_process_potential.png")
    plt.close(fig)

    print(f"point_process_potential.png saved (n_events={n_events}, "
          f"真値: mu={mu_true} alpha={alpha_true} beta={beta_true} 分岐比={branching_true:.3f}, "
          f"事後平均: mu={mu_hat:.3f} alpha={alpha_hat:.3f} beta={beta_hat:.3f} "
          f"分岐比={branching_hat_samples.mean():.3f})")


def plot_mnar_selection_model():
    """値が大きいほど観測されにくいMNARデータで、観測されたデータだけを
    使うナイーブな推定と、欠測確率自体をΦ(a+γ_y・y)としてモデル化した
    Heckman型Selection modelの推定を比較する。欠測尤度は、観測されない
    値yを周辺化する積分をGauss-Hermite求積で近似して尤度に組み込む。"""

    rng = np.random.default_rng(61)
    mu_true, sigma_true = 10.0, 3.0
    n = 800
    a_true, gamma_true = 0.25, -0.25  # 値が大きいほど観測されにくい(MNAR)
    y_full = rng.normal(mu_true, sigma_true, n)
    p_obs = stats.norm.cdf(a_true + gamma_true * (y_full - mu_true))
    observed = rng.uniform(size=n) < p_obs
    y_obs = y_full[observed]
    n_obs, n_miss = observed.sum(), (~observed).sum()

    naive_mean = float(y_obs.mean())

    # Gauss-Hermite求積ノード(欠測尤度の積分近似に使う、モデルパラメータに依存しない定数)
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(40)
    gh_nodes_t = pt.as_tensor_variable(gh_nodes)
    gh_weights_t = pt.as_tensor_variable(gh_weights)

    def norm_cdf(z):
        return 0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))

    with pm.Model():
        mu = pm.Normal("mu", 10.0, 5.0)
        sigma = pm.HalfNormal("sigma", 5.0)
        a = pm.Normal("a", 0.0, 2.0)
        gamma = pm.Normal("gamma", 0.0, 1.0)

        # 観測された対象: y自体の尤度 x 観測される確率
        log_dens_obs = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), y_obs)
        log_p_obs_given_y = pt.log(norm_cdf(a + gamma * (y_obs - mu)) + 1e-12)
        ll_observed = pt.sum(log_dens_obs + log_p_obs_given_y)

        # 欠測した対象: P(R=0) = ∫ phi(y;mu,sigma)(1-Phi(a+gamma(y-mu))) dy をGH求積で近似
        y_nodes = mu + pt.sqrt(2.0) * sigma * gh_nodes_t
        p_miss_at_node = 1 - norm_cdf(a + gamma * (y_nodes - mu))
        p_missing = pt.sum(gh_weights_t * p_miss_at_node) / pt.sqrt(np.pi)
        ll_missing = n_miss * pt.log(p_missing + 1e-12)

        pm.Potential("loglik", ll_observed + ll_missing)

        idata = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                           random_seed=6, progressbar=False,
                           compute_convergence_checks=False)

    post = idata.posterior
    mu_selection = float(post["mu"].mean())
    mu_selection_samples = post["mu"].values.flatten()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = ["ナイーブ\n(観測データのみ平均)", "Selection model\n(欠測メカニズムを尤度に組込)"]
    ax.axhline(mu_true, color="black", lw=1.3, ls="--", label=f"真の母集団平均={mu_true}")
    ax.scatter([0], [naive_mean], s=160, color=COLOR_DIVERGENT, zorder=3, label="ナイーブ推定(点推定)")
    vp = ax.violinplot([mu_selection_samples], positions=[1], widths=0.5, showmeans=True)
    for body in vp["bodies"]:
        body.set_facecolor(COLOR_OK)
        body.set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("推定された母集団平均")
    ax.set_title(f"MNAR(値が大きいほど観測されにくい, 欠測率{n_miss/n:.0%})の下での推定\n"
                 f"ナイーブ={naive_mean:.3f} vs Selection model事後平均={mu_selection:.3f}(真値={mu_true})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mnar_selection_model.png")
    plt.close(fig)

    print(f"mnar_selection_model.png saved (n={n}, 欠測率={n_miss/n:.3f}, 真の平均={mu_true}, "
          f"ナイーブ推定={naive_mean:.4f}(誤差{naive_mean-mu_true:+.4f}), "
          f"Selection model事後平均={mu_selection:.4f}(誤差{mu_selection-mu_true:+.4f}))")


def plot_gp_latent_poisson_vs_naive_gaussian():
    """Poisson尤度のGP回帰(y ~ Poisson(exp(f(x))))で、正しく`pm.gp.Latent`で
    潜在関数fを明示的にサンプリングするアプローチと、log(count+1)を
    ガウス近似して`pm.gp.Marginal`にそのまま突っ込む簡便法を比較し、
    計数が少ない領域で簡便法が系統的に偏ることを示す。"""

    rng = np.random.default_rng(71)
    x = np.linspace(0, 10, 40)
    f_true = 1.0 + 1.5 * np.sin(0.8 * x)  # 真の対数強度(低計数域を含む)
    lam_true = np.exp(f_true)
    y = rng.poisson(lam_true)

    x_grid = np.linspace(0, 10, 150)
    f_true_grid = 1.0 + 1.5 * np.sin(0.8 * x_grid)

    with pm.Model() as model_latent:
        ell = pm.Gamma("ell", alpha=3, beta=1)
        eta = pm.HalfNormal("eta", 1.5)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=x[:, None])
        pm.Poisson("y", mu=pt.exp(f), observed=y)
        idata_latent = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                                  random_seed=8, progressbar=False,
                                  compute_convergence_checks=False)
        f_pred = gp.conditional("f_pred", Xnew=x_grid[:, None])
        pred_latent = pm.sample_posterior_predictive(idata_latent, var_names=["f_pred"],
                                                       progressbar=False)

    f_latent_mean = pred_latent.posterior_predictive["f_pred"].mean(dim=("chain", "draw")).values

    # 簡便法(誤り): count+1の対数を取ってガウス近似し、pm.gp.Marginalに直接突っ込む
    log_y = np.log(y + 1.0)
    with pm.Model() as model_marginal:
        ell2 = pm.Gamma("ell", alpha=3, beta=1)
        eta2 = pm.HalfNormal("eta", 1.5)
        cov2 = eta2 ** 2 * pm.gp.cov.ExpQuad(1, ell2)
        gp2 = pm.gp.Marginal(cov_func=cov2)
        sigma_n = pm.HalfNormal("sigma_n", 1.0)
        gp2.marginal_likelihood("y_obs", X=x[:, None], y=log_y, sigma=sigma_n)
        idata_marginal = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                                    random_seed=9, progressbar=False,
                                    compute_convergence_checks=False)
        f_pred2 = gp2.conditional("f_pred2", Xnew=x_grid[:, None])
        pred_marginal = pm.sample_posterior_predictive(idata_marginal, var_names=["f_pred2"],
                                                         progressbar=False)

    f_marginal_mean = pred_marginal.posterior_predictive["f_pred2"].mean(dim=("chain", "draw")).values

    low_count_region = f_true_grid < 0.3  # 真の対数強度が低い(計数が少ない)領域
    rmse_latent_low = float(np.sqrt(np.mean((f_latent_mean[low_count_region] - f_true_grid[low_count_region]) ** 2)))
    rmse_marginal_low = float(np.sqrt(np.mean((f_marginal_mean[low_count_region] - f_true_grid[low_count_region]) ** 2)))
    rmse_latent_all = float(np.sqrt(np.mean((f_latent_mean - f_true_grid) ** 2)))
    rmse_marginal_all = float(np.sqrt(np.mean((f_marginal_mean - f_true_grid) ** 2)))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(x, np.log(y + 1e-9).clip(min=-1), s=18, color="black", alpha=0.4,
               label="観測計数(log目盛り, y=0は下限にクリップして表示)")
    ax.plot(x_grid, f_true_grid, color="black", lw=1.5, ls="--", label="真の対数強度 f(x)")
    ax.plot(x_grid, f_latent_mean, color=COLOR_OK, lw=2,
             label=f"pm.gp.Latent+Poisson(全域RMSE={rmse_latent_all:.3f}, 低計数域RMSE={rmse_latent_low:.3f})")
    ax.plot(x_grid, f_marginal_mean, color=COLOR_DIVERGENT, lw=2,
             label=f"log(count+1)をpm.gp.Marginalに直接投入(全域RMSE={rmse_marginal_all:.3f}, 低計数域RMSE={rmse_marginal_low:.3f})")
    ax.fill_between(x_grid, -1.5, 4.5, where=low_count_region, color="gray", alpha=0.1, step="mid")
    ax.set_xlabel("x")
    ax.set_ylabel("対数強度(灰色網掛け=真の低計数域)")
    ax.set_title("非ガウス尤度のGP回帰: 潜在関数を明示的にサンプリングする方が\n低計数域でのバイアスが小さい")
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gp_latent_poisson_vs_naive_gaussian.png")
    plt.close(fig)

    print(f"gp_latent_poisson_vs_naive_gaussian.png saved (全域RMSE: Latent={rmse_latent_all:.4f} "
          f"Marginal(log近似)={rmse_marginal_all:.4f}; 低計数域RMSE: Latent={rmse_latent_low:.4f} "
          f"Marginal(log近似)={rmse_marginal_low:.4f})")


if __name__ == "__main__":
    plot_censoring_bias()
    plot_markov_switching_forward_algorithm()
    plot_seir_observation_mapping()
    plot_unified_distribution_family()
    plot_point_process_potential()
    plot_mnar_selection_model()
    plot_gp_latent_poisson_vs_naive_gaussian()
