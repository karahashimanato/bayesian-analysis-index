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

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "implementation-hacks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


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
    plot_advi_variance_inflation()
    plot_piecewise_exponential_exposure()
    plot_scan_vs_vectorized_hawkes()
