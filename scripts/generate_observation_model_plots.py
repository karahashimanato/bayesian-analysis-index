"""
techniques/observation-model.md に埋め込む可視化画像を生成するスクリプト。

1. 「右側打ち切りを尤度に直接組み込む」の実例として、打ち切りを正しく
   尤度に組み込んだモデルと、打ち切りを無視してイベントとして扱った
   ナイーブなモデルの生存曲線を比較する。
2. 「離散潜在状態はforward algorithmで周辺化する」の実例として、
   2レジームのMarkov-Switchingモデルをforward algorithmで周辺化した
   尤度でフィットし、推定パラメータから復元したレジーム確率と真のレジームを
   比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_observation_model_plots.py

出力先: assets/observation-model/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


if __name__ == "__main__":
    plot_censoring_bias()
    plot_markov_switching_forward_algorithm()
