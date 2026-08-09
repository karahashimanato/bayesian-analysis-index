"""
tools/inference-methods.md に埋め込む可視化画像を生成するスクリプト。

階層Thompson Samplingが「事後分布の確信が強まりすぎて探索が止まる」
ロックインを起こすメカニズムを、早期探索が偏ったスナップショットに対する
「次にその腕が選ばれる確率(P(argmax))」の違いとして、独立モデルと比較して示す。

実行方法:
    source .venv/bin/activate
    python scripts/generate_inference_methods_plots.py

出力先: assets/inference-methods/*.png
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "inference-methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_thompson_sampling_lockin():
    """階層Thompson Samplingが、探索不足の腕を早期の不運な観測データだけで
    「有望でない」と過信し、探索を止めてしまう(ロックイン)ことを示す。"""

    n_arms = 5
    true_p = np.array([0.10, 0.10, 0.10, 0.10, 0.18])
    best_arm = 4
    arm_labels = [f"腕{i}" for i in range(n_arms)]

    rng = np.random.default_rng(8)
    trials = np.array([500, 500, 500, 500, 20])  # 真に最良の腕4だけ試行が少ない
    succ = rng.binomial(trials, true_p)

    with pm.Model():
        mu = pm.Beta("mu", 2, 2)
        kappa = pm.Gamma("kappa", alpha=4.0, beta=0.04)  # 「腕同士は似ている」という事前の想定(平均100)
        alpha = pm.Deterministic("alpha", mu * kappa)
        beta = pm.Deterministic("beta", (1 - mu) * kappa)
        p = pm.Beta("p", alpha=alpha, beta=beta, shape=n_arms)
        pm.Binomial("y", n=trials, p=p, observed=succ)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
    p_hier = idata.posterior["p"].values.reshape(-1, n_arms)
    prob_best_hier = (p_hier.argmax(axis=1)[:, None] == np.arange(n_arms)).mean(axis=0)

    rng_indep = np.random.default_rng(0)
    n_mc = p_hier.shape[0]
    p_indep = rng_indep.beta(1 + succ, 1 + (trials - succ), size=(n_mc, n_arms))
    prob_best_indep = (p_indep.argmax(axis=1)[:, None] == np.arange(n_arms)).mean(axis=0)

    false_leader = int(np.argmax(prob_best_hier[:4]))  # 階層モデルが誤って推す腕

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    x = np.arange(n_arms)
    width = 0.35
    b1 = axes[0].bar(x - width / 2, prob_best_hier, width, color=COLOR_DIVERGENT, label="階層モデル")
    b2 = axes[0].bar(x + width / 2, prob_best_indep, width, color=COLOR_OK, label="独立モデル")
    for bars in (b1, b2):
        for rect in bars:
            axes[0].annotate(f"{rect.get_height():.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                              ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lbl}\nCTR{p:.0%}\nn={t}" for lbl, p, t in zip(arm_labels, true_p, trials)],
                             fontsize=8)
    axes[0].set_ylabel("P(次にこの腕が選ばれる) = P(argmax p)")
    axes[0].set_title(f"独立モデルは真に最良の腕{best_arm}を推すが、\n階層モデルは試行数の多い腕{false_leader}を誤って推す")
    axes[0].legend(fontsize=9)

    for p_samples, color, label, ls in [
        (p_indep[:, best_arm], COLOR_OK, "独立モデル", "-"),
        (p_hier[:, best_arm], COLOR_DIVERGENT, "階層モデル", "-"),
    ]:
        axes[1].hist(p_samples, bins=50, density=True, color=color, alpha=0.5, label=label)
    axes[1].axvline(true_p[best_arm], color="black", lw=1.2, ls="--", label=f"真の値={true_p[best_arm]:.2f}")
    axes[1].set_xlabel("腕4のCTR事後分布")
    axes[1].set_ylabel("density")
    axes[1].set_title("階層モデルは他の腕に引っ張られ\n事後分布が狭く低い位置に固まる")
    axes[1].legend(fontsize=9)

    fig.suptitle("階層Thompson Samplingのロックイン: 探索不足の腕への過信", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "thompson_sampling_lockin.png")
    plt.close(fig)

    print(f"thompson_sampling_lockin.png saved "
          f"(P(腕4選択)| hier={prob_best_hier[best_arm]:.3f}, indep={prob_best_indep[best_arm]:.3f}; "
          f"腕4事後平均| hier={p_hier[:,best_arm].mean():.3f}, indep={p_indep[:,best_arm].mean():.3f})")


def _leapfrog(theta, r, eps, grad_log_p):
    r = r + 0.5 * eps * grad_log_p(theta)
    theta = theta + eps * r
    r = r + 0.5 * eps * grad_log_p(theta)
    return theta, r


def _no_uturn(theta_plus, theta_minus, r_plus, r_minus):
    d = theta_plus - theta_minus
    return (d @ r_minus >= 0) and (d @ r_plus >= 0)


def _build_nuts_trajectory(theta0, eps, grad_log_p, rng, max_doublings=8):
    r0 = rng.normal(size=2)
    theta_minus = theta_plus = theta0.copy()
    r_minus = r_plus = r0.copy()
    trajectory = [theta0.copy()]
    j = 0
    ok = True
    while ok and j < max_doublings:
        direction = 1 if rng.uniform() > 0.5 else -1
        th, r = (theta_plus, r_plus) if direction == 1 else (theta_minus, r_minus)
        new_points = []
        for _ in range(2 ** j):
            th, r = _leapfrog(th, r, direction * eps, grad_log_p)
            new_points.append(th.copy())
        if direction == 1:
            theta_plus, r_plus = th, r
        else:
            theta_minus, r_minus = th, r
        trajectory.extend(new_points)
        ok = _no_uturn(theta_plus, theta_minus, r_plus, r_minus)
        j += 1
    return np.array(trajectory), j


def plot_nuts_trajectory():
    """相関の強い2次元ガウス分布を目標分布に、leapfrog積分による軌道構築と
    "doubling"手続き(木を2倍ずつ伸ばし、U-turn条件で停止)を実際にnumpyで
    実装し、1回のNUTS軌道がどれだけ遠くまで一気に移動するかを、
    ランダムウォークMetropolis法の1ステップと比較する。"""

    rho = 0.9
    cov = np.array([[1.0, rho], [rho, 1.0]])
    cov_inv = np.linalg.inv(cov)

    def grad_log_p(theta):
        return -cov_inv @ theta

    rng = np.random.default_rng(3)
    theta0 = np.array([-1.5, 1.3])
    traj, n_doublings = _build_nuts_trajectory(theta0, 0.15, grad_log_p, rng, max_doublings=8)

    rng_mh = np.random.default_rng(3)
    mh_step = rng_mh.normal(0, 0.3, size=2)
    theta_mh = theta0 + mh_step

    xg = np.linspace(-4, 4, 200)
    yg = np.linspace(-4, 4, 200)
    XG, YG = np.meshgrid(xg, yg)
    pos = np.dstack([XG, YG])
    dens = np.exp(-0.5 * np.einsum("...i,ij,...j->...", pos, cov_inv, pos))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.contour(XG, YG, dens, levels=8, colors="gray", alpha=0.5, linewidths=0.8)
    ax.plot(traj[:, 0], traj[:, 1], "-o", color=COLOR_OK, markersize=4, lw=1.5,
            label=f"NUTS軌道(leapfrog {len(traj) - 1}ステップ, doubling {n_doublings}回)")
    ax.scatter([theta0[0]], [theta0[1]], color="black", s=80, zorder=5, label="開始点")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=COLOR_OK, marker="*", s=250,
               edgecolor="black", zorder=5, label="U-turn検出で停止した終点")
    ax.annotate("", xy=theta_mh, xytext=theta0,
                arrowprops=dict(arrowstyle="->", color=COLOR_DIVERGENT, lw=2))
    ax.scatter([theta_mh[0]], [theta_mh[1]], color=COLOR_DIVERGENT, s=60, zorder=5,
               label="Metropolis-Hastingsの1提案(ランダムウォーク)")
    dist_nuts = np.linalg.norm(traj[-1] - theta0)
    dist_mh = np.linalg.norm(theta_mh - theta0)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(f"NUTSは1回の軌道で{dist_nuts:.2f}移動するが、\nMHの1提案は{dist_mh:.2f}しか移動しない(相関ρ={rho})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nuts_trajectory.png")
    plt.close(fig)

    print(f"nuts_trajectory.png saved (leapfrog_steps={len(traj) - 1}, doublings={n_doublings}, "
          f"dist_nuts={dist_nuts:.3f}, dist_mh={dist_mh:.3f})")


def plot_advi_vs_nuts():
    """相関の強い2次元正規分布にNUTSとmean-field ADVIを実際にPyMCでフィットし、
    パラメータ間の相関を無視するmean-field近似が事後の不確実性(SD)を
    過小評価することを示す。"""

    rng = np.random.default_rng(5)
    rho = 0.95
    cov_true = np.array([[1.0, rho], [rho, 1.0]])
    n = 50
    data = rng.multivariate_normal([2.0, -1.0], cov_true, size=n)

    with pm.Model():
        mu = pm.Normal("mu", 0, 5, shape=2)
        chol, _, _ = pm.LKJCholeskyCov("chol_cov", n=2, eta=2, sd_dist=pm.HalfNormal.dist(1))
        pm.MvNormal("y", mu=mu, chol=chol, observed=data)

        idata = pm.sample(500, tune=500, chains=2, cores=2, target_accept=0.8,
                           random_seed=1, progressbar=False, compute_convergence_checks=False)
        approx = pm.fit(n=15000, method="advi", random_seed=2, progressbar=False)
        idata_advi = approx.sample(2000)

    mu_nuts = idata.posterior["mu"].values.reshape(-1, 2)
    mu_advi = idata_advi.posterior["mu"].values.reshape(-1, 2)
    sd_nuts = mu_nuts.std(axis=0)
    sd_advi = mu_advi.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    ax = axes[0]
    ax.scatter(mu_nuts[:, 0], mu_nuts[:, 1], s=4, alpha=0.25, color=COLOR_OK, label="NUTS")
    ax.scatter(mu_advi[:, 0], mu_advi[:, 1], s=4, alpha=0.25, color=COLOR_DIVERGENT, label="mean-field ADVI")
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("NUTSの事後(相関を保持した細長い分布)と\nADVIの事後(より丸みを帯びた広い分布)")
    ax.legend(fontsize=9, markerscale=3)

    ax = axes[1]
    labels = [r"$\mu_1$のSD", r"$\mu_2$のSD"]
    x = np.arange(2)
    width = 0.35
    b1 = ax.bar(x - width / 2, sd_nuts, width, color=COLOR_OK, label="NUTS")
    b2 = ax.bar(x + width / 2, sd_advi, width, color=COLOR_DIVERGENT, label="mean-field ADVI")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("事後SD")
    ax.set_title(f"この設定ではADVIのSDはNUTSの約{(sd_advi / sd_nuts).mean():.1f}倍と、\nむしろ過大(共分散の不確実性込みで)")
    ax.legend(fontsize=9)

    fig.suptitle(f"mean-field ADVI(相関ρ={rho}のデータ、共分散も同時推定)は\n相関を無視した丸い分布になるが、この設定ではSDの過大評価が起きた", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "advi_vs_nuts_underestimation.png")
    plt.close(fig)

    print(f"advi_vs_nuts_underestimation.png saved "
          f"(sd_nuts={sd_nuts}, sd_advi={sd_advi}, ratio={(sd_nuts / sd_advi)})")


def plot_replay_method():
    """一様ランダムなログ収集方策で集めた合成ログに対し、Replay法で
    epsilon-greedy評価方策の性能を推定する。行動が一致しない行を破棄するため
    有効なラウンド数がログ全体よりかなり少なくなること、それでも推定値が
    理論的な収束値に一致することを示す。"""

    rng = np.random.default_rng(7)
    K = 4
    true_ctr = np.array([0.05, 0.08, 0.12, 0.07])
    n_log = 150000
    logged_arm = rng.integers(0, K, n_log)
    logged_reward = rng.binomial(1, true_ctr[logged_arm])

    eps_greedy = 0.1
    counts = np.ones(K)
    sums = np.zeros(K)
    accepted_rewards = []

    for t in range(n_log):
        if rng.uniform() < eps_greedy:
            a = rng.integers(0, K)
        else:
            a = np.argmax(sums / counts)
        if a == logged_arm[t]:
            r = logged_reward[t]
            counts[a] += 1
            sums[a] += r
            accepted_rewards.append(r)

    accepted_rewards = np.array(accepted_rewards)
    n_accepted = len(accepted_rewards)
    running_mean = np.cumsum(accepted_rewards) / np.arange(1, n_accepted + 1)
    theoretical = (1 - eps_greedy) * true_ctr.max() + eps_greedy * true_ctr.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(np.arange(1, n_accepted + 1), running_mean, color=COLOR_OK, lw=1.2)
    ax.axhline(theoretical, color="black", lw=1.2, ls="--",
               label=f"理論的な収束値={theoretical:.4f}")
    ax.set_xlabel("採用されたラウンド数")
    ax.set_ylabel("Replay法による推定平均報酬")
    ax.set_title(f"採用ラウンドが進むにつれ推定値が\n理論値に収束する(最終値={running_mean[-1]:.4f})")
    ax.legend(fontsize=9)

    ax = axes[1]
    bars = ax.bar(["ログ全体", "採用されたラウンド\n(行動が一致した行)"], [n_log, n_accepted],
                   color=[COLOR_DIVERGENT, COLOR_OK], alpha=0.85)
    for b, v in zip(bars, [n_log, n_accepted]):
        ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("ラウンド数")
    ax.set_title(f"行動が不一致の行は破棄され、\n有効に使えるのは全体の{n_accepted / n_log * 100:.1f}%のみ")

    fig.suptitle("Replay法: 一様ランダムなログから評価方策の性能を推定する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "replay_method.png")
    plt.close(fig)

    print(f"replay_method.png saved (n_log={n_log}, n_accepted={n_accepted}, "
          f"final_estimate={running_mean[-1]:.4f}, theoretical={theoretical:.4f})")


def plot_gp_marginal():
    """ガウス尤度のGP回帰でpm.gp.Marginalを実際にPyMCでフィットし、
    潜在関数fを明示的にサンプリングせず解析的に周辺化した上で
    カーネルハイパーパラメータだけをNUTSでサンプリングすることを示す。"""

    rng = np.random.default_rng(9)
    n = 20
    x = np.sort(rng.uniform(0, 10, n))
    f_true_fn = lambda xx: np.sin(xx) + 0.3 * xx
    y = f_true_fn(x) + rng.normal(0, 0.3, n)

    t0 = time.time()
    with pm.Model() as model:
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.HalfNormal("eta", 1.5)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Marginal(cov_func=cov)
        sigma = pm.HalfNormal("sigma", 0.5)
        gp.marginal_likelihood("y", X=x[:, None], y=y, sigma=sigma)
        idata = pm.sample(400, tune=400, chains=2, cores=2, target_accept=0.8,
                           random_seed=1, progressbar=False, compute_convergence_checks=False)

        x_new = np.linspace(-1, 11, 100)
        f_pred = gp.conditional("f_pred", Xnew=x_new[:, None])
        pred = pm.sample_posterior_predictive(idata, var_names=["f_pred"], progressbar=False)
    t_fit = time.time() - t0

    f_samples = pred.posterior_predictive["f_pred"].values.reshape(-1, len(x_new))
    f_mean = f_samples.mean(axis=0)
    f_lo, f_hi = np.percentile(f_samples, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_new, f_true_fn(x_new), color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x, y, color="black", s=25, zorder=4, label="観測データ")
    ax.plot(x_new, f_mean, color=COLOR_OK, lw=2, label="事後平均")
    ax.fill_between(x_new, f_lo, f_hi, color=COLOR_OK, alpha=0.25, label="95%区間")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"pm.gp.Marginal: 潜在関数fを解析的に周辺化し、\nカーネルHP(ell, eta, sigma)だけをNUTSでサンプリング(N={n}点, {t_fit:.1f}秒)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gp_marginal.png")
    plt.close(fig)

    print(f"gp_marginal.png saved (n={n}, fit_time={t_fit:.1f}s)")


def plot_gp_latent_hsgp():
    """非ガウス尤度(Poisson)のGP回帰で、潜在関数fを明示的にサンプリングする
    pm.gp.Latent(厳密)とpm.gp.HSGP(基底関数近似)を実際にPyMCでフィットし、
    実行時間と復元精度を比較する。"""

    rng = np.random.default_rng(13)
    n = 18
    x = np.sort(rng.uniform(0, 10, n))
    f_true_fn = lambda xx: 1.0 + 0.6 * np.sin(xx)
    log_rate = f_true_fn(x)
    y = rng.poisson(np.exp(log_rate))
    x_new = np.linspace(0, 10, 60)

    t0 = time.time()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.HalfNormal("eta", 1.0)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=x[:, None])
        pm.Poisson("y", mu=pm.math.exp(f), observed=y)
        idata_latent = pm.sample(300, tune=300, chains=2, cores=2, target_accept=0.9,
                                  random_seed=2, progressbar=False, compute_convergence_checks=False)
        f_new_latent = gp.conditional("f_new", Xnew=x_new[:, None])
        pred_latent = pm.sample_posterior_predictive(idata_latent, var_names=["f_new"], progressbar=False)
    t_latent = time.time() - t0

    t0 = time.time()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.HalfNormal("eta", 1.0)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp_hsgp = pm.gp.HSGP(m=[15], c=2.0, cov_func=cov)
        f_hsgp = gp_hsgp.prior("f", X=x[:, None])
        pm.Poisson("y", mu=pm.math.exp(f_hsgp), observed=y)
        idata_hsgp = pm.sample(300, tune=300, chains=2, cores=2, target_accept=0.9,
                                random_seed=3, progressbar=False, compute_convergence_checks=False)
        f_new_hsgp = gp_hsgp.conditional("f_new", Xnew=x_new[:, None])
        pred_hsgp = pm.sample_posterior_predictive(idata_hsgp, var_names=["f_new"], progressbar=False)
    t_hsgp = time.time() - t0

    f_samples_latent = pred_latent.posterior_predictive["f_new"].values.reshape(-1, len(x_new))
    f_samples_hsgp = pred_hsgp.posterior_predictive["f_new"].values.reshape(-1, len(x_new))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_new, f_true_fn(x_new), color="black", lw=1.2, ls="--", label="真の対数強度")
    ax.scatter(x, np.log(y + 0.1), color="gray", s=20, alpha=0.6, label="観測(log(y+0.1))")
    ax.plot(x_new, f_samples_latent.mean(axis=0), color=COLOR_OK, lw=2,
            label=f"pm.gp.Latent(厳密, {t_latent:.1f}秒)")
    ax.plot(x_new, f_samples_hsgp.mean(axis=0), color=COLOR_ALT, lw=2, ls="--",
            label=f"pm.gp.HSGP(近似, {t_hsgp:.1f}秒)")
    ax.set_xlabel("x")
    ax.set_ylabel("対数強度 f(x)")
    ax.set_title(f"厳密GP(pm.gp.Latent)とHSGP近似はほぼ一致する対数強度を復元する\n"
                 f"(N={n}点と少ないため両者とも過度に滑らかで、真の周期変動までは追えない。"
                 f"HSGPは{t_latent / t_hsgp:.1f}倍高速)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gp_latent_vs_hsgp.png")
    plt.close(fig)

    print(f"gp_latent_vs_hsgp.png saved (n={n}, t_latent={t_latent:.1f}s, t_hsgp={t_hsgp:.1f}s, "
          f"speedup={t_latent / t_hsgp:.2f}x)")


def plot_gp_marginal_approx():
    """大規模データに対する厳密GP(pm.gp.Marginal)とVFE誘導点近似スパースGP
    (pm.gp.MarginalApprox)を実際にPyMCでフィットし、実行時間の差を示す。"""

    rng = np.random.default_rng(17)
    n = 120
    x = np.sort(rng.uniform(0, 20, n))
    f_true_fn = lambda xx: np.sin(xx * 0.8) + 0.15 * xx
    y = f_true_fn(x) + rng.normal(0, 0.4, n)
    x_new = np.linspace(-2, 22, 100)

    t0 = time.time()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.HalfNormal("eta", 1.5)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp_exact = pm.gp.Marginal(cov_func=cov)
        sigma = pm.HalfNormal("sigma", 0.5)
        gp_exact.marginal_likelihood("y", X=x[:, None], y=y, sigma=sigma)
        idata_exact = pm.sample(300, tune=300, chains=2, cores=2, target_accept=0.8,
                                 random_seed=4, progressbar=False, compute_convergence_checks=False)
        f_pred_exact = gp_exact.conditional("f_pred", Xnew=x_new[:, None])
        pred_exact = pm.sample_posterior_predictive(idata_exact, var_names=["f_pred"], progressbar=False)
    t_exact = time.time() - t0

    M = 15
    Xu = pm.gp.util.kmeans_inducing_points(M, x[:, None])
    t0 = time.time()
    with pm.Model():
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.HalfNormal("eta", 1.5)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp_vfe = pm.gp.MarginalApprox(cov_func=cov, approx="VFE")
        sigma = pm.HalfNormal("sigma", 0.5)
        gp_vfe.marginal_likelihood("y", X=x[:, None], Xu=Xu, y=y, sigma=sigma)
        idata_vfe = pm.sample(300, tune=300, chains=2, cores=2, target_accept=0.8,
                               random_seed=5, progressbar=False, compute_convergence_checks=False)
        f_pred_vfe = gp_vfe.conditional("f_pred", Xnew=x_new[:, None])
        pred_vfe = pm.sample_posterior_predictive(idata_vfe, var_names=["f_pred"], progressbar=False)
    t_vfe = time.time() - t0

    f_exact = pred_exact.posterior_predictive["f_pred"].values.reshape(-1, len(x_new))
    f_vfe = pred_vfe.posterior_predictive["f_pred"].values.reshape(-1, len(x_new))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x_new, f_true_fn(x_new), color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x, y, color="gray", s=10, alpha=0.5, label="観測データ")
    ax.plot(x_new, f_exact.mean(axis=0), color=COLOR_OK, lw=1.5, label="厳密GP(pm.gp.Marginal)")
    ax.plot(x_new, f_vfe.mean(axis=0), color=COLOR_ALT, lw=1.5, ls="--",
            label=f"VFEスパースGP(M={M}誘導点)")
    ax.scatter(Xu.flatten(), np.full(M, min(y) - 0.5),
               marker="^", color=COLOR_DIVERGENT, s=30, label="誘導点の位置")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"N={n}点に対する厳密GPとVFE近似の予測平均")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    bars = ax.bar(["厳密GP\n(pm.gp.Marginal)", f"VFEスパースGP\n(M={M}誘導点)"],
                   [t_exact, t_vfe], color=[COLOR_OK, COLOR_ALT], alpha=0.85)
    for b, v in zip(bars, [t_exact, t_vfe]):
        ax.annotate(f"{v:.1f}秒", (b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("フィット+予測の合計時間 [秒]")
    ax.set_title(f"この規模(N={n}点、M={M}誘導点)ではVFEはむしろ遅い\n(厳密GPの{t_vfe / t_exact:.1f}倍)")

    fig.suptitle("pm.gp.MarginalApprox(VFE): 誘導点近似の恩恵はNが十分大きい場合に限られる\n"
                 "(小規模ではVFE自体のオーバーヘッドが厳密GPのO(N³)を上回ることがある)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "gp_marginal_approx.png")
    plt.close(fig)

    print(f"gp_marginal_approx.png saved (n={n}, M={M}, t_exact={t_exact:.1f}s, t_vfe={t_vfe:.1f}s, "
          f"speedup={t_exact / t_vfe:.2f}x)")


if __name__ == "__main__":
    plot_thompson_sampling_lockin()
    plot_nuts_trajectory()
    plot_advi_vs_nuts()
    plot_replay_method()
    plot_gp_marginal()
    plot_gp_latent_hsgp()
    plot_gp_marginal_approx()
