"""
tools/observation-models.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際に階層ベイズモデルをサンプリングし、
  1. Beta-Binomial: 観測回数の少ない個体ほど事後推定が全体平均へ強く縮小する(部分プーリング)
  2. Gamma-Poisson: overdispersedなカウントデータに対し、Poisson固定分散モデルは
     予測区間を過小評価し、Gamma-Poisson階層モデルは正しく捉える
ことを描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_observation_models_plots.py

出力先: assets/observation-models/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "observation-models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_beta_binomial_shrinkage():
    """打率のような個体ごとの成功率を階層Beta-Binomialモデルで推定し、
    観測回数が少ない個体ほど事後推定が全体平均へ強く縮小することを示す。"""

    rng = np.random.default_rng(5)
    J = 18
    n_ab = rng.choice([10, 15, 25, 40, 80, 150, 300, 500], size=J)
    true_mu, true_kappa = 0.27, 80.0
    true_p = rng.beta(true_mu * true_kappa, (1 - true_mu) * true_kappa, size=J)
    x_obs = rng.binomial(n_ab, true_p)
    obs_rate = x_obs / n_ab

    with pm.Model():
        mu = pm.Beta("mu", 2, 5)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=0.02)
        alpha = pm.Deterministic("alpha", mu * kappa)
        beta = pm.Deterministic("beta", (1 - mu) * kappa)
        p = pm.Beta("p", alpha=alpha, beta=beta, shape=J)
        pm.Binomial("x", n=n_ab, p=p, observed=x_obs)
        idata = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    p_post_mean = idata.posterior["p"].values.mean(axis=(0, 1))
    mu_post = float(idata.posterior["mu"].values.mean())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    lims = (0.0, 0.6)
    ax.plot(lims, lims, "--", color="gray", lw=1, label="y = x(縮小なし)", zorder=1)
    ax.axhline(mu_post, color="black", lw=0.8, ls=":", label=f"全体平均 μ={mu_post:.3f}", zorder=1)

    for orate, pmean, n in zip(obs_rate, p_post_mean, n_ab):
        ax.plot([orate, orate], [orate, pmean], color="gray", lw=0.6, alpha=0.5, zorder=2)

    sizes = 15 + 60 * (n_ab / n_ab.max())
    sc = ax.scatter(obs_rate, p_post_mean, s=sizes, c=n_ab, cmap="viridis",
                     edgecolor="black", linewidth=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("観測打席数 n")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("観測された打率(x/n)")
    ax.set_ylabel("事後平均(部分プーリング後)")
    ax.set_title("階層Beta-Binomialモデルによる部分プーリング\n観測回数が少ない個体ほど全体平均へ強く縮小する")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_binomial_shrinkage.png")
    plt.close(fig)

    shrink = p_post_mean - obs_rate
    small_n_mask = n_ab <= 15
    large_n_mask = n_ab >= 300
    print(f"beta_binomial_shrinkage.png saved (mu_post={mu_post:.3f}, "
          f"mean|shrink| n<=15: {np.abs(shrink[small_n_mask]).mean():.3f}, "
          f"n>=300: {np.abs(shrink[large_n_mask]).mean():.3f})")


def plot_gamma_poisson_overdispersion():
    """overdispersedなカウントデータに対し、Poisson固定分散モデルの予測区間が
    過小評価になる一方、Gamma-Poisson階層モデルは実測の散らばりを正しく捉えることを示す。"""

    rng = np.random.default_rng(9)
    n = 40
    true_mean = 8.0
    true_alpha_conc = 3.0  # 小さいほどoverdispersionが強い
    lam_i = rng.gamma(true_alpha_conc, true_mean / true_alpha_conc, size=n)
    y_obs = rng.poisson(lam_i)

    with pm.Model():
        lam = pm.Gamma("lam", alpha=2.0, beta=0.2)
        pm.Poisson("y", mu=lam, observed=y_obs)
        idata_poisson = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
        ppc_poisson = pm.sample_posterior_predictive(idata_poisson, progressbar=False)

    with pm.Model():
        mu = pm.Gamma("mu", alpha=2.0, beta=0.2)
        alpha_conc = pm.Gamma("alpha_conc", alpha=2.0, beta=0.3)
        beta = pm.Deterministic("beta", alpha_conc / mu)
        lam_i_gp = pm.Gamma("lam_i", alpha=alpha_conc, beta=beta, shape=n)
        pm.Poisson("y", mu=lam_i_gp, observed=y_obs)
        idata_gp = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )
        ppc_gp = pm.sample_posterior_predictive(idata_gp, progressbar=False)

    y_pred_poisson = ppc_poisson.posterior_predictive["y"].values.reshape(-1, n)
    y_pred_gp = ppc_gp.posterior_predictive["y"].values.reshape(-1, n)

    lo_p, hi_p = np.percentile(y_pred_poisson.flatten(), [2.5, 97.5])
    lo_gp, hi_gp = np.percentile(y_pred_gp.flatten(), [2.5, 97.5])

    coverage_poisson = np.mean((y_obs >= lo_p) & (y_obs <= hi_p))
    coverage_gp = np.mean((y_obs >= lo_gp) & (y_obs <= hi_gp))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    bins = np.arange(0, max(y_obs.max(), y_pred_gp.max()) + 2) - 0.5

    for ax, y_pred, lo, hi, cov, color, title in [
        (axes[0], y_pred_poisson, lo_p, hi_p, coverage_poisson, COLOR_OK, "Poisson固定分散モデル"),
        (axes[1], y_pred_gp, lo_gp, hi_gp, coverage_gp, COLOR_ALT, "Gamma-Poisson階層モデル"),
    ]:
        ax.hist(y_pred.flatten(), bins=bins, density=True, color=color, alpha=0.4, label="事後予測分布")
        ax.hist(y_obs, bins=bins, density=True, histtype="step", color="black", lw=1.5, label="実測データ")
        ax.axvspan(lo, hi, color=color, alpha=0.15, label=f"95%予測区間 [{lo:.0f}, {hi:.0f}]")
        ax.set_title(f"{title}\n実測データの区間内カバー率: {cov*100:.0f}%")
        ax.set_xlabel("カウント数")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("密度")
    fig.suptitle("Gamma-Poissonによるoverdispersion補正", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "gamma_poisson_overdispersion.png")
    plt.close(fig)
    print(f"gamma_poisson_overdispersion.png saved "
          f"(obs var/mean={y_obs.var()/y_obs.mean():.2f}, "
          f"Poisson 95%PI=[{lo_p:.0f},{hi_p:.0f}] coverage={coverage_poisson*100:.0f}%, "
          f"Gamma-Poisson 95%PI=[{lo_gp:.0f},{hi_gp:.0f}] coverage={coverage_gp*100:.0f}%)")


def plot_normal_vs_studentt_robustness():
    """外れ値を含む連続データに対し、Normal観測分布の回帰は外れ値に
    回帰直線が引っ張られるが、Student-t観測分布(裾が重い)は外れ値の
    影響を大きく受けずに真の傾きに近い推定を保つことを示す。"""

    rng = np.random.default_rng(6)
    n = 60
    true_slope, true_intercept = 1.2, 0.0
    x = rng.uniform(-3, 3, n)
    y = true_intercept + true_slope * x + rng.normal(0, 0.6, n)

    # 高レバレッジ(xが大きい)側に、一貫して下方向へ外れる外れ値を配置する
    # (ランダムな符号だと線形回帰の傾きへの影響が平均的に打ち消し合うため)
    n_outliers = 5
    outlier_idx = np.argsort(x)[-n_outliers:]
    y[outlier_idx] = (true_intercept + true_slope * x[outlier_idx]) - rng.uniform(9, 12, n_outliers)

    with pm.Model():
        b0 = pm.Normal("b0", 0, 5)
        b1 = pm.Normal("b1", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", mu=b0 + b1 * x, sigma=sigma, observed=y)
        idata_normal = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                                  random_seed=1, progressbar=False,
                                  compute_convergence_checks=False)

    with pm.Model():
        b0 = pm.Normal("b0", 0, 5)
        b1 = pm.Normal("b1", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)  # 自由度(小さいほど裾が重い)
        pm.StudentT("y", mu=b0 + b1 * x, sigma=sigma, nu=nu, observed=y)
        idata_t = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                             random_seed=1, progressbar=False,
                             compute_convergence_checks=False)

    b0_n, b1_n = float(idata_normal.posterior["b0"].mean()), float(idata_normal.posterior["b1"].mean())
    b0_t, b1_t = float(idata_t.posterior["b0"].mean()), float(idata_t.posterior["b1"].mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    is_outlier = np.zeros(n, dtype=bool)
    is_outlier[outlier_idx] = True
    ax.scatter(x[~is_outlier], y[~is_outlier], color="gray", s=25, alpha=0.7, label="通常のデータ")
    ax.scatter(x[is_outlier], y[is_outlier], color=COLOR_DIVERGENT, s=45, marker="x",
               label="外れ値")

    x_grid = np.linspace(-3, 3, 100)
    ax.plot(x_grid, true_intercept + true_slope * x_grid, color="black", lw=1.3, ls="--",
            label=f"真の直線(傾き={true_slope})")
    ax.plot(x_grid, b0_n + b1_n * x_grid, color=COLOR_DIVERGENT, lw=2,
            label=f"Normal観測分布(傾き={b1_n:.3f})")
    ax.plot(x_grid, b0_t + b1_t * x_grid, color=COLOR_OK, lw=2,
            label=f"Student-t観測分布(傾き={b1_t:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"外れ値({n_outliers}/{n}件)に対しNormalは回帰直線が引っ張られるが\n"
                 f"Student-tは真の傾き({true_slope})に近い推定を保つ")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "normal_vs_studentt_robustness.png")
    plt.close(fig)

    print(f"normal_vs_studentt_robustness.png saved "
          f"(真の傾き={true_slope}, Normal推定={b1_n:.3f}, Student-t推定={b1_t:.3f})")


def plot_poisson_equidispersion_fit():
    """等分散(分散=平均)なカウントデータに対しPoissonモデルをPyMCでフィットし、
    事後予測チェック(PPC)が観測分布を正しく捉えることを示す(隣接するGamma-Poisson
    エントリの過分散データに対するPPC過小評価との対比)。"""

    rng = np.random.default_rng(7)
    true_lambda = 8.0
    n = 200
    counts = rng.poisson(true_lambda, n)
    mean_obs, var_obs = counts.mean(), counts.var()

    with pm.Model():
        lam = pm.Gamma("lam", alpha=2, beta=0.2)
        pm.Poisson("y", mu=lam, observed=counts)
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)
        ppc = pm.sample_posterior_predictive(idata, random_seed=2, progressbar=False)

    ppc_draws = ppc.posterior_predictive["y"].values.reshape(-1, n)
    lo, hi = np.percentile(ppc_draws, [2.5, 97.5], axis=0)
    coverage = float(np.mean((counts >= lo) & (counts <= hi)))

    bins = np.arange(0, counts.max() + 3) - 0.5
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(counts, bins=bins, density=True, color=COLOR_ALT, alpha=0.6,
            label="観測データ")
    ax.hist(ppc_draws.flatten(), bins=bins, density=True, histtype="step",
            color=COLOR_OK, lw=2, label="事後予測分布")
    ax.set_xlabel("カウント値")
    ax.set_ylabel("密度")
    ax.set_title(f"Poisson: 分散/平均比={var_obs/mean_obs:.2f}(等分散)のデータに対し\n"
                 f"事後予測95%区間のカバレッジ={coverage*100:.0f}%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "poisson_equidispersion_fit.png")
    plt.close(fig)

    print(f"poisson_equidispersion_fit.png saved "
          f"(mean={mean_obs:.2f}, var={var_obs:.2f}, coverage={coverage:.3f})")


def plot_bernoulli_binomial_equivalence():
    """試行を行単位のBernoulli尤度で扱う場合と、集計済み成功回数のBinomial尤度で
    扱う場合が、数学的に同一の事後分布を与えることを実際にPyMCで確認する。"""

    rng = np.random.default_rng(3)
    n = 200
    true_p = 0.35
    trials = rng.binomial(1, true_p, n)
    k = int(trials.sum())

    with pm.Model():
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("x", p=p, observed=trials)
        idata_bern = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                                random_seed=1, progressbar=False,
                                compute_convergence_checks=False)

    with pm.Model():
        p = pm.Beta("p", 1, 1)
        pm.Binomial("x", n=n, p=p, observed=k)
        idata_binom = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                                 random_seed=2, progressbar=False,
                                 compute_convergence_checks=False)

    p_bern = idata_bern.posterior["p"].values.flatten()
    p_binom = idata_binom.posterior["p"].values.flatten()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(p_bern, bins=50, density=True, color=COLOR_ALT, alpha=0.5,
            label=f"Bernoulli尤度(行単位n={n}): 事後平均={p_bern.mean():.3f}")
    ax.hist(p_binom, bins=50, density=True, color=COLOR_OK, alpha=0.5,
            label=f"Binomial尤度(集計値k={k}/{n}): 事後平均={p_binom.mean():.3f}")
    ax.axvline(true_p, color="black", lw=1.3, ls="--", label=f"真値p={true_p}")
    ax.set_xlabel("p")
    ax.set_ylabel("密度")
    ax.set_title("Bernoulli(行単位)とBinomial(集計値)は数学的に同一の事後分布を与える")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bernoulli_binomial_equivalence.png")
    plt.close(fig)

    print(f"bernoulli_binomial_equivalence.png saved "
          f"(Bernoulli事後平均={p_bern.mean():.4f}, Binomial事後平均={p_binom.mean():.4f})")


def plot_dirichlet_multinomial_bounded_variance():
    """集中度パラメータを0に近づけたとき、Dirichlet-Multinomialの分散は総量Nで
    頭打ちになる一方、Gamma-Poissonの分散は発散することを解析式で比較する。"""

    N = 100
    pi1 = 0.3
    concentrations = np.logspace(-2, 2, 100)

    # Dirichlet-Multinomial: Var(x1) = N*pi1*(1-pi1) * (conc+N)/(conc+1)
    var_dm = N * pi1 * (1 - pi1) * (concentrations + N) / (concentrations + 1)

    # Gamma-Poisson(負の二項): mu固定、Var(x) = mu + mu^2/alpha_conc
    mu = N * pi1
    var_gp = mu + mu ** 2 / concentrations

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(concentrations, var_dm, color=COLOR_OK, lw=2.2,
            label=f"Dirichlet-Multinomial(N={N}固定)")
    ax.plot(concentrations, var_gp, color=COLOR_DIVERGENT, lw=2.2,
            label="Gamma-Poisson(総量Nの制約なし)")
    ax.axhline(N ** 2 * pi1 * (1 - pi1), color=COLOR_OK, lw=1, ls=":",
               label=f"DMの下限(N²π(1-π)={N**2*pi1*(1-pi1):.0f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("集中度パラメータ(対数軸)")
    ax.set_ylabel("カテゴリ1のカウントの分散(対数軸)")
    ax.set_title("集中度→0でDirichlet-Multinomialの分散は頭打ちだが\nGamma-Poissonは発散する")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dirichlet_multinomial_bounded_variance.png")
    plt.close(fig)

    print(f"dirichlet_multinomial_bounded_variance.png saved "
          f"(conc=0.01: DM分散={var_dm[0]:.1f}, GP分散={var_gp[0]:.1f})")


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


def plot_hazard_exponential_vs_weibull():
    """ハザード率が時間とともに増加する合成生存時間データに対し、Exponential
    (ハザード一定)とWeibull(時間依存ハザード)を実際にPyMCでフィットし、
    Kaplan-Meier曲線との整合性を比較する。"""

    rng = np.random.default_rng(11)
    n = 300
    true_k, true_lam = 1.8, 10.0
    t = (rng.weibull(true_k, n) * true_lam)
    cens_time = 15.0
    event = (t <= cens_time).astype(int)
    t_obs = np.minimum(t, cens_time)

    km_t, km_s = _kaplan_meier(t_obs, event)

    with pm.Model():
        lam_rate = pm.Gamma("lam_rate", 2, 2)
        loglik = event * pt.log(lam_rate) - lam_rate * t_obs
        pm.Potential("loglik", loglik.sum())
        idata_exp = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)
    lam_exp = float(idata_exp.posterior["lam_rate"].mean())
    s_exp = np.exp(-lam_exp * km_t)

    with pm.Model():
        k = pm.Gamma("k", 2, 1)
        lam = pm.Gamma("lam", 2, 0.2)
        logh = pt.log(k / lam) + (k - 1) * pt.log(t_obs / lam)
        logS = -((t_obs / lam) ** k)
        loglik = event * logh + logS
        pm.Potential("loglik", loglik.sum())
        idata_weib = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                                random_seed=2, progressbar=False,
                                compute_convergence_checks=False)
    k_est = float(idata_weib.posterior["k"].mean())
    lam_weib = float(idata_weib.posterior["lam"].mean())
    s_weib = np.exp(-((km_t / lam_weib) ** k_est))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.step(km_t, km_s, where="post", color="black", lw=2, label="Kaplan-Meier推定")
    ax.plot(km_t, s_exp, color=COLOR_DIVERGENT, lw=2,
            label=f"Exponentialフィット(λ={lam_exp:.3f})")
    ax.plot(km_t, s_weib, color=COLOR_OK, lw=2,
            label=f"Weibullフィット(k={k_est:.2f}, λ={lam_weib:.2f})")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("生存確率 S(t)")
    ax.set_title(f"真のハザードは時間とともに増加(Weibull k_true={true_k})\n"
                 f"Exponential(ハザード一定)はKM曲線と系統的にズレるが、Weibullは追従する")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hazard_exponential_vs_weibull.png")
    plt.close(fig)

    print(f"hazard_exponential_vs_weibull.png saved "
          f"(Exponential: lambda={lam_exp:.3f}, Weibull: k={k_est:.3f}(真値{true_k}) "
          f"lam={lam_weib:.3f}(真値{true_lam}))")


def plot_piecewise_exponential_recovery():
    """U字型の真のベースラインハザードを持つ合成生存時間データに対し、
    区間ごとに定数ハザードを推定するPiecewise Exponentialモデルを実際に
    PyMCでフィットし、U字型の形状を復元できることを示す。"""

    rng = np.random.default_rng(13)
    n = 500
    breaks = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    true_h = np.array([0.35, 0.15, 0.08, 0.08, 0.15, 0.35])
    durations = np.diff(breaks)
    cumH_at_break = np.concatenate([[0.0], np.cumsum(true_h * durations)])

    E = rng.exponential(1.0, n)
    t = np.empty(n)
    for i in range(n):
        k_idx = min(np.searchsorted(cumH_at_break, E[i], side="right") - 1, len(true_h) - 1)
        t[i] = breaks[k_idx] + (E[i] - cumH_at_break[k_idx]) / true_h[k_idx]

    cens_time = breaks[-1]
    event = (t <= cens_time).astype(int)
    t_obs = np.minimum(t, cens_time)

    n_int = len(true_h)
    exposure = np.zeros((n, n_int))
    d_ik = np.zeros((n, n_int))
    for k_idx in range(n_int):
        lo, hi = breaks[k_idx], breaks[k_idx + 1]
        exposure[:, k_idx] = np.clip(t_obs, lo, hi) - lo
        exposure[:, k_idx] = np.maximum(exposure[:, k_idx], 0)
        d_ik[:, k_idx] = ((t_obs > lo) & (t_obs <= hi) & (event == 1)).astype(float)

    exposure_sum = exposure.sum(axis=0)
    d_sum = d_ik.sum(axis=0)

    with pm.Model():
        h = pm.Gamma("h", alpha=1.5, beta=5, shape=n_int)
        loglik = pt.sum(d_sum * pt.log(h) - h * exposure_sum)
        pm.Potential("loglik", loglik)
        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)
    h_est = idata.posterior["h"].values.reshape(-1, n_int).mean(axis=0)
    h_lo, h_hi = np.percentile(idata.posterior["h"].values.reshape(-1, n_int), [5, 95], axis=0)

    mid = (breaks[:-1] + breaks[1:]) / 2
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.step(breaks, np.append(true_h, true_h[-1]), where="post", color="black", lw=1.5,
            ls="--", label="真のベースラインハザード(U字型)")
    ax.errorbar(mid, h_est, yerr=[h_est - h_lo, h_hi - h_est], fmt="o", color=COLOR_OK,
                capsize=4, ms=6, label="Piecewise Exponentialの区間別推定(90%区間)")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("ハザード率 h(t)")
    ax.set_title("Piecewise Exponentialモデルは6区間のU字型ハザードを復元する")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "piecewise_exponential_recovery.png")
    plt.close(fig)

    print(f"piecewise_exponential_recovery.png saved "
          f"(true_h={np.round(true_h, 3).tolist()}, h_est={np.round(h_est, 3).tolist()})")


def plot_frailty_group_recovery():
    """指数ハザードに乗算的な変量効果(frailty)を持つ合成生存時間データに対し、
    Frailtyモデルを実際にPyMCでフィットし、グループ間の系統差を復元できることを示す。"""

    rng = np.random.default_rng(17)
    n_groups = 5
    n_per_group = 100
    baseline_h = 0.1
    true_z = np.array([0.5, 0.8, 1.0, 1.3, 2.0])
    group = np.repeat(np.arange(n_groups), n_per_group)
    n = len(group)
    lam_i = baseline_h * true_z[group]
    t = rng.exponential(1 / lam_i)
    cens_time = 8.0
    event = (t <= cens_time).astype(int)
    t_obs = np.minimum(t, cens_time)

    empirical_rate = np.array([
        event[group == g].sum() / t_obs[group == g].sum() for g in range(n_groups)
    ])

    with pm.Model():
        lam0 = pm.Gamma("lam0", 2, 20)
        sigma_z = pm.HalfNormal("sigma_z", 0.5)
        log_z_raw = pm.Normal("log_z_raw", 0, 1, shape=n_groups)
        log_z = log_z_raw * sigma_z - pt.mean(log_z_raw * sigma_z)
        z = pm.Deterministic("z", pt.exp(log_z))
        lam_ind = lam0 * z[group]
        loglik = event * pt.log(lam_ind) - lam_ind * t_obs
        pm.Potential("loglik", loglik.sum())
        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)
    ndiv = int(idata.sample_stats["diverging"].sum())
    z_draws = idata.posterior["z"].values.reshape(-1, n_groups)
    z_mean = z_draws.mean(axis=0)
    z_lo, z_hi = np.percentile(z_draws, [5, 95], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(np.arange(n_groups) - 0.15, true_z, width=0.3, color="black",
                alpha=0.4, label="真のz")
    axes[0].errorbar(np.arange(n_groups) + 0.15, z_mean,
                      yerr=[z_mean - z_lo, z_hi - z_mean], fmt="o", color=COLOR_OK,
                      capsize=4, ms=6, label="Frailtyモデルの推定z(90%区間)")
    axes[0].set_xticks(np.arange(n_groups))
    axes[0].set_xlabel("グループ")
    axes[0].set_ylabel("frailty z")
    axes[0].set_title(f"グループ別frailtyの復元(divergence={ndiv})")
    axes[0].legend(fontsize=8)

    axes[1].bar(np.arange(n_groups), empirical_rate, color=COLOR_ALT, alpha=0.7,
                label="経験的ハザード率(件数/曝露時間)")
    axes[1].axhline(baseline_h, color="black", lw=1.3, ls="--",
                     label=f"frailty無視の単一推定(baseline={baseline_h})")
    axes[1].set_xticks(np.arange(n_groups))
    axes[1].set_xlabel("グループ")
    axes[1].set_ylabel("ハザード率")
    axes[1].set_title("frailtyを無視すると単一の平均値でグループ差を捉え損ねる")
    axes[1].legend(fontsize=8)

    fig.suptitle("Frailty(変量効果): 共変量だけでは説明できないグループ間の系統差を復元", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "frailty_group_recovery.png")
    plt.close(fig)

    print(f"frailty_group_recovery.png saved (divergence={ndiv}, true_z={true_z.tolist()}, "
          f"z_est={np.round(z_mean, 3).tolist()})")


def _simulate_hawkes(mu, kappa, beta, t_max, rng):
    events = []
    t = 0.0
    while t < t_max:
        lam_bar = mu + sum(kappa * np.exp(-beta * (t - s)) for s in events)
        lam_bar = max(lam_bar, mu)
        w = rng.exponential(1 / lam_bar)
        t += w
        if t >= t_max:
            break
        lam_t = mu + sum(kappa * np.exp(-beta * (t - s)) for s in events)
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)
    return np.array(events)


def plot_hawkes_process_recovery():
    """自己励起点過程(Hawkes過程)から合成イベント系列を生成し、対数尤度
    (Σlog λ(t_i) - ∫λ(t)dt)をpm.Potentialで直接実装してPyMCでフィットし、
    強度関数のパラメータを復元できることを示す。"""

    rng = np.random.default_rng(19)
    mu_true, kappa_true, beta_true = 0.5, 0.8, 1.2
    t_max = 60.0
    events = _simulate_hawkes(mu_true, kappa_true, beta_true, t_max, rng)
    n_events = len(events)

    ev = pt.as_tensor(events)
    with pm.Model():
        mu = pm.Gamma("mu", 2, 4)
        kappa = pm.Gamma("kappa", 2, 2)
        beta = pm.Gamma("beta", 2, 1)
        diff = ev[:, None] - ev[None, :]
        mask = diff > 0
        excitation = pt.sum(pt.switch(mask, kappa * pt.exp(-beta * diff), 0.0), axis=1)
        lam_at_events = mu + excitation
        log_lam_term = pt.sum(pt.log(lam_at_events))
        integral_term = mu * t_max + (kappa / beta) * pt.sum(1 - pt.exp(-beta * (t_max - ev)))
        pm.Potential("loglik", log_lam_term - integral_term)
        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.95,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)
    ndiv = int(idata.sample_stats["diverging"].sum())
    mu_est = float(idata.posterior["mu"].mean())
    kappa_est = float(idata.posterior["kappa"].mean())
    beta_est = float(idata.posterior["beta"].mean())

    t_grid = np.linspace(0, 25, 1000)

    def intensity(t_grid, mu_v, kappa_v, beta_v, events):
        lam = np.full_like(t_grid, mu_v)
        for s in events[events < t_grid[-1]]:
            lam += np.where(t_grid > s, kappa_v * np.exp(-beta_v * (t_grid - s)), 0.0)
        return lam

    lam_true = intensity(t_grid, mu_true, kappa_true, beta_true, events)
    lam_est = intensity(t_grid, mu_est, kappa_est, beta_est, events)
    events_shown = events[events < 25]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_grid, lam_true, color="black", lw=1.5, ls="--", label="真の強度関数λ(t)")
    ax.plot(t_grid, lam_est, color=COLOR_OK, lw=2, label="推定した強度関数λ(t)")
    ax.plot(events_shown, np.zeros_like(events_shown), "|", color=COLOR_DIVERGENT,
            ms=14, mew=1.5, label="イベント発生時刻")
    ax.set_xlabel("時間 t")
    ax.set_ylabel("強度 λ(t)")
    ax.set_title(f"Hawkes過程: μ={mu_est:.2f}(真値{mu_true}), κ={kappa_est:.2f}(真値{kappa_true}), "
                 f"β={beta_est:.2f}(真値{beta_true})\n(全{n_events}イベント中t<25の範囲、divergence={ndiv})")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hawkes_process_recovery.png")
    plt.close(fig)

    print(f"hawkes_process_recovery.png saved (n_events={n_events}, divergence={ndiv}, "
          f"mu_est={mu_est:.3f}, kappa_est={kappa_est:.3f}, beta_est={beta_est:.3f})")


if __name__ == "__main__":
    plot_beta_binomial_shrinkage()
    plot_gamma_poisson_overdispersion()
    plot_poisson_equidispersion_fit()
    plot_bernoulli_binomial_equivalence()
    plot_dirichlet_multinomial_bounded_variance()
    plot_hazard_exponential_vs_weibull()
    plot_piecewise_exponential_recovery()
    plot_frailty_group_recovery()
    plot_hawkes_process_recovery()
    plot_normal_vs_studentt_robustness()
