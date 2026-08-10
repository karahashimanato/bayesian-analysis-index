"""
techniques/reparameterization.md に埋め込む可視化画像を生成するスクリプト。

PyMC で実際にサンプリングし、再パラメータ化(reparameterization)が
非識別性・多峰性をどう解消するかを before/after で描画する:
  1. 三角関数の極形式(A, φ) vs 直交形式(β1, β2)
  2. Ridge型非識別性: (κ, β) vs 比M=κ/β

実行方法:
    source .venv/bin/activate
    python scripts/generate_reparameterization_plots.py

出力先: assets/reparameterization/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from scipy import stats

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "reparameterization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_ratio_constant_epsilon_reparam():
    """R0=beta/gammaのように比が本質的な量の場合、betaとgammaを独立に
    事前分布からサンプルするとR0のprior predictiveが暴走しうるが、
    R0=1+epsilonのように比自体を「定数+小さな揺らぎ」で再パラメータ化すると
    直接コントロールできることを実際にサンプリングして示す。"""

    rng = np.random.default_rng(1)
    n = 50000

    # 独立パラメータ化: betaは通常のスケール、gammaは0に近づきうる
    beta_indep = stats.halfnorm.rvs(scale=2.0, size=n, random_state=rng)
    gamma_indep = stats.halfnorm.rvs(scale=1.0, size=n, random_state=rng)
    R0_indep = beta_indep / np.clip(gamma_indep, 1e-6, None)

    # 再パラメータ化: R0 = 1 + epsilon, epsilon ~ Gamma(2, scale=0.3)
    epsilon = stats.gamma.rvs(a=2.0, scale=0.3, size=n, random_state=rng)
    R0_reparam = 1.0 + epsilon

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bins = np.logspace(-1, 7, 60)
    axes[0].hist(R0_indep, bins=bins, color=COLOR_DIVERGENT, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("R0 = beta/gamma(対数軸)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"独立パラメータ化 beta~HalfNormal(2), gamma~HalfNormal(1)\n"
                       f"(中央値={np.median(R0_indep):.2f}, 99%ile={np.percentile(R0_indep,99):,.0f}, "
                       f"最大={R0_indep.max():,.0f})")

    axes[1].hist(R0_reparam, bins=60, color=COLOR_OK, alpha=0.7)
    axes[1].set_xlabel("R0 = 1 + epsilon")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"再パラメータ化 R0=1+epsilon, epsilon~Gamma(2,0.3)\n"
                       f"(中央値={np.median(R0_reparam):.2f}, 99%ile={np.percentile(R0_reparam,99):.2f}, "
                       f"最大={R0_reparam.max():.2f})")

    fig.suptitle("比R0=beta/gammaを独立パラメータ化すると暴走するが、\n比自体を「定数+ε」で再パラメータ化すると直接コントロールできる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "ratio_constant_epsilon_reparam.png")
    plt.close(fig)

    print(f"ratio_constant_epsilon_reparam.png saved "
          f"(独立: 中央値={np.median(R0_indep):.2f} 最大={R0_indep.max():.0f}, "
          f"再パラメータ化: 中央値={np.median(R0_reparam):.2f} 最大={R0_reparam.max():.2f})")


def plot_derived_vs_independent_i0():
    """I0(初期感染者数)を独立にサンプルすると、モデルが暗に予測する
    初期時点の新規感染数(gamma*I0)が実際の観測値と数値的に矛盾する
    組み合わせが多数を占めうるが、I0=incidence_obs[0]/gammaのように
    既知の観測値と他パラメータから導出すれば、この矛盾が構造的に
    排除されることを実際にサンプリングして示す。"""

    rng = np.random.default_rng(5)
    n = 30000
    observed_first_incidence = 12.0  # 最初の時点で実際に観測された新規感染数

    gamma = stats.halfnorm.rvs(scale=0.5, size=n, random_state=rng)

    # 独立パラメータ化: I0を観測値と無関係にサンプル
    I0_indep = stats.halfnorm.rvs(scale=50.0, size=n, random_state=rng)
    predicted_incidence_indep = gamma * I0_indep
    rel_error_indep = np.abs(predicted_incidence_indep - observed_first_incidence) / observed_first_incidence

    # 導出パラメータ化: I0 = observed_first_incidence / gamma
    I0_derived = observed_first_incidence / np.clip(gamma, 1e-6, None)
    predicted_incidence_derived = gamma * I0_derived  # 構造的に常にobserved_first_incidenceと一致

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    frac_badly_off = float(np.mean(rel_error_indep > 0.5))
    axes[0].hist(np.clip(predicted_incidence_indep, 0, 100), bins=60, color=COLOR_DIVERGENT, alpha=0.7)
    axes[0].axvline(observed_first_incidence, color="black", ls="--", lw=1.5,
                     label=f"実際に観測された初期新規感染数={observed_first_incidence:.0f}")
    axes[0].set_xlabel("gamma×I0(暗に予測される初期新規感染数、100で打ち切り)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"I0を独立にサンプル\n(相対誤差>50%の割合={frac_badly_off:.1%})")
    axes[0].legend(fontsize=8.5)

    axes[1].bar([observed_first_incidence], [n], width=1.5, color=COLOR_OK, alpha=0.8,
                label=f"gamma×I0_derivedは全{n:,}サンプルとも\n観測値={observed_first_incidence:.0f}と一致")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 5000)
    axes[1].set_xlabel("gamma×I0_derived(100で打ち切り、左と同じ軸)")
    axes[1].set_ylabel("count")
    axes[1].set_title("I0を observed/gamma として導出\n(矛盾する組み合わせが構造的に存在しない)")
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle("I0を独立サンプルすると観測値と矛盾する組み合わせが多数を占めるが、\n他量から導出すれば矛盾は構造的に排除される", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "derived_vs_independent_i0.png")
    plt.close(fig)

    print(f"derived_vs_independent_i0.png saved "
          f"(独立サンプル: 相対誤差>50%の割合={frac_badly_off:.1%}, "
          f"予測値の中央値={np.median(predicted_incidence_indep):.1f} vs 観測値={observed_first_incidence:.0f}, "
          f"導出パラメータ化は常に一致)")


def plot_complexity_creates_nonidentifiability():
    """シンプルなモデル(切片+単一の傾き)を、同じ役割を持つ係数を2つに
    分けた冗長な形(beta1, beta2が両方xにかかる)へ拡張すると、
    beta1・beta2は個別には識別されず事前分布のスケールに敏感になる
    (感度分析で検出できる)が、識別可能な合成量beta1+beta2は
    データが支持する値のまま安定していることを実際にサンプリングして示す。"""

    rng = np.random.default_rng(9)
    n = 60
    x = rng.uniform(-2, 2, n)
    true_slope = 1.2
    y = 0.5 + true_slope * x + rng.normal(0, 0.3, n)

    def fit_redundant(beta1_prior_sigma, seed):
        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 2)
            beta1 = pm.Normal("beta1", 0, beta1_prior_sigma)
            beta2 = pm.Normal("beta2", 0, 1.0)
            pm.Deterministic("beta_sum", beta1 + beta2)
            pm.Normal("y", beta0 + (beta1 + beta2) * x, 0.3, observed=y)
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        return idata

    idata_tight = fit_redundant(0.1, 1)   # beta1の事前分布を狭く
    idata_wide = fit_redundant(3.0, 1)    # beta1の事前分布を広く

    b1_tight = idata_tight.posterior["beta1"].values.flatten()
    b1_wide = idata_wide.posterior["beta1"].values.flatten()
    sum_tight = idata_tight.posterior["beta_sum"].values.flatten()
    sum_wide = idata_wide.posterior["beta_sum"].values.flatten()

    with pm.Model():
        beta0_s = pm.Normal("beta0", 0, 2)
        beta1_s = pm.Normal("beta1", 0, 1.0)
        pm.Normal("y", beta0_s + beta1_s * x, 0.3, observed=y)
        idata_simple = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                  random_seed=1, progressbar=False,
                                  compute_convergence_checks=False)
    slope_simple = idata_simple.posterior["beta1"].values.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(b1_tight, bins=50, density=True, color=COLOR_OK, alpha=0.55,
                 label=f"beta1の事前分布を狭く(σ=0.1)\n平均={b1_tight.mean():.2f}")
    axes[0].hist(b1_wide, bins=50, density=True, color=COLOR_DIVERGENT, alpha=0.55,
                 label=f"beta1の事前分布を広く(σ=3.0)\n平均={b1_wide.mean():.2f}")
    axes[0].set_xlabel("beta1(個別には識別されない冗長パラメータ)")
    axes[0].set_ylabel("density")
    axes[0].set_title("beta1単体は事前分布のスケールを\n変えるだけで posterior が大きく動く")
    axes[0].legend(fontsize=8)

    axes[1].hist(sum_tight, bins=50, density=True, color=COLOR_OK, alpha=0.55,
                 label=f"beta1狭い設定でのbeta1+beta2\n平均={sum_tight.mean():.2f}")
    axes[1].hist(sum_wide, bins=50, density=True, color=COLOR_DIVERGENT, alpha=0.55,
                 label=f"beta1広い設定でのbeta1+beta2\n平均={sum_wide.mean():.2f}")
    axes[1].axvline(true_slope, color="black", ls="--", lw=1.5, label=f"真の傾き={true_slope}")
    axes[1].axvline(slope_simple.mean(), color=COLOR_ALT, ls=":", lw=1.5,
                     label=f"単純モデルの傾き推定={slope_simple.mean():.2f}")
    axes[1].set_xlabel("識別可能な合成量 beta1+beta2")
    axes[1].set_ylabel("density")
    axes[1].set_title("合成量は事前分布のスケールに\nほぼ依らず安定する(データが支持する値)")
    axes[1].legend(fontsize=7.5)

    fig.suptitle("モデルを冗長に拡張すると新しい非識別性が生まれるが、\n感度分析(事前分布のスケールを変える)で検出できる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "complexity_creates_nonidentifiability.png")
    plt.close(fig)

    print(f"complexity_creates_nonidentifiability.png saved "
          f"(真の傾き={true_slope}, 単純モデル推定={slope_simple.mean():.3f}, "
          f"beta1(狭い事前)平均={b1_tight.mean():.3f}, beta1(広い事前)平均={b1_wide.mean():.3f}, "
          f"beta_sum(狭い)平均={sum_tight.mean():.3f}, beta_sum(広い)平均={sum_wide.mean():.3f})")


def plot_necessary_vs_wished_constraint():
    """分岐比M(倍率)を「安定していてほしい」という願望からM<1に制約すると、
    有限観測期間では真にM>1でも問題なくデータを説明できる場合に、その
    願望的制約がむしろ推定を歪めることを実際にサンプリングして示す。
    M<1はAR(1)の定常性のように数学的に必須な条件ではなく、単なる期待に
    過ぎない場合がある、という区別を具体例で示す。"""

    rng = np.random.default_rng(23)
    T = 15
    M_true = 1.15  # 有限期間では問題なく観測できる、緩やかに増加する系列
    x0 = 5.0
    sigma_obs = 0.5

    x = np.empty(T)
    x_prev = x0
    for t in range(T):
        x_prev = M_true * x_prev + rng.normal(0, sigma_obs)
        x[t] = x_prev

    def fit(constrained, seed):
        with pm.Model():
            if constrained:
                # 願望的制約: M<1(安定していてほしいという期待)。
                # Betaの台は(0,1)なので、この制約はモデルの構造そのものに組み込まれる
                M = pm.Deterministic("M_eff", pm.Beta("M_raw", 2.0, 2.0))
            else:
                # 制約なし: (0, inf)の開いた事前分布(必須でなければデータに語らせる)
                M = pm.Gamma("M_eff", alpha=3.0, beta=2.0)

            mu = [x0]
            x_prev_sym = x0
            preds = []
            for t in range(T):
                x_prev_sym = M * x_prev_sym
                preds.append(x_prev_sym)
            import pytensor.tensor as ptt
            pred_stack = ptt.stack(preds)
            pm.Normal("obs", pred_stack, sigma_obs, observed=x)
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        return idata

    idata_constrained = fit(True, 1)
    idata_open = fit(False, 1)

    M_constrained = idata_constrained.posterior["M_eff"].values.flatten()
    M_open = idata_open.posterior["M_eff"].values.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(np.arange(1, T + 1), x, "o-", color="black", label="観測データ(真のM=1.15)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x_t")
    axes[0].set_title("有限期間では緩やかに増加するだけで\n観測上は何の問題もない系列")
    axes[0].legend(fontsize=9)

    axes[1].hist(M_constrained, bins=50, density=True, color=COLOR_DIVERGENT, alpha=0.55,
                 label=f"願望的制約 M<1(Betaベース)\n平均={M_constrained.mean():.3f}")
    axes[1].hist(M_open, bins=50, density=True, color=COLOR_OK, alpha=0.55,
                 label=f"制約なし M~Gamma(0,∞)\n平均={M_open.mean():.3f}")
    axes[1].axvline(M_true, color="black", ls="--", lw=1.5, label=f"真の値={M_true}")
    axes[1].set_xlabel("M(倍率)")
    axes[1].set_ylabel("density")
    axes[1].set_title("M<1という願望的制約を課すと\n真の値(M>1)を推定できない")
    axes[1].legend(fontsize=8)

    fig.suptitle("M<1は数学的に必須な制約ではなく単なる期待である場合、\n制約を課すとむしろ推定を歪める", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "necessary_vs_wished_constraint.png")
    plt.close(fig)

    print(f"necessary_vs_wished_constraint.png saved "
          f"(真のM={M_true}, 願望的制約(M<1)平均={M_constrained.mean():.3f}, "
          f"制約なし平均={M_open.mean():.3f})")


def _build_grid_adjacency(side):
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


def plot_bym_theta_phi_ess_bym2_fix():
    """BYMモデルで非構造項thetaの分散sigma_thetaと空間構造項phiの分散
    sigma_phiを別々にモデル化すると、「観測されたばらつきをどちらが
    説明するか」という冗長な自由度のせいでsigma_thetaのESSが著しく
    低くなることを実際にPyMCでフィットして示す。BYM2の(sigma,rho)への
    再パラメータ化でこのESSの低下が解消することも合わせて示す。"""

    rng = np.random.default_rng(3)
    side = 6
    N = side * side
    W = _build_grid_adjacency(side)
    D = W.sum(axis=1)
    Q = np.diag(D) - W
    Q_pert = Q + np.eye(N) * 1e-6
    scale = float(np.exp(np.mean(np.log(np.diag(np.linalg.inv(Q_pert))))))

    rr, cc = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    phi_true = 0.15 * (rr.flatten() + cc.flatten()) - 0.15 * (side - 1)
    phi_true -= phi_true.mean()
    theta_true = rng.normal(0, 0.3, N)
    E = rng.uniform(50, 150, N)
    counts = rng.poisson(E * np.exp(phi_true + theta_true))

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma_theta = pm.HalfNormal("sigma_theta", 1)
        sigma_phi = pm.HalfNormal("sigma_phi", 1)
        theta = pm.Normal("theta", 0, sigma_theta, shape=N)
        phi = pm.ICAR("phi", W=W, sigma=sigma_phi)
        pm.Poisson("y", mu=E * pm.math.exp(beta0 + theta + phi), observed=counts)
        idata_bym = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=1, progressbar=False,
                               compute_convergence_checks=False)

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 2)
        sigma = pm.HalfNormal("sigma", 1)
        rho = pm.Beta("rho", 2, 2)
        theta_star = pm.Normal("theta_star", 0, 1, shape=N)
        phi_star = pm.ICAR("phi_star", W=W, sigma=1)
        combined = sigma * (pt.sqrt(1 - rho) * theta_star + pt.sqrt(rho / scale) * phi_star)
        pm.Poisson("y", mu=E * pt.exp(beta0 + combined), observed=counts)
        idata_bym2 = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                                random_seed=2, progressbar=False,
                                compute_convergence_checks=False)

    sigma_theta_v = idata_bym.posterior["sigma_theta"].values.flatten()
    sigma_phi_v = idata_bym.posterior["sigma_phi"].values.flatten()
    corr_bym = float(np.corrcoef(sigma_theta_v, sigma_phi_v)[0, 1])
    ess_sigma_theta = float(az.ess(idata_bym, var_names=["sigma_theta"]).sigma_theta.values)
    ess_sigma_phi = float(az.ess(idata_bym, var_names=["sigma_phi"]).sigma_phi.values)

    sigma_v = idata_bym2.posterior["sigma"].values.flatten()
    rho_v = idata_bym2.posterior["rho"].values.flatten()
    ess_sigma = float(az.ess(idata_bym2, var_names=["sigma"]).sigma.values)
    ess_rho = float(az.ess(idata_bym2, var_names=["rho"]).rho.values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].scatter(sigma_theta_v, sigma_phi_v, s=4, alpha=0.2, color=COLOR_DIVERGENT)
    axes[0].set_xlabel("sigma_theta(非構造項の分散)")
    axes[0].set_ylabel("sigma_phi(空間構造項の分散)")
    axes[0].set_title(f"BYM: sigma_theta・sigma_phiの事後ペアプロット\n相関={corr_bym:.2f}"
                       f"(どちらが「ばらつき」を\n説明するかを奪い合う冗長な自由度)")

    labels = ["sigma_theta\n(BYM)", "sigma_phi\n(BYM)", "sigma\n(BYM2)", "rho\n(BYM2)"]
    ess_vals = [ess_sigma_theta, ess_sigma_phi, ess_sigma, ess_rho]
    colors = [COLOR_DIVERGENT, COLOR_DIVERGENT, COLOR_OK, COLOR_OK]
    axes[1].bar(labels, ess_vals, color=colors)
    for i, v in enumerate(ess_vals):
        axes[1].annotate(f"{v:.0f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_ylabel("ESS(bulk)")
    axes[1].set_title("BYMのsigma_thetaはESSが著しく低いが、\nBYM2のsigma・rhoはどちらも改善する")

    fig.suptitle("BYMの分散パラメータ(sigma_theta, sigma_phi)は冗長な自由度でESSが低下するが、\nBYM2の(sigma,rho)再パラメータ化で解消する", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT_DIR / "bym_theta_phi_ess_bym2_fix.png")
    plt.close(fig)

    print(f"bym_theta_phi_ess_bym2_fix.png saved "
          f"(sigma_theta-sigma_phi相関={corr_bym:.3f}, "
          f"ESS sigma_theta={ess_sigma_theta:.0f}, ESS sigma_phi={ess_sigma_phi:.0f}, "
          f"ESS sigma(BYM2)={ess_sigma:.0f}, ESS rho(BYM2)={ess_rho:.0f})")


def plot_rw1_centering_beta0_confound():
    """RW1(GaussianRandomWalk)の絶対水準が切片beta0と交絡し、両者の事後分布が
    強く相関することを実際にPyMCでフィットして示す。RW1の値を線形予測子に
    組み込む前に中心化する(平均を引く)と、この交絡とESSの低下が
    解消することも合わせて示す。"""

    rng = np.random.default_rng(11)
    T = 40
    sigma_rw = 0.15
    rw_true = np.cumsum(rng.normal(0, sigma_rw, T))
    beta0_true = 2.0
    y = beta0_true + rw_true + rng.normal(0, 0.3, T)

    def fit(centered, seed):
        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 5)
            sigma = pm.HalfNormal("sigma", 1)
            rw_raw = pm.GaussianRandomWalk("rw_raw", sigma=sigma,
                                            init_dist=pm.Normal.dist(0, 1), steps=T - 1)
            if centered:
                contribution = pm.Deterministic("contribution", rw_raw - pt.mean(rw_raw))
            else:
                contribution = rw_raw
            pm.Normal("y", beta0 + contribution, 0.3, observed=y)
            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        return idata

    idata_unc = fit(False, 1)
    idata_c = fit(True, 2)

    beta0_unc = idata_unc.posterior["beta0"].values.flatten()
    mean_rw_unc = idata_unc.posterior["rw_raw"].values.reshape(-1, T).mean(axis=1)
    corr_unc = float(np.corrcoef(beta0_unc, mean_rw_unc)[0, 1])
    ess_unc = float(az.ess(idata_unc, var_names=["beta0"]).beta0.values)

    beta0_c = idata_c.posterior["beta0"].values.flatten()
    mean_rw_c = idata_c.posterior["rw_raw"].values.reshape(-1, T).mean(axis=1)
    corr_c = float(np.corrcoef(beta0_c, mean_rw_c)[0, 1])
    ess_c = float(az.ess(idata_c, var_names=["beta0"]).beta0.values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].scatter(mean_rw_unc, beta0_unc, s=4, alpha=0.2, color=COLOR_DIVERGENT,
                     label=f"中心化なし(相関={corr_unc:.3f})")
    axes[0].scatter(mean_rw_c, beta0_c, s=4, alpha=0.2, color=COLOR_OK,
                     label=f"事前に中心化(相関={corr_c:.3f})")
    axes[0].set_xlabel("mean(rw_raw)(RW1の平均水準)")
    axes[0].set_ylabel("beta0")
    axes[0].set_title("中心化なしではbeta0とRW1の水準が\nほぼ完全に交絡する")
    axes[0].legend(fontsize=8.5)

    labels = ["beta0\n(中心化なし)", "beta0\n(事前に中心化)"]
    ess_vals = [ess_unc, ess_c]
    axes[1].bar(labels, ess_vals, color=[COLOR_DIVERGENT, COLOR_OK])
    for i, v in enumerate(ess_vals):
        axes[1].annotate(f"{v:.0f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    axes[1].set_ylabel("ESS(bulk)")
    axes[1].set_title("中心化によりbeta0のESSが改善する")

    fig.suptitle("RW1の絶対水準はbeta0と交絡しうるが、\n組み込む前に中心化すると解消する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "rw1_centering_beta0_confound.png")
    plt.close(fig)

    print(f"rw1_centering_beta0_confound.png saved "
          f"(中心化なし: 相関={corr_unc:.3f} ESS={ess_unc:.0f}, "
          f"事前に中心化: 相関={corr_c:.3f} ESS={ess_c:.0f})")


def plot_trig_reparameterization():
    """極形式(A, phi)は位相の周期性ゆえに事前分布のレンジ次第で
    見せかけの多峰性を生むが、直交形式(beta1, beta2)は単峰であることを示す。"""

    rng = np.random.default_rng(1)
    omega = 2 * np.pi / 7.0  # 周期7、既知として外生的に固定(periodogram相当)
    t = np.linspace(0, 40, 200)
    true_A, true_phi = 2.0, 0.6
    y = true_A * np.sin(omega * t + true_phi) + rng.normal(0, 0.3, size=t.size)

    # 極形式: phiの事前分布が複数周期分にまたがるため、周期のズレ分だけ
    # 「見た目上の」複数モードが生まれる(尤度自体は単峰)
    mode_starts = [true_phi - 2 * np.pi, true_phi, true_phi + 2 * np.pi, true_phi]
    with pm.Model():
        A = pm.HalfNormal("A", 5.0)
        phi = pm.Uniform("phi", -2 * np.pi, 4 * np.pi)
        mu = A * pm.math.sin(omega * t + phi)
        pm.Normal("obs", mu, 0.3, observed=y)
        idata_polar = pm.sample(
            1500, tune=1000, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, initvals=[{"phi": s, "A": true_A} for s in mode_starts],
            compute_convergence_checks=False,
        )

    # 直交形式: beta1 = A*cos(phi), beta2 = A*sin(phi) は線形回帰係数そのもの
    with pm.Model():
        beta1 = pm.Normal("beta1", 0.0, 5.0)
        beta2 = pm.Normal("beta2", 0.0, 5.0)
        mu = beta1 * np.sin(omega * t) + beta2 * np.cos(omega * t)
        pm.Normal("obs", mu, 0.3, observed=y)
        idata_cart = pm.sample(
            1500, tune=1000, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    import arviz as az
    rhat_phi = float(az.rhat(idata_polar, var_names=["phi"])["phi"].values)
    rhat_b1 = float(az.rhat(idata_cart, var_names=["beta1"])["beta1"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    n_chains = idata_polar.posterior.sizes["chain"]
    for c in range(n_chains):
        axes[0].scatter(
            idata_polar.posterior["phi"].values[c], idata_polar.posterior["A"].values[c],
            s=4, alpha=0.3, color=COLOR_CHAIN[c % len(COLOR_CHAIN)], label=f"chain {c}",
        )
    axes[0].set_xlabel("φ (phase)")
    axes[0].set_ylabel("A (amplitude)")
    axes[0].set_title(f"極形式 A·sin(ωt+φ)\nr_hat[φ]={rhat_phi:.2f}(周期ズレ分だけ見せかけの多峰性)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=3)

    for c in range(n_chains):
        axes[1].scatter(
            idata_cart.posterior["beta1"].values[c], idata_cart.posterior["beta2"].values[c],
            s=4, alpha=0.3, color=COLOR_CHAIN[c % len(COLOR_CHAIN)],
        )
    axes[1].set_xlabel("β1 (= A·cosφ)")
    axes[1].set_ylabel("β2 (= A·sinφ)")
    axes[1].set_title(f"直交形式 β1·sin(ωt)+β2·cos(ωt)\nr_hat[β1]={rhat_b1:.2f}(単峰)")

    fig.suptitle("三角関数パラメータの再パラメータ化: 極形式(左) vs 直交形式(右)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "trig_reparameterization.png")
    plt.close(fig)
    print(f"trig_reparameterization.png saved (r_hat: polar phi={rhat_phi:.2f}, cartesian beta1={rhat_b1:.2f})")


def plot_ridge_ratio_reparameterization():
    """比 M=kappa/beta が本質的な意味を持つ場合、(kappa, beta)を独立に
    サンプルするとridge状の非識別性でdivergenceが起きるが、Mを直接
    パラメータ化するとdivergenceが解消することを示す。"""

    true_M = 2.0

    # 元のパラメータ化: kappa, beta を独立に事前分布からサンプル
    # -> 尤度は比 kappa/beta にしか制約されないため、原点を通るray状のridgeができる
    # (target_accept=0.6はPyMCのデフォルト0.8より弱く、幾何学的な病理を意図的に露出させる設定)
    with pm.Model():
        kappa = pm.Gamma("kappa", alpha=1.0, beta=0.3)
        beta = pm.Gamma("beta", alpha=1.0, beta=0.3)
        pm.Normal("obs", kappa / beta, 0.01, observed=np.array([true_M]))
        idata_raw = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.6, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    # 再パラメータ化: 比M自体を直接サンプルし、betaはMと無関係な自由パラメータとして残す
    with pm.Model():
        M = pm.Gamma("M", alpha=4.0, beta=2.0)  # 平均2付近
        beta_free = pm.Gamma("beta_free", alpha=2.0, beta=0.3)
        pm.Deterministic("kappa_derived", M * beta_free)
        pm.Normal("obs", M, 0.01, observed=np.array([true_M]))
        idata_reparam = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.6, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    div_raw = idata_raw.sample_stats["diverging"].values.flatten()
    div_reparam = idata_reparam.sample_stats["diverging"].values.flatten()
    n_div_raw, n_div_reparam = int(div_raw.sum()), int(div_reparam.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    k = idata_raw.posterior["kappa"].values.flatten()
    b = idata_raw.posterior["beta"].values.flatten()
    axes[0].scatter(b[~div_raw], k[~div_raw], s=4, alpha=0.25, color=COLOR_OK)
    axes[0].scatter(b[div_raw], k[div_raw], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[0].set_xlabel("β")
    axes[0].set_ylabel("κ")
    axes[0].set_title(f"元のパラメータ化 (κ, β)\nκ/β=M にしか制約されずray状のridge\ndivergence: {n_div_raw}/{len(div_raw)}")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    m = idata_reparam.posterior["M"].values.flatten()
    bf = idata_reparam.posterior["beta_free"].values.flatten()
    axes[1].scatter(bf[~div_reparam], m[~div_reparam], s=4, alpha=0.25, color=COLOR_ALT)
    axes[1].scatter(bf[div_reparam], m[div_reparam], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[1].set_xlabel("β_free(自由パラメータ)")
    axes[1].set_ylabel("M = κ/β(直接推定)")
    axes[1].set_title(f"再パラメータ化 (M, β_free)\nMを直接観測するのでridgeが解消\ndivergence: {n_div_reparam}/{len(div_reparam)}")
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Ridge型非識別性: 比パラメータ化による解消", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "ridge_ratio_reparameterization.png")
    plt.close(fig)
    print(f"ridge_ratio_reparameterization.png saved (divergence: {n_div_raw}/{len(div_raw)} -> {n_div_reparam}/{len(div_reparam)})")


def plot_noncentered_ctr_funnel():
    """腕ごとのCTRを推定する階層ロジスティック回帰で、中心化パラメータ化が
    funnel状のdivergenceを起こし、非中心化パラメータ化で解消することを示す。"""

    rng = np.random.default_rng(3)
    n_arms = np.array([20, 20, 18, 15, 15, 10, 10, 8, 8, 5])  # 試行回数が少ない腕を含む
    true_mu_logit = -2.4  # 全体平均CTR ≈ 8.3%
    true_sigma_arm = 0.15
    J = len(n_arms)
    theta_true = true_mu_logit + true_sigma_arm * rng.normal(size=J)
    p_true = 1 / (1 + np.exp(-theta_true))
    y_obs = rng.binomial(n_arms, p_true)

    with pm.Model():
        mu_logit = pm.Normal("mu_logit", -2.0, 1.0)
        sigma_arm = pm.HalfNormal("sigma_arm", 1.0)
        theta_logit = pm.Normal("theta_logit", mu_logit, sigma_arm, shape=J)
        p = pm.Deterministic("p", pm.math.invlogit(theta_logit))
        pm.Binomial("y", n=n_arms, p=p, observed=y_obs)
        idata_centered = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.8, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    with pm.Model():
        mu_logit = pm.Normal("mu_logit", -2.0, 1.0)
        sigma_arm = pm.HalfNormal("sigma_arm", 1.0)
        offset_raw = pm.Normal("offset_raw", 0.0, 1.0, shape=J)
        theta_logit = pm.Deterministic("theta_logit", mu_logit + sigma_arm * offset_raw)
        p = pm.Deterministic("p", pm.math.invlogit(theta_logit))
        pm.Binomial("y", n=n_arms, p=p, observed=y_obs)
        idata_noncentered = pm.sample(
            2000, tune=1500, chains=4, target_accept=0.8, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    div_c = idata_centered.sample_stats["diverging"].values.flatten()
    div_n = idata_noncentered.sample_stats["diverging"].values.flatten()
    n_div_c, n_div_n = int(div_c.sum()), int(div_n.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    sigma_c = idata_centered.posterior["sigma_arm"].values.flatten()
    theta0_c = idata_centered.posterior["theta_logit"].values[:, :, 0].flatten()
    axes[0].scatter(theta0_c[~div_c], sigma_c[~div_c], s=4, alpha=0.25, color=COLOR_OK)
    axes[0].scatter(theta0_c[div_c], sigma_c[div_c], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[0].set_title(f"中心化パラメータ化\nθ_1 ~ Normal(μ, σ)\ndivergence: {n_div_c}/{len(div_c)}")
    axes[0].set_xlabel("θ_1 (腕1のlogit-CTR)")
    axes[0].set_ylabel("σ_arm(腕間のばらつき)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    sigma_n = idata_noncentered.posterior["sigma_arm"].values.flatten()
    theta0_n = idata_noncentered.posterior["theta_logit"].values[:, :, 0].flatten()
    axes[1].scatter(theta0_n[~div_n], sigma_n[~div_n], s=4, alpha=0.25, color=COLOR_ALT)
    axes[1].scatter(theta0_n[div_n], sigma_n[div_n], s=12, alpha=0.85, color=COLOR_DIVERGENT, label="divergence")
    axes[1].set_title(f"非中心化パラメータ化\nθ_1 = μ + σ・offset_1\ndivergence: {n_div_n}/{len(div_n)}")
    axes[1].set_xlabel("θ_1 (腕1のlogit-CTR)")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("階層ロジスティック回帰(腕ごとのCTR)におけるfunnel回避", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "noncentered_ctr_funnel.png")
    plt.close(fig)
    print(f"noncentered_ctr_funnel.png saved (divergence: {n_div_c}/{len(div_c)} -> {n_div_n}/{len(div_n)})")


def plot_gp_hsgp_mean_basis_fix():
    """非ガウス尤度(Poisson)のHSGP回帰で、平均関数muを自由パラメータのまま
    多数の基底関数(m=20)を使うと深刻なdivergence/r_hat悪化を起こすが、
    muを固定定数にし基底関数数を絞る(m=6)とほぼ解消することを示す。
    mu単体の推定値はどちらも真値からズレる(ridge型の非識別性)が、
    実際にフィットされる合成量mu+fはどちらの設定でもほぼ同じ精度で
    真の対数レートを再現することも合わせて示す。"""

    rng = np.random.default_rng(6)
    N = 150
    X = np.linspace(0, 10, N)
    mu_true = np.log(30.0)
    true_f = 1.0 * np.sin(0.4 * X)
    log_rate_true = mu_true + true_f
    counts = rng.poisson(np.exp(log_rate_true))

    def fit(free_mu, m, eta_upper, seed):
        with pm.Model():
            if free_mu:
                mu = pm.Normal("mu", np.log(counts.mean() + 1), 2.0)
            else:
                mu = np.log(counts.mean() + 1)

            eta = pm.HalfNormal("eta", eta_upper)
            ls = 2.5  # 固定(GP自体の長さスケール非識別性は本デモの対象外)
            cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ls)
            gp = pm.gp.HSGP(m=[m], c=2.0, cov_func=cov)
            f = gp.prior("f", X=X[:, None])

            log_rate = mu + f
            pm.Poisson("y", mu=pm.math.exp(log_rate), observed=counts)

            idata = pm.sample(1000, tune=2500, chains=4, target_accept=0.95,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)

        n_div = int(idata.sample_stats["diverging"].sum())
        rhat_ds = az.rhat(idata)
        rhat_max = max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars)
        f_mean = idata.posterior["f"].values.reshape(-1, N).mean(axis=0)
        mu_mean = float(idata.posterior["mu"].values.mean()) if free_mu else float(mu)
        return n_div, rhat_max, f_mean, mu_mean

    n_div_broken, rhat_broken, f_broken, mu_broken = fit(True, 20, 5.0, 1)
    n_div_fixed, rhat_fixed, f_fixed, mu_fixed = fit(False, 6, 2.0, 1)

    lograte_broken = mu_broken + f_broken
    lograte_fixed = mu_fixed + f_fixed
    total_draws = 4000

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    labels = [f"muを自由推定\nm=20(壊れた設定)", f"muを固定定数\nm=6(修正後)"]
    x_pos = np.arange(2)
    divs = [n_div_broken, n_div_fixed]
    colors = [COLOR_DIVERGENT, COLOR_OK]
    axes[0].bar(x_pos, divs, color=colors, width=0.5)
    for i, (d, rhat) in enumerate(zip(divs, [rhat_broken, rhat_fixed])):
        axes[0].annotate(f"{d}/{total_draws}\nr_hat_max={rhat:.2f}", (i, d),
                          xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("divergence数")
    axes[0].set_title("divergence数とr_hat")

    axes[1].plot(X, log_rate_true, color="black", lw=2, label="真の対数レート")
    axes[1].plot(X, lograte_broken, color=COLOR_DIVERGENT, lw=1.5, ls="--",
                 label=f"壊れた設定のmu+f(mu={mu_broken:.2f})")
    axes[1].plot(X, lograte_fixed, color=COLOR_OK, lw=1.5, ls=":",
                 label=f"修正後のmu+f(mu={mu_fixed:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("log(rate)")
    axes[1].set_title(f"mu単体は真値(mu={mu_true:.2f})からズレるが\nmu+fはどちらもほぼ真の曲線を再現")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("HSGP: muを固定し基底関数数を絞ることでdivergenceを解消する\n(点推定mu+fは元々妥当だが、サンプリング自体の信頼性が違う)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "gp_hsgp_mean_basis_fix.png")
    plt.close(fig)

    print(f"gp_hsgp_mean_basis_fix.png saved "
          f"(divergence: {n_div_broken}/{total_draws} -> {n_div_fixed}/{total_draws}, "
          f"r_hat_max: {rhat_broken:.2f} -> {rhat_fixed:.2f}, "
          f"mu: broken={mu_broken:.2f} fixed={mu_fixed:.2f} true={mu_true:.2f})")


if __name__ == "__main__":
    plot_ratio_constant_epsilon_reparam()
    plot_derived_vs_independent_i0()
    plot_complexity_creates_nonidentifiability()
    plot_trig_reparameterization()
    plot_necessary_vs_wished_constraint()
    plot_ridge_ratio_reparameterization()
    plot_noncentered_ctr_funnel()
    plot_bym_theta_phi_ess_bym2_fix()
    plot_gp_hsgp_mean_basis_fix()
    plot_rw1_centering_beta0_confound()
