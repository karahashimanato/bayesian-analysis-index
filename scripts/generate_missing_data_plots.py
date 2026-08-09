"""
tools/missing-data.md に埋め込む可視化画像を生成するスクリプト。

  1. MCAR/MAR/MNAR: 3つの欠測メカニズムを合成データで再現し、
     値と欠測確率の関係が質的に異なることを示す。
  2. フルベイズ同時モデル: PyMCのマスク配列による自動補完を使い、
     完全ケース分析(CC)と比べて周辺平均のバイアスがMCAR/MARでは
     大きく減るが、MNARでは残ることを示す。

実行方法:
    source .venv/bin/activate
    python scripts/generate_missing_data_plots.py

出力先: assets/missing-data/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "missing-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_mcar_mar_mnar_mechanisms():
    """MCAR/MAR/MNARの3つの欠測メカニズムを合成データで再現し、
    値yと欠測確率の関係が質的に異なることを示す。"""

    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 1, n)

    p_mcar = np.full(n, 0.35)
    p_mar = 1 / (1 + np.exp(-(x - 0.5)))
    p_mnar = 1 / (1 + np.exp(-(y - y.mean())))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)
    for ax, (name, p) in zip(axes, [
        ("MCAR\n(欠測確率はyにもxにも依存しない)", p_mcar),
        ("MAR\n(欠測確率は観測済みxに依存)", p_mar),
        ("MNAR\n(欠測確率はy自体に依存)", p_mnar),
    ]):
        miss = rng.uniform(0, 1, n) < p
        ax.scatter(x[~miss], y[~miss], s=10, alpha=0.4, color=COLOR_OK, label="観測")
        ax.scatter(x[miss], y[miss], s=10, alpha=0.5, color=COLOR_DIVERGENT, label="欠測")
        ax.set_xlabel("x(観測済み)")
        ax.set_title(name, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("y")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("MCAR/MAR/MNAR: yの値と欠測/観測の分布の関係", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "mcar_mar_mnar_mechanisms.png")
    plt.close(fig)

    print("mcar_mar_mnar_mechanisms.png saved")


def plot_full_bayes_bias_comparison():
    """完全ケース分析(CC)とフルベイズ同時モデル(PyMCのマスク配列による
    自動補完)の周辺平均バイアスを、MCAR/MAR/MNARの3条件で比較する。"""

    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    true_beta0, true_beta1 = 2.0, 1.5
    y = true_beta0 + true_beta1 * x + rng.normal(0, 1, n)
    true_mean = y.mean()

    miss_mcar = rng.uniform(0, 1, n) < 0.3
    p_mar = 1 / (1 + np.exp(-(x - 0.5)))
    miss_mar = rng.uniform(0, 1, n) < p_mar
    p_mnar = 1 / (1 + np.exp(-(y - y.mean())))
    miss_mnar = rng.uniform(0, 1, n) < p_mnar

    def fit_full_bayes(miss, seed):
        y_masked = np.ma.masked_array(y, mask=miss)
        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 10)
            beta1 = pm.Normal("beta1", 0, 10)
            sigma = pm.HalfNormal("sigma", 5)
            mu = beta0 + beta1 * x
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_masked)
            idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                               random_seed=seed, progressbar=False,
                               compute_convergence_checks=False)
        y_mis_mean = idata.posterior["y_obs_unobserved"].values.reshape(-1, miss.sum()).mean(axis=0)
        y_full_est = y.copy()
        y_full_est[miss] = y_mis_mean
        return y_full_est.mean()

    results = {}
    for name, miss, seed in [("MCAR", miss_mcar, 1), ("MAR", miss_mar, 2), ("MNAR", miss_mnar, 3)]:
        cc_bias = y[~miss].mean() - true_mean
        fb_bias = fit_full_bayes(miss, seed) - true_mean
        results[name] = (cc_bias, fb_bias, miss.mean())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = list(results.keys())
    cc_biases = [results[k][0] for k in labels]
    fb_biases = [results[k][1] for k in labels]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, cc_biases, width=width, color=COLOR_DIVERGENT, label="完全ケース分析(CC)")
    ax.bar(x_pos + width / 2, fb_biases, width=width, color=COLOR_OK, label="フルベイズ同時モデル")
    ax.axhline(0, color="black", lw=0.8)
    for i, (cc, fb) in enumerate(zip(cc_biases, fb_biases)):
        ax.annotate(f"{cc:+.3f}", (i - width / 2, cc), xytext=(0, 4 if cc >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{fb:+.3f}", (i + width / 2, fb), xytext=(0, 4 if fb >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{k}\n(欠測率{results[k][2]:.0%})" for k in labels])
    ax.set_ylabel("周辺平均の推定バイアス")
    ax.set_title("フルベイズ同時モデルはMAR下でCCよりバイアスを大きく減らすが\nMNAR下では改善するもののバイアスが残る")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_bayes_bias_comparison.png")
    plt.close(fig)

    print("full_bayes_bias_comparison.png saved (" +
          ", ".join(f"{k}: CC={results[k][0]:+.3f} FullBayes={results[k][1]:+.3f}" for k in labels) + ")")


def _mice_impute_once(x_full, y_obs, miss, m, rng):
    """単一変数(y)のMICE: 観測データでOLS回帰をフィットし、欠測を
    predicted+残差ノイズでm回補完し、Rubinのルールで統合する。"""
    n = len(x_full)
    x_obs = x_full[~miss]
    y_o = y_obs[~miss]
    X_obs = np.column_stack([np.ones_like(x_obs), x_obs])
    beta_hat, _, _, _ = np.linalg.lstsq(X_obs, y_o, rcond=None)
    resid_std = (y_o - X_obs @ beta_hat).std(ddof=2)

    X_mis = np.column_stack([np.ones(miss.sum()), x_full[miss]])
    pred_mis = X_mis @ beta_hat

    means, variances = [], []
    for _ in range(m):
        y_imp = y_obs.copy()
        y_imp[miss] = pred_mis + rng.normal(0, resid_std, miss.sum())
        means.append(y_imp.mean())
        variances.append(y_imp.var(ddof=1) / n)
    q_bar = np.mean(means)
    total_var = np.mean(variances) + (1 + 1 / m) * np.var(means, ddof=1)
    return q_bar, np.sqrt(total_var)


def plot_mice_coverage():
    """単一変数のMICE(chained equationsの簡易版、m=20回)を繰り返しシミュレーションし、
    正規近似95%信用区間の実測カバレッジが名目の95%を下回ることを示す。"""

    rng = np.random.default_rng(0)
    n = 200
    true_beta0, true_beta1 = 2.0, 1.5
    true_mean = true_beta0
    n_rep = 60
    m_impute = 20

    results = []
    for _ in range(n_rep):
        x = rng.normal(0, 1, n)
        y = true_beta0 + true_beta1 * x + rng.normal(0, 1, n)
        miss = rng.uniform(0, 1, n) < 0.3
        q_bar, se = _mice_impute_once(x, y, miss, m_impute, rng)
        results.append((q_bar, q_bar - 1.96 * se, q_bar + 1.96 * se))

    covered = np.array([lo <= true_mean <= hi for _, lo, hi in results])
    coverage_rate = covered.mean()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    order = np.arange(n_rep)
    for i, (mean, lo, hi) in enumerate(results):
        color = COLOR_OK if covered[i] else COLOR_DIVERGENT
        ax.plot([lo, hi], [i, i], color=color, lw=1.2, alpha=0.8)
        ax.plot(mean, i, "o", color=color, ms=2.5)
    ax.axvline(true_mean, color="black", lw=1.3, ls="--", label=f"真の周辺平均={true_mean}")
    ax.plot([], [], color=COLOR_OK, lw=1.5, label="95%CIが真値を含む")
    ax.plot([], [], color=COLOR_DIVERGENT, lw=1.5, label="95%CIが真値を含まない")
    ax.set_xlabel("MICEで推定した周辺平均(95%信用区間)")
    ax.set_ylabel("シミュレーション反復")
    ax.set_title(f"MICE(m={m_impute})の正規近似95%区間の実測カバレッジ={coverage_rate:.1%}\n"
                 f"(名目の95%を下回る)")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mice_coverage.png")
    plt.close(fig)

    print(f"mice_coverage.png saved (empirical coverage over {n_rep} reps: {coverage_rate:.1%}, nominal 95%)")


def plot_selection_model_recovery():
    """Selection Model(Heckman型)をPyMCで実際にサンプリングし、アウトカム式
    (y ~ Normal(beta0+beta1・x, sigma))と選択式(観測確率がyの値自体にも
    依存するロジスティック選択式)を同時に推定した場合の、非無視可能性
    パラメータgamma_yの復元を示す。選択式に、アウトカム式には含まれない
    除外制約変数(z_cov)を入れることで識別を確保している。"""

    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(0, 1, n)
    z_cov = rng.normal(0, 1, n)  # 除外制約: 選択式のみに現れる変数

    true_beta0, true_beta1 = 2.0, 1.5
    true_gamma_y = -1.0
    true_alpha0, true_alpha1 = 0.3, 0.5

    y = true_beta0 + true_beta1 * x + rng.normal(0, 1, n)
    selection_logit = true_alpha0 + true_alpha1 * z_cov + true_gamma_y * y
    p_obs_true = 1 / (1 + np.exp(-selection_logit))
    observed_ind = rng.uniform(0, 1, n) < p_obs_true

    y_masked = np.ma.masked_array(y, mask=~observed_ind)

    with pm.Model():
        beta0 = pm.Normal("beta0", 0, 10)
        beta1 = pm.Normal("beta1", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        y_rv = pm.Normal("y", mu=beta0 + beta1 * x, sigma=sigma, observed=y_masked)

        alpha0 = pm.Normal("alpha0", 0, 5)
        alpha1 = pm.Normal("alpha1", 0, 5)
        gamma_y = pm.Normal("gamma_y", 0, 5)
        p_obs = pm.math.sigmoid(alpha0 + alpha1 * z_cov + gamma_y * y_rv)
        pm.Bernoulli("observed_ind", p=p_obs, observed=observed_ind.astype(int))

        idata = pm.sample(1000, tune=1500, chains=4, target_accept=0.9,
                           random_seed=1, progressbar=False,
                           compute_convergence_checks=False)

    n_div = int(idata.sample_stats["diverging"].sum())
    gamma_draws = idata.posterior["gamma_y"].values.flatten()
    gamma_mean = gamma_draws.mean()
    gamma_lo, gamma_hi = np.percentile(gamma_draws, [2.5, 97.5])
    beta0_est = float(idata.posterior["beta0"].values.mean())
    beta1_est = float(idata.posterior["beta1"].values.mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(gamma_draws, bins=50, density=True, color=COLOR_OK, alpha=0.6,
            label=f"gamma_y事後分布(平均={gamma_mean:.2f})")
    ax.axvline(true_gamma_y, color=COLOR_DIVERGENT, lw=1.5, ls="--", label=f"真値={true_gamma_y}")
    ax.axvline(0, color="black", lw=1, ls=":", label="gamma_y=0(MAR相当)")
    ax.set_xlabel("gamma_y(観測確率がyの値自体に依存する強さ)")
    ax.set_ylabel("density")
    ax.set_title(f"観測率{observed_ind.mean():.0%}のMNARデータからgamma_yを復元\n"
                 f"(divergence={n_div}, beta0={beta0_est:.2f}(真値{true_beta0}), "
                 f"beta1={beta1_est:.2f}(真値{true_beta1}))")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "selection_model_recovery.png")
    plt.close(fig)

    print(f"selection_model_recovery.png saved "
          f"(観測率={observed_ind.mean():.1%}, divergence={n_div}, "
          f"gamma_y={gamma_mean:.3f}[{gamma_lo:.2f},{gamma_hi:.2f}](true {true_gamma_y}), "
          f"beta0={beta0_est:.3f}(true {true_beta0}), beta1={beta1_est:.3f}(true {true_beta1}))")


def plot_pattern_mixture_sensitivity():
    """Pattern-Mixture Modelの感度パラメータdeltaを動かし、欠測群の分布が
    観測群の分布からどれだけズレていると仮定するかによって周辺平均の推定が
    どう変わるかを示す。真の周辺平均を再現するdelta*を、合成データでは
    既知の真値として求められる。"""

    rng = np.random.default_rng(21)
    n = 400
    true_delta = 1.2  # 欠測群は観測群よりyがtrue_deltaだけ高い(Pattern-Mixtureの前提通りの構造)
    miss = rng.uniform(0, 1, n) < 0.5  # どちらの群に入るか自体はyと無関係に半々で決まる
    mu_base = 3.0
    y_full = mu_base + np.where(miss, true_delta, 0.0) + rng.normal(0, 1.0, n)

    mu_obs = y_full[~miss].mean()
    p_miss = miss.mean()
    true_marginal_mean = y_full.mean()

    delta_range = np.linspace(-1, 3, 200)
    implied_mean = (1 - p_miss) * mu_obs + p_miss * (mu_obs + delta_range)
    delta_star = float(np.interp(true_marginal_mean, implied_mean, delta_range))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(delta_range, implied_mean, color=COLOR_OK, lw=2)
    ax.axhline(true_marginal_mean, color=COLOR_DIVERGENT, lw=1.3, ls="--",
               label=f"真の周辺平均={true_marginal_mean:.3f}")
    ax.axvline(0, color="black", lw=1, ls=":", label="delta=0(MAR相当)")
    ax.axvline(delta_star, color=COLOR_ALT, lw=1.3, ls="--",
               label=f"delta*={delta_star:.3f}(真の周辺平均を再現)")
    ax.scatter([delta_star], [true_marginal_mean], color=COLOR_ALT, zorder=5, s=50)
    ax.set_xlabel("感度パラメータ delta")
    ax.set_ylabel("周辺平均の推定値")
    ax.set_title(f"deltaを動かすと周辺平均の推定がどう変わるか\n"
                 f"(delta*は事後的にしか分からない、真のdelta={true_delta})")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pattern_mixture_sensitivity.png")
    plt.close(fig)

    print(f"pattern_mixture_sensitivity.png saved "
          f"(delta*={delta_star:.3f}, 真のdelta={true_delta}, 真の周辺平均={true_marginal_mean:.3f}, "
          f"mu_obs={mu_obs:.3f}, 欠測率={p_miss:.1%})")


def plot_multivariate_joint_imputation():
    """相関する2変数(y1,y2)を、独立に補完するモデルと、逐次条件付け分解
    (y2|y1の回帰係数beta_crossを使う)による同時補完モデルで実際にPyMCで
    フィットし、補完精度(RMSE)を比較する。"""

    rng = np.random.default_rng(5)
    n = 300
    rho_true = 0.7
    mu1_true, mu2_true = 5.0, 3.0
    sigma1_true, sigma2_true = 1.0, 1.5

    y1_full = rng.normal(mu1_true, sigma1_true, n)
    y2_full = mu2_true + rho_true * sigma2_true / sigma1_true * (y1_full - mu1_true) + \
        rng.normal(0, sigma2_true * np.sqrt(1 - rho_true ** 2), n)

    # y2のみ30%が欠測(y1は全て観測: 「もう片方が観測されている」ケースを再現)
    miss2 = rng.uniform(0, 1, n) < 0.3
    y2_masked = np.ma.masked_array(y2_full, mask=miss2)

    with pm.Model():
        mu2 = pm.Normal("mu2", 0, 10)
        sigma2 = pm.HalfNormal("sigma2", 5)
        pm.Normal("y2_indep", mu=mu2, sigma=sigma2, observed=y2_masked)
        idata_indep = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                                 random_seed=1, progressbar=False,
                                 compute_convergence_checks=False)
    y2_mis_indep = idata_indep.posterior["y2_indep_unobserved"].values.reshape(-1, miss2.sum()).mean(axis=0)

    with pm.Model():
        mu2 = pm.Normal("mu2", 0, 10)
        sigma2 = pm.HalfNormal("sigma2", 5)
        beta_cross = pm.Normal("beta_cross", 0, 5)
        mu2_cond = mu2 + beta_cross * (y1_full - y1_full.mean())
        pm.Normal("y2_joint", mu=mu2_cond, sigma=sigma2, observed=y2_masked)
        idata_joint = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                                 random_seed=2, progressbar=False,
                                 compute_convergence_checks=False)
    y2_mis_joint = idata_joint.posterior["y2_joint_unobserved"].values.reshape(-1, miss2.sum()).mean(axis=0)

    y2_true_missing = y2_full[miss2]
    rmse_indep = float(np.sqrt(np.mean((y2_mis_indep - y2_true_missing) ** 2)))
    rmse_joint = float(np.sqrt(np.mean((y2_mis_joint - y2_true_missing) ** 2)))
    improvement = (rmse_indep - rmse_joint) / rmse_indep

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = ["独立モデル\n(y2のみで補完)", "同時モデル\n(y1との相関を利用)"]
    rmses = [rmse_indep, rmse_joint]
    ax.bar(labels, rmses, color=[COLOR_DIVERGENT, COLOR_OK], width=0.5)
    for i, r in enumerate(rmses):
        ax.annotate(f"{r:.3f}", (i, r), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylabel("補完値のRMSE(真の欠測値との差)")
    ax.set_title(f"相関rho={rho_true}の2変数で、同時モデルは独立モデルより\n"
                 f"補完RMSEを{improvement:.1%}改善する")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multivariate_joint_imputation.png")
    plt.close(fig)

    print(f"multivariate_joint_imputation.png saved "
          f"(RMSE: 独立={rmse_indep:.3f}, 同時={rmse_joint:.3f}, 改善={improvement:.1%})")


if __name__ == "__main__":
    plot_mcar_mar_mnar_mechanisms()
    plot_full_bayes_bias_comparison()
    plot_mice_coverage()
    plot_selection_model_recovery()
    plot_pattern_mixture_sensitivity()
    plot_multivariate_joint_imputation()
