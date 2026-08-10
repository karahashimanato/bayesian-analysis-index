"""
techniques/prior-predictive-check.md に埋め込む可視化画像を生成するスクリプト。

「分母のパラメータが0に近づくと分散が発散する」病理(Gamma-Poisson階層モデルの
μ²/α_concなど)を、Exponential事前分布とGamma(shape>1)事前分布の
prior predictiveの違いとして実際にサンプリングして描画する。

「判断基準は極値だけでなく割合・信用区間幅で見る」の実例として、理論上の
範囲(min/max)が同じ2つのprior predictive分布(Beta(5,5)とBeta(0.3,0.3)を
同じ[0,1000]にスケール)が、現実的な範囲に入る質量の割合ではどれだけ
異なるかを比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_prior_predictive_plots.py

出力先: assets/prior-predictive/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "prior-predictive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_steady_state_prior_center():
    """SISモデルの解析的な定常状態(endemic equilibrium) I*/N = 1 - gamma/beta
    に、EDAで観測した平衡感染割合を代入してbetaの事前分布の中心値を逆算する。
    「勘で決めた」事前分布と「定常状態の式から逆算した」事前分布とで、
    prior predictiveの平衡感染割合がどれだけ現実的な範囲に集中するかを
    実際にサンプリングして比較する。"""

    rng = np.random.default_rng(7)
    n = 50000
    gamma_fixed = 0.2  # 平均感染期間の逆数(EDAで既知とする)
    observed_plateau = 0.30  # EDAで観測した平衡感染割合

    # 勘で決めた事前分布: betaのスケール感を漠然とUniform(0,3)に置く
    beta_naive = rng.uniform(0.01, 3.0, n)

    # 定常状態の式 I*/N = 1 - gamma/beta を実データの平衡水準に代入して
    # beta0 = gamma / (1 - observed_plateau) を逆算し、その周りにLogNormalを置く
    beta0_derived = gamma_fixed / (1 - observed_plateau)
    beta_derived = rng.lognormal(np.log(beta0_derived), 0.3, n)

    def equilibrium(beta):
        return np.clip(1 - gamma_fixed / beta, 0, 1)

    eq_naive = equilibrium(beta_naive)
    eq_derived = equilibrium(beta_derived)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(eq_naive, bins=60, density=True, color=COLOR_DIVERGENT, alpha=0.6,
                 label=f"勘で決めた事前分布 beta~Uniform(0.01,3)\n(std={eq_naive.std():.3f})")
    axes[0].axvline(observed_plateau, color="black", ls="--", lw=1.5, label=f"EDAで観測した平衡水準={observed_plateau}")
    axes[0].set_xlabel("prior predictiveの平衡感染割合 I*/N")
    axes[0].set_ylabel("density")
    axes[0].set_title("勘で決めた事前分布は\n平衡水準がほぼ一様に散らばる")
    axes[0].legend(fontsize=8.5)

    axes[1].hist(eq_derived, bins=60, density=True, color=COLOR_OK, alpha=0.6,
                 label=f"定常状態の式から逆算 beta0={beta0_derived:.3f}\n(std={eq_derived.std():.3f})")
    axes[1].axvline(observed_plateau, color="black", ls="--", lw=1.5, label=f"EDAで観測した平衡水準={observed_plateau}")
    axes[1].set_xlabel("prior predictiveの平衡感染割合 I*/N")
    axes[1].set_ylabel("density")
    axes[1].set_title("定常状態の式から逆算すると\n観測した平衡水準の周りに集中する")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("解析的な定常状態の式にEDAの実測値を代入して事前分布の中心を決める", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "steady_state_prior_center.png")
    plt.close(fig)

    print(f"steady_state_prior_center.png saved "
          f"(観測平衡水準={observed_plateau}, 逆算したbeta0={beta0_derived:.3f}, "
          f"勘の事前分布std={eq_naive.std():.3f}, 逆算事前分布std={eq_derived.std():.3f})")


def plot_jensen_prior_predictive_catch():
    """対数正規リンクのモデル y = exp(mu + sigma*eps) で、muの事前分布(中心)を
    一切変えずにsigmaの事前分布(散らばり)だけを広げても、Jensen不等式
    (E[exp(X)] = exp(mu+sigma^2/2))により予測平均が体系的にズレることを
    実際にサンプリングして示す。この種のズレはmuの事前分布だけを眺めていても
    気づけず、prior predictiveを実際に計算して初めて発覚する。"""

    rng = np.random.default_rng(11)
    n = 30000
    mu_fixed = 0.0  # muの事前分布の中心は一切変えない

    sigma_scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    pred_means = []
    for s in sigma_scales:
        sigma = stats.halfnorm.rvs(scale=s, size=n, random_state=rng)
        eps = rng.normal(0, 1, n)
        y = np.exp(mu_fixed + sigma * eps)
        pred_means.append(y.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(sigma_scales, pred_means, color=COLOR_DIVERGENT, lw=1.5, marker="o")
    axes[0].axhline(np.exp(mu_fixed), color="black", ls="--", lw=1.2,
                     label=f"muだけから期待される値 exp(mu)={np.exp(mu_fixed):.1f}")
    axes[0].set_xlabel("sigmaの事前分布のスケール(muは常に0のまま)")
    axes[0].set_ylabel("prior predictiveの平均 E[y]")
    axes[0].set_title("muを一切動かさなくても\nsigmaの事前分布を広げるだけで予測平均がズレる")
    axes[0].legend(fontsize=8.5)

    sigma_wide = stats.halfnorm.rvs(scale=1.0, size=n, random_state=rng)
    eps_wide = rng.normal(0, 1, n)
    y_wide = np.exp(mu_fixed + sigma_wide * eps_wide)
    axes[1].hist(np.clip(y_wide, 0, 10), bins=80, color=COLOR_ALT, alpha=0.7,
                 label=f"sigma事前分布スケール=1.0のprior predictive\n平均={y_wide.mean():.2f}(中央値={np.median(y_wide):.2f})")
    axes[1].axvline(np.exp(mu_fixed), color="black", ls="--", lw=1.5,
                     label=f"muだけから期待される値={np.exp(mu_fixed):.1f}")
    axes[1].set_xlabel("y = exp(mu + sigma*eps)  (10で打ち切って表示)")
    axes[1].set_ylabel("count")
    axes[1].set_title("実際にヒストグラムを見て初めて\n平均が右に大きくズレていると気づける")
    axes[1].legend(fontsize=8)

    fig.suptitle("Jensen不等式由来のズレは、prior predictiveを実際に計算しないと気づけない", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "jensen_prior_predictive_catch.png")
    plt.close(fig)

    print("jensen_prior_predictive_catch.png saved (" +
          ", ".join(f"sigma_scale={s}:E[y]={m:.3f}" for s, m in zip(sigma_scales, pred_means)) +
          f" | muのみから期待される値={np.exp(mu_fixed):.3f})")


def plot_runaway_culprit_shifts():
    """Hawkes過程の期待総イベント数 mu*T/(1-kappa/beta) は分岐比kappa/betaが
    1に近づくと発散する。kappaの事前分布だけを引き締めても、betaの事前分布は
    そのままなら分岐比はまだ1に近づきうるため、暴走の主犯がbetaへ移るだけで
    prior predictiveの暴走自体は解消しないことをペアプロットで示す。"""

    rng = np.random.default_rng(9)
    n = 20000
    mu_fixed, T_fixed = 0.3, 50.0

    def expected_total(kappa, beta):
        branching = kappa / beta
        with np.errstate(divide="ignore", invalid="ignore"):
            total = np.where(branching < 0.999, mu_fixed * T_fixed / (1 - branching), np.nan)
        return branching, total

    # 段階1: kappa・betaとも同じ広い事前分布
    kappa_1 = stats.halfnorm.rvs(scale=1.0, size=n, random_state=rng)
    beta_1 = stats.halfnorm.rvs(scale=1.0, size=n, random_state=rng)
    branching_1, total_1 = expected_total(kappa_1, beta_1)
    exploded_1 = np.isnan(total_1) | (total_1 > 5000)

    # 段階2: kappaの事前分布だけを構造的に引き締める(betaはそのまま)
    kappa_2 = stats.halfnorm.rvs(scale=0.3, size=n, random_state=rng)
    beta_2 = stats.halfnorm.rvs(scale=1.0, size=n, random_state=rng)
    branching_2, total_2 = expected_total(kappa_2, beta_2)
    exploded_2 = np.isnan(total_2) | (total_2 > 5000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, kappa, beta, exploded, title, frac in [
        (axes[0], kappa_1, beta_1, exploded_1, "段階1: kappa・betaとも広い事前分布\n(暴走の主犯はどちらでもありうる)", exploded_1.mean()),
        (axes[1], kappa_2, beta_2, exploded_2, "段階2: kappaだけ引き締めても\n暴走の主犯がbetaへ移るだけ", exploded_2.mean()),
    ]:
        ax.scatter(beta[~exploded], kappa[~exploded], s=3, color=COLOR_OK, alpha=0.3, label="暴走せず")
        ax.scatter(beta[exploded], kappa[exploded], s=3, color=COLOR_DIVERGENT, alpha=0.3, label="暴走(分岐比が1に近い)")
        line_x = np.linspace(0.01, max(beta.max(), kappa.max()), 100)
        ax.plot(line_x, line_x, color="black", ls="--", lw=1.0, label="kappa=beta(分岐比=1)")
        ax.set_xlabel("beta")
        ax.set_ylabel("kappa")
        ax.set_xlim(0, np.percentile(beta, 99.5))
        ax.set_ylim(0, np.percentile(np.concatenate([kappa_1, kappa_2]), 99.5))
        ax.set_title(f"{title}\n(暴走割合={frac:.1%})")
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("kappa・betaのペアプロットで、暴走の主犯が移っただけかを確認する", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "runaway_culprit_shifts.png")
    plt.close(fig)

    print(f"runaway_culprit_shifts.png saved "
          f"(段階1暴走割合={exploded_1.mean():.1%}, 段階2暴走割合={exploded_2.mean():.1%}, "
          f"段階1 total中央値={np.nanmedian(total_1):.0f}, 段階2 total中央値={np.nanmedian(total_2):.0f})")


def plot_nonuniform_dynamic_range_coverage():
    """対象関数の値域が場所によって大きく異なる(非一様なダイナミックレンジを
    持つ)場合、GPの単一の振幅etaでは平坦な領域と急峻な領域を両立して
    タイトに覆うことが難しいことを、なだらかな関数と非一様な関数の
    prior predictive包絡線を実際に計算して比較する。"""

    rng = np.random.default_rng(13)
    x = np.linspace(-2, 2, 200)
    n_draws = 50

    def smooth_target(x):
        return 0.5 * x  # なだらかで値域が一様(std小)

    def peaky_target(x):
        return 0.5 * x - 8.0 * np.exp(-8.0 * x ** 2)  # 大部分は平坦だが中央だけ急峻

    def rbf_kernel(x1, x2, ell, eta):
        d2 = (x1[:, None] - x2[None, :]) ** 2
        return eta ** 2 * np.exp(-0.5 * d2 / ell ** 2)

    def draw_envelope(target_fn, ell_scale, eta_scale):
        y_true = target_fn(x)
        ell_draws = stats.gamma.rvs(a=3, scale=ell_scale, size=n_draws, random_state=rng)
        eta_draws = stats.halfnorm.rvs(scale=eta_scale, size=n_draws, random_state=rng)
        funcs = np.empty((n_draws, len(x)))
        for i in range(n_draws):
            K = rbf_kernel(x, x, ell_draws[i], eta_draws[i]) + 1e-6 * np.eye(len(x))
            L = np.linalg.cholesky(K)
            funcs[i] = L @ rng.normal(size=len(x))
        return y_true, funcs

    y_smooth, funcs_smooth = draw_envelope(smooth_target, 0.5, 0.5)
    # peakyな対象は、ピーク(-8)まで包絡線が届くようetaを広げる必要がある
    y_peaky, funcs_peaky = draw_envelope(peaky_target, 0.5, 2.0)

    flat_mask = np.abs(x) > 1.0  # peaky対象のうち「平坦な領域」だけを見る

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, y_true, funcs, eta_scale, title in [
        (axes[0], y_smooth, funcs_smooth, 0.5, "値域が一様な対象関数\n(std={:.2f}, eta事前分布スケール=0.5)".format(y_smooth.std())),
        (axes[1], y_peaky, funcs_peaky, 2.0, "値域が非一様な対象関数\n(std={:.2f}, eta事前分布スケール=2.0)".format(y_peaky.std())),
    ]:
        env_lo, env_hi = funcs.min(axis=0), funcs.max(axis=0)
        for i in range(n_draws):
            ax.plot(x, funcs[i], color=COLOR_OK, alpha=0.1, lw=1)
        ax.fill_between(x, env_lo, env_hi, color=COLOR_OK, alpha=0.1, label=f"eta事前分布スケール={eta_scale}の包絡線")
        ax.plot(x, y_true, color=COLOR_DIVERGENT, lw=2, label="真の対象関数")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(title)
        ax.legend(fontsize=8.5)

    flat_true_range = float(y_peaky[flat_mask].max() - y_peaky[flat_mask].min())
    flat_env_range = float(np.mean(funcs_peaky[:, flat_mask].max(axis=0) - funcs_peaky[:, flat_mask].min(axis=0)))
    axes[1].set_title(axes[1].get_title() + f"\n(平坦域|x|>1での余剰幅: 真の値域{flat_true_range:.2f} vs 包絡線{flat_env_range:.2f})")

    fig.suptitle("ピークまで届くようetaを広げると、その同じetaが平坦な領域を\nかなり余分に覆ってしまう(単一etaでの両立が難しい)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "nonuniform_dynamic_range_coverage.png")
    plt.close(fig)

    print(f"nonuniform_dynamic_range_coverage.png saved "
          f"(smooth: std={y_smooth.std():.3f}, peaky: std={y_peaky.std():.3f}, "
          f"peakyの平坦域|x|>1での真の値域={flat_true_range:.3f} vs 包絡線の幅={flat_env_range:.3f})")


def plot_borrowed_vs_derived_lengthscale():
    """チュートリアルの(単位の異なる)長さスケールをそのまま流用した場合と、
    実データの点間隔の中央値から長さスケールを導出した場合とで、GPの
    prior predictiveドローが実データの空間スケールとどれだけ噛み合うかを
    実際に計算して比較する。"""

    rng = np.random.default_rng(17)
    # 実データ: 1次元上のイベント位置(km単位)で、最近傍間隔の中央値が2.4km
    n_events = 40
    event_positions = np.sort(rng.uniform(0, 50, n_events))
    nn_gaps = np.diff(event_positions)
    median_gap = float(np.median(nn_gaps))

    x = np.linspace(0, 50, 300)

    def rbf_kernel(x1, x2, ell, eta=1.0):
        d2 = (x1[:, None] - x2[None, :]) ** 2
        return eta ** 2 * np.exp(-0.5 * d2 / ell ** 2)

    def draw_functions(ell, n_draws=8):
        K = rbf_kernel(x, x, ell) + 1e-6 * np.eye(len(x))
        L = np.linalg.cholesky(K)
        return np.array([L @ rng.normal(size=len(x)) for _ in range(n_draws)])

    ell_borrowed = 12.0  # 別チュートリアル(anonymous unit)からそのまま流用した値
    ell_derived = median_gap  # このデータの最近傍間隔の中央値から導出

    funcs_borrowed = draw_functions(ell_borrowed)
    funcs_derived = draw_functions(ell_derived)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, funcs, ell, title in [
        (axes[0], funcs_borrowed, ell_borrowed, f"チュートリアルの値をそのまま流用\nell={ell_borrowed}(別データの単位のまま)"),
        (axes[1], funcs_derived, ell_derived, f"このデータの最近傍間隔の中央値から導出\nell={ell_derived:.2f}km"),
    ]:
        for f in funcs:
            ax.plot(x, f, color=COLOR_OK, alpha=0.5, lw=1)
        for e in event_positions:
            ax.axvline(e, color=COLOR_DIVERGENT, alpha=0.25, lw=0.8)
        ax.set_xlabel("x(km)")
        ax.set_ylabel("f(x)(prior predictiveのドロー)")
        ax.set_title(title)

    fig.suptitle(f"実データの最近傍間隔(中央値={median_gap:.2f}km、縦線はイベント位置)に対し、\n"
                 f"流用したellは関数の変動スケールが噛み合わない", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "borrowed_vs_derived_lengthscale.png")
    plt.close(fig)

    print(f"borrowed_vs_derived_lengthscale.png saved "
          f"(実データの最近傍間隔中央値={median_gap:.3f}km, "
          f"流用ell={ell_borrowed}, 導出ell={ell_derived:.3f})")


def plot_reference_prior_vs_data_scale():
    """BYM2のような相対リスクモデルの切片beta0について、参照実装の緩い
    事前分布(N(0,5))をそのまま使った場合と、このデータの期待件数Eのスケール
    から締め直した事前分布(N(0,0.3))とで、prior predictiveの合計件数が
    実測の合計件数に対してどれだけ現実的かを実際にサンプリングして比較する。"""

    rng = np.random.default_rng(19)
    n_areas = 60
    E = rng.uniform(3, 20, n_areas)  # 各地区の期待件数(EDAで既知)
    observed_total = float(np.round(E.sum() * 1.05))  # 実測の合計件数(相対リスク~1.05相当)

    n_draws = 20000
    beta0_reference = rng.normal(0, 5.0, n_draws)
    beta0_derived = rng.normal(0, 0.3, n_draws)

    total_reference = np.array([np.sum(E * np.exp(b)) for b in beta0_reference[:2000]])
    total_derived = np.array([np.sum(E * np.exp(b)) for b in beta0_derived[:2000]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bins_ref = np.logspace(0, np.log10(max(total_reference.max(), 10)), 60)
    axes[0].hist(total_reference, bins=bins_ref, color=COLOR_DIVERGENT, alpha=0.7)
    axes[0].axvline(observed_total, color="black", ls="--", lw=1.5, label=f"実測の合計件数={observed_total:.0f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("prior predictiveの合計件数(対数軸)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"参照実装の事前分布 beta0~N(0,5)\n(中央値={np.median(total_reference):,.0f}, "
                       f"最大={total_reference.max():,.0f})")
    axes[0].legend(fontsize=8.5)

    axes[1].hist(total_derived, bins=60, color=COLOR_OK, alpha=0.7)
    axes[1].axvline(observed_total, color="black", ls="--", lw=1.5, label=f"実測の合計件数={observed_total:.0f}")
    axes[1].set_xlabel("prior predictiveの合計件数")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"データの実スケールから締め直した事前分布 beta0~N(0,0.3)\n(中央値={np.median(total_derived):,.0f}, "
                       f"最大={total_derived.max():,.0f})")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("参照実装の事前分布は中央値こそ近いが、裾が実測の数億倍まで暴走しうる。\nデータの実スケールから締め直すと裾も現実的な範囲に収まる", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "reference_prior_vs_data_scale.png")
    plt.close(fig)

    print(f"reference_prior_vs_data_scale.png saved "
          f"(実測合計={observed_total:.0f}, "
          f"参照実装prior: 中央値={np.median(total_reference):.0f} 最大={total_reference.max():.0f}, "
          f"締め直しprior: 中央値={np.median(total_derived):.0f} 最大={total_derived.max():.0f})")


def plot_denominator_variance_explosion():
    """分母に来るパラメータαの事前分布次第で、分散 mu^2/alpha の
    prior predictiveが暴走するかどうかが決まることを示す。"""

    rng = np.random.default_rng(0)
    n = 50000
    mu = 10.0  # 観測データの平均スケール(固定)

    alpha_exp = rng.exponential(1.0, n)       # Exponential(1): 0で密度が最大
    alpha_gamma = rng.gamma(2.0, 1.0, n)      # Gamma(shape=2): 0で密度がゼロ

    var_exp = mu**2 / alpha_exp
    var_gamma = mu**2 / alpha_gamma

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # 左: alphaそのものの事前分布(0付近の密度の違いが原因)
    x = np.linspace(0.001, 4, 500)
    axes[0].plot(x, stats.expon.pdf(x, scale=1.0), color=COLOR_DIVERGENT,
                 label="Exponential(1): 0で密度が最大")
    axes[0].plot(x, stats.gamma.pdf(x, a=2.0, scale=1.0), color=COLOR_OK,
                 label="Gamma(shape=2): 0で密度がゼロ")
    axes[0].set_xlabel("α(分母のパラメータ)")
    axes[0].set_ylabel("density")
    axes[0].set_title("原因: αの事前分布が\n0付近にどれだけ質量を置くか")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # 右: 結果として生じる分散 mu^2/alpha の prior predictive(対数軸)
    bins = np.logspace(0, 6, 60)
    axes[1].hist(var_exp, bins=bins, alpha=0.6, color=COLOR_DIVERGENT,
                 label=f"Exponential(1)由来\n(99%ile={np.percentile(var_exp,99):,.0f}, "
                       f"最大={var_exp.max():,.0f})")
    axes[1].hist(var_gamma, bins=bins, alpha=0.6, color=COLOR_OK,
                 label=f"Gamma(shape=2)由来\n(99%ile={np.percentile(var_gamma,99):,.0f}, "
                       f"最大={var_gamma.max():,.0f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("prior predictiveの分散 = μ²/α (μ=10で固定)")
    axes[1].set_ylabel("count")
    axes[1].set_title("結果: 分散のprior predictiveが\n暴走するかどうか")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("「分母のパラメータが0に近づくと分散が発散する」病理", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "denominator_variance_explosion.png")
    plt.close(fig)
    print(f"denominator_variance_explosion.png saved "
          f"(Exponential p99={np.percentile(var_exp,99):.0f} max={var_exp.max():.0f}, "
          f"Gamma p99={np.percentile(var_gamma,99):.0f} max={var_gamma.max():.0f})")


def plot_extreme_vs_proportion():
    """理論上の範囲(min/max)が同じでも、現実的な範囲に入る質量の割合は
    形状次第で大きく異なることを示す。"""

    rng = np.random.default_rng(1)
    n = 20000
    scale = 1000.0
    band_lo, band_hi = 300.0, 700.0

    well = rng.beta(5.0, 5.0, n) * scale       # 中央に集中
    mis = rng.beta(0.3, 0.3, n) * scale        # 両端(0, 1000)に偏るU字型

    prop_well = np.mean((well >= band_lo) & (well <= band_hi))
    prop_mis = np.mean((mis >= band_lo) & (mis <= band_hi))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bins = np.linspace(0, scale, 60)
    axes[0].axvspan(band_lo, band_hi, color="gray", alpha=0.15, label="現実的な範囲[300,700]")
    axes[0].hist(well, bins=bins, alpha=0.6, color=COLOR_OK,
                 label=f"Beta(5,5)(min={well.min():.0f}, max={well.max():.0f})")
    axes[0].hist(mis, bins=bins, alpha=0.6, color=COLOR_DIVERGENT,
                 label=f"Beta(0.3,0.3)(min={mis.min():.0f}, max={mis.max():.0f})")
    axes[0].set_xlabel("prior predictiveの値")
    axes[0].set_ylabel("count")
    axes[0].set_title("理論上の範囲[0,1000]はほぼ同じ")
    axes[0].legend(loc="upper center", fontsize=8, framealpha=0.9)

    labels = ["Beta(5,5)\n(中央集中)", "Beta(0.3,0.3)\n(両端に偏る)"]
    props = [prop_well * 100, prop_mis * 100]
    axes[1].bar(labels, props, color=[COLOR_OK, COLOR_DIVERGENT], alpha=0.85)
    for i, p in enumerate(props):
        axes[1].text(i, p + 1.5, f"{p:.1f}%", ha="center", fontsize=11)
    axes[1].set_ylabel("現実的な範囲[300,700]に入る割合(%)")
    axes[1].set_title("割合で見ると大きく異なる")
    axes[1].set_ylim(0, 100)

    fig.suptitle("min/maxの範囲が同じでも、現実的な範囲に入る割合は形状次第で大きく異なる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "extreme_vs_proportion.png")
    plt.close(fig)
    print(f"extreme_vs_proportion.png saved "
          f"(Beta(5,5): min={well.min():.0f} max={well.max():.0f} 帯域内={prop_well*100:.1f}%, "
          f"Beta(0.3,0.3): min={mis.min():.0f} max={mis.max():.0f} 帯域内={prop_mis*100:.1f}%)")


def plot_gp_hyperparameter_envelope():
    """GP回帰のハイパーパラメータ(長さスケールell、振幅eta、観測ノイズsigma)を
    事前分布から50回ドローし、それぞれで得られる関数を実データのスケールに
    重ねて、包絡線がデータの範囲を覆いつつ暴走していないかを目視確認する。"""

    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 100)
    n_draws = 50

    # 実データのスケールを模したもの(例: 世界平均気温偏差)
    data_lo, data_hi = -0.5, 1.3

    ell_draws = stats.gamma.rvs(a=3, scale=1, size=n_draws, random_state=rng)
    eta_draws = stats.halfnorm.rvs(scale=0.5, size=n_draws, random_state=rng)
    sigma_draws = stats.halfnorm.rvs(scale=0.2, size=n_draws, random_state=rng)

    def rbf_kernel(x1, x2, ell, eta):
        d2 = (x1[:, None] - x2[None, :]) ** 2
        return eta ** 2 * np.exp(-0.5 * d2 / ell ** 2)

    funcs = np.empty((n_draws, len(x)))
    for i in range(n_draws):
        K = rbf_kernel(x, x, ell_draws[i], eta_draws[i]) + 1e-6 * np.eye(len(x))
        L = np.linalg.cholesky(K)
        f = L @ rng.normal(size=len(x))
        funcs[i] = f + rng.normal(0, sigma_draws[i], len(x))

    env_lo, env_hi = funcs.min(axis=0), funcs.max(axis=0)
    frac_within = float(np.mean((funcs.min(axis=1) <= data_hi) & (funcs.max(axis=1) >= data_lo)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(n_draws):
        ax.plot(x, funcs[i], color=COLOR_OK, alpha=0.15, lw=1)
    ax.fill_between(x, env_lo, env_hi, color=COLOR_OK, alpha=0.08, label="50ドローの包絡線")
    ax.axhspan(data_lo, data_hi, color=COLOR_DIVERGENT, alpha=0.15,
               label=f"実データのスケール[{data_lo}, {data_hi}]")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)(prior predictiveのドロー)")
    ax.set_title(f"GPハイパーパラメータの事前分布から関数を50回ドローし\n実データのスケールを覆っているか目視確認する"
                 f"\n(データ範囲と重なるドロー: {frac_within:.0%})")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gp_hyperparameter_envelope.png")
    plt.close(fig)

    print(f"gp_hyperparameter_envelope.png saved "
          f"(envelope=[{env_lo.min():.2f}, {env_hi.max():.2f}], "
          f"データ範囲=[{data_lo},{data_hi}], 重なるドロー割合={frac_within:.0%})")


if __name__ == "__main__":
    plot_steady_state_prior_center()
    plot_jensen_prior_predictive_catch()
    plot_denominator_variance_explosion()
    plot_runaway_culprit_shifts()
    plot_extreme_vs_proportion()
    plot_gp_hyperparameter_envelope()
    plot_nonuniform_dynamic_range_coverage()
    plot_borrowed_vs_derived_lengthscale()
    plot_reference_prior_vs_data_scale()
