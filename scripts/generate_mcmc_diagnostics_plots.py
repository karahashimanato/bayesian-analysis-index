"""
tools/mcmc-diagnostics.md に埋め込む可視化画像を生成するスクリプト。

変化点(changepoint)モデルを実際にPyMCでサンプリングし、離散変数(変化点位置tau、
Compound StepでMetropolis法が使われる)のESSが連続変数(Poisson率)より
低くなりやすいことを描画する。

また、Neal's funnel(急峻に曲がった事後分布)をtarget_accept=0.8/0.99の
2通りでサンプリングし、divergence数とESSがどうトレードオフするかを描画する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_mcmc_diagnostics_plots.py

出力先: assets/mcmc-diagnostics/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "mcmc-diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_discrete_ess_gap():
    """変化点モデル(tau=離散, lambda1/lambda2=連続)で、同じdraws数でも
    離散変数のESSが連続変数より1桁近く低くなることを示す。"""

    rng = np.random.default_rng(2)
    T = 100
    true_tau = 45
    true_lambda1, true_lambda2 = 3.0, 8.0
    counts = np.concatenate([
        rng.poisson(true_lambda1, size=true_tau),
        rng.poisson(true_lambda2, size=T - true_tau),
    ])
    idx = np.arange(T)

    with pm.Model():
        tau = pm.DiscreteUniform("tau", lower=0, upper=T - 1)
        lambda1 = pm.Exponential("lambda1", 1.0)
        lambda2 = pm.Exponential("lambda2", 1.0)
        rate = pt.switch(idx < tau, lambda1, lambda2)
        pm.Poisson("obs", rate, observed=counts)
        idata = pm.sample(4000, tune=2000, chains=4, random_seed=0,
                           progressbar=False, compute_convergence_checks=False)

    ess = az.ess(idata, var_names=["tau", "lambda1", "lambda2"])
    ess_tau = float(ess["tau"].values)
    ess_l1 = float(ess["lambda1"].values)
    ess_l2 = float(ess["lambda2"].values)
    total_draws = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    names = ["tau\n(離散, Metropolis)", "lambda1\n(連続, NUTS)", "lambda2\n(連続, NUTS)"]
    values = [ess_tau, ess_l1, ess_l2]
    colors = [COLOR_ALT, COLOR_OK, COLOR_OK]
    bars = ax.bar(names, values, color=colors, alpha=0.85)
    ax.axhline(total_draws, color="black", linestyle=":", linewidth=1,
               label=f"総draws数 = {total_draws}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + total_draws * 0.02,
                f"{v:.0f}\n({v/total_draws*100:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("ESS (ess_bulk)")
    ax.set_title(f"変化点モデル: 離散変数(tau)のESSは連続変数の約"
                 f"{ess_l1/ess_tau:.0f}分の1\n(PyMCがCompound StepでtauにMetropolis法を使うため)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "discrete_ess_gap.png")
    plt.close(fig)
    print(f"discrete_ess_gap.png saved (ESS tau={ess_tau:.0f}, lambda1={ess_l1:.0f}, "
          f"lambda2={ess_l2:.0f}, total_draws={total_draws})")


def plot_target_accept_tradeoff():
    """Neal's funnel(急峻に曲がった事後分布)をtarget_accept=0.8/0.99の
    2通りでサンプリングし、divergence数の減少とESSの変化を比較する。"""

    D = 9

    def build_model():
        with pm.Model() as model:
            v = pm.Normal("v", 0, 3)
            pm.Normal("x", 0, pt.exp(v / 2), shape=D)
        return model

    results = {}
    for ta in [0.8, 0.99]:
        model = build_model()
        with model:
            idata = pm.sample(2000, tune=1500, chains=4, target_accept=ta,
                               random_seed=5, progressbar=False,
                               compute_convergence_checks=False)
        n_div = int(idata.sample_stats["diverging"].sum())
        ess_v = float(az.ess(idata, var_names=["v"])["v"].values)
        results[ta] = (n_div, ess_v)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    tas = [0.8, 0.99]
    divs = [results[ta][0] for ta in tas]
    esss = [results[ta][1] for ta in tas]
    colors = [COLOR_ALT, COLOR_OK]

    axes[0].bar([str(ta) for ta in tas], divs, color=colors, alpha=0.85)
    axes[0].set_xlabel("target_accept")
    axes[0].set_ylabel("divergence数(全chain合計)")
    axes[0].set_title("divergence数")
    for i, d in enumerate(divs):
        axes[0].text(i, d, f"{d}", ha="center", va="bottom", fontsize=10)

    axes[1].bar([str(ta) for ta in tas], esss, color=colors, alpha=0.85)
    axes[1].set_xlabel("target_accept")
    axes[1].set_ylabel("ESS (v, ess_bulk)")
    axes[1].set_title("ESS")
    for i, e in enumerate(esss):
        axes[1].text(i, e, f"{e:.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Neal's funnelでtarget_accept=0.8→0.99にすると\n"
                  f"divergenceは{divs[0]}→{divs[1]}に減るが、ESSは{esss[0]:.0f}→{esss[1]:.0f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_accept_tradeoff.png")
    plt.close(fig)
    print(f"target_accept_tradeoff.png saved "
          f"(0.8: divergence={divs[0]}, ess={esss[0]:.0f} / "
          f"0.99: divergence={divs[1]}, ess={esss[1]:.0f})")


def plot_rhat_within_between_variance():
    """r_hatの定義そのもの(チェーン内分散Wとチェーン間分散Bの比)を、
    健全なchain群と不健全なchain群それぞれで実際に計算して示す。"""

    def sample_healthy():
        with pm.Model():
            pm.Normal("mu", 0, 5)
            idata = pm.sample(1000, tune=1000, chains=4, random_seed=0,
                               progressbar=False, compute_convergence_checks=False)
        return idata

    def sample_unhealthy():
        with pm.Model():
            pm.Normal("mu", 0, 5)
            step = pm.Metropolis(scaling=0.02)
            idata = pm.sample(200, tune=50, chains=4, step=step, random_seed=0,
                               progressbar=False, compute_convergence_checks=False,
                               initvals=[{"mu": v} for v in [-8, -3, 3, 8]])
        return idata

    idata_h = sample_healthy()
    idata_u = sample_unhealthy()

    results = {}
    for name, idata in [("healthy", idata_h), ("unhealthy", idata_u)]:
        mu = idata.posterior["mu"].values  # (chain, draw)
        n_draws = mu.shape[1]
        chain_means = mu.mean(axis=1)
        W = mu.var(axis=1, ddof=1).mean()
        B = n_draws * chain_means.var(ddof=1)
        rhat_official = float(az.rhat(idata, var_names=["mu"])["mu"].values)
        results[name] = dict(mu=mu, W=W, B=B, rhat=rhat_official)
        print(f"{name}: W={W:.3f}, B={B:.3f}, r_hat(arviz, rank normalized)={rhat_official:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, name, title in [
        (axes[0], "healthy", "健全: chain間で似た値を探索\n(B ≈ W)"),
        (axes[1], "unhealthy", "不健全: chainごとに別々の値に固定\n(B >> W)"),
    ]:
        r = results[name]
        for c in range(r["mu"].shape[0]):
            ax.plot(r["mu"][c], color=COLOR_CHAIN[c % len(COLOR_CHAIN)], linewidth=0.9,
                    alpha=0.85, label=f"chain {c}")
        ax.set_xlabel("draw")
        ax.set_title(f"{title}\nW={r['W']:.2f}, B={r['B']:.2f}, r_hat={r['rhat']:.2f}")
        ax.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9)

    axes[0].set_ylabel("mu")
    fig.suptitle("r_hat: チェーン内分散(W)とチェーン間分散(B)の比で収束を測る", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "rhat_within_between_variance.png")
    plt.close(fig)
    print(f"rhat_within_between_variance.png saved "
          f"(healthy r_hat={results['healthy']['rhat']:.2f}, "
          f"unhealthy r_hat={results['unhealthy']['rhat']:.2f})")


def plot_divergence_leapfrog_energy():
    """HMC/NUTSが使うleapfrog積分器で、ステップサイズがターゲット分布の
    曲率に対して大きすぎるとエネルギー保存則から急激に外れる
    (=divergence)ことを、単純な調和振動子ポテンシャルで直接示す。"""

    sigma = 1.0  # U(x) = x^2 / (2*sigma^2) の調和振動子(安定限界は eps=2*sigma)

    def U(x):
        return 0.5 * x**2 / sigma**2

    def grad_U(x):
        return x / sigma**2

    def leapfrog(x0, p0, eps, n_steps):
        x, p = x0, p0
        xs, ps = [x], [p]
        for _ in range(n_steps):
            p = p - 0.5 * eps * grad_U(x)
            x = x + eps * p
            p = p - 0.5 * eps * grad_U(x)
            xs.append(x)
            ps.append(p)
        return np.array(xs), np.array(ps)

    x0, p0 = 0.0, 1.0
    H0 = U(x0) + 0.5 * p0**2
    eps_values = [0.3, 1.8, 2.05]
    labels = [f"eps={e}(安定域)" if e < 2.0 else f"eps={e}(不安定域、eps>2σ)" for e in eps_values]
    colors = [COLOR_OK, COLOR_ALT, COLOR_DIVERGENT]
    max_energy_error = 1000.0  # PyMC/Stanのdivergence判定デフォルト閾値

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    for eps, label, color in zip(eps_values, labels, colors):
        xs, ps = leapfrog(x0, p0, eps, 40)
        Hs = U(xs) + 0.5 * ps**2
        energy_error = np.abs(Hs - H0)
        ax.plot(energy_error + 1e-12, color=color, linewidth=1.8, label=label)
    ax.axhline(max_energy_error, color="black", linestyle=":", linewidth=1,
               label=f"divergence判定閾値(|ΔH|>{max_energy_error:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("leapfrogステップ数")
    ax.set_ylabel("エネルギー誤差 |H(t) - H(0)|(対数軸)")
    ax.set_title("ステップサイズが安定限界(eps=2σ)を超えると\nエネルギー誤差が指数的に爆発する")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # 安定/発散でスケールが桁違いになるため、位相空間の軌道は別軸で並べる
    xs_stable, ps_stable = leapfrog(x0, p0, 0.3, 40)
    n_div_steps = 12  # 発散側は序盤だけ見せないとスケールが暴走して軌道が潰れる
    xs_div, ps_div = leapfrog(x0, p0, 2.05, n_div_steps)

    ax = axes[1]
    ax.plot(xs_stable, ps_stable, color=COLOR_OK, linewidth=1, marker="o", markersize=2.5)
    ax.set_xlabel("x(位置)")
    ax.set_ylabel("p(運動量)")
    ax.set_title("eps=0.3(安定)\n位相空間で閉軌道を描く")

    ax = axes[2]
    ax.plot(xs_div, ps_div, color=COLOR_DIVERGENT, linewidth=1, marker="o", markersize=2.5)
    ax.set_xlabel("x(位置)")
    ax.set_title(f"eps=2.05(divergence)\n最初の{n_div_steps}stepで既に外側へ暴走")

    fig.suptitle("Divergence: leapfrog積分のエネルギー保存則からの逸脱", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "divergence_leapfrog_energy.png")
    plt.close(fig)
    print("divergence_leapfrog_energy.png saved "
          f"(eps=0.3 max|dH|={np.max(np.abs(U(leapfrog(x0,p0,0.3,40)[0])+0.5*leapfrog(x0,p0,0.3,40)[1]**2-H0)):.4g}, "
          f"eps=2.05 max|dH|={np.max(np.abs(U(leapfrog(x0,p0,2.05,40)[0])+0.5*leapfrog(x0,p0,2.05,40)[1]**2-H0)):.4g})")


if __name__ == "__main__":
    plot_discrete_ess_gap()
    plot_target_accept_tradeoff()
    plot_rhat_within_between_variance()
    plot_divergence_leapfrog_energy()
