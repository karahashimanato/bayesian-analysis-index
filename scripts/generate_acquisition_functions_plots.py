"""
tools/acquisition-functions.md に埋め込む可視化画像を生成するスクリプト。

局所最適(x~2、微小な凹凸あり)と大域最適(x~7)を持つ1次元関数に対し、
4つの獲得関数(PI/EI/UCB/GP-TS)で逐次ベイズ最適化(10反復)を実行し、
(1) 同一のGP事後分布(初期3点のみ)上で各獲得関数が次にどこを提案するか、
(2) 反復ごとのregret(真の大域最適との差)の収束の違い
を比較する。GP回帰は固定ハイパーパラメータのRBFカーネルによる閉形式の
事後分布(Cholesky分解)を直接計算し、MCMCは使わない。

実行方法:
    source .venv/bin/activate
    python scripts/generate_acquisition_functions_plots.py

出力先: assets/acquisition-functions/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "acquisition-functions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def true_f(x):
    """局所最適(x~2、高さ0.8、微小な凹凸あり)と大域最適(x~7、高さ1.0)を持つ関数。"""
    local = 0.8 * np.exp(-0.15 * (x - 2) ** 2) * (1 + 0.05 * np.sin(15 * x))
    global_peak = 1.0 * np.exp(-0.5 * (x - 7) ** 2)
    return local + global_peak


X_GRID = np.linspace(0, 10, 400)
GLOBAL_MAX = float(true_f(X_GRID).max())


def rbf_kernel(x1, x2, ell=1.0, var=1.0):
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return var * np.exp(-0.5 * d2 / ell ** 2)


def gp_posterior(x_obs, y_obs, x_star, ell=1.0, var=1.0, noise=1e-6):
    K = rbf_kernel(x_obs, x_obs, ell, var) + noise * np.eye(len(x_obs))
    K_star = rbf_kernel(x_obs, x_star, ell, var)
    K_star_star = rbf_kernel(x_star, x_star, ell, var)
    K_inv = np.linalg.inv(K)
    mu = K_star.T @ K_inv @ y_obs
    cov = K_star_star - K_star.T @ K_inv @ K_star
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return mu, sigma, cov


def acq_pi(mu, sigma, f_best, xi=0.001):
    z = (mu - f_best - xi) / sigma
    return norm.cdf(z)


def acq_ei(mu, sigma, f_best, xi=0.001):
    z = (mu - f_best - xi) / sigma
    return (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def acq_ucb(mu, sigma, kappa=2.0):
    return mu + kappa * sigma


def run_bo(acq_type, x_init, n_iter=10, seed=0):
    rng = np.random.default_rng(seed)
    x_obs, y_obs = x_init.copy(), true_f(x_init)
    regrets = []
    history = [(x_obs.copy(), y_obs.copy())]
    for _ in range(n_iter):
        mu, sigma, cov = gp_posterior(x_obs, y_obs, X_GRID)
        f_best = y_obs.max()
        if acq_type == "PI":
            acq = acq_pi(mu, sigma, f_best)
        elif acq_type == "EI":
            acq = acq_ei(mu, sigma, f_best)
        elif acq_type == "UCB":
            acq = acq_ucb(mu, sigma)
        elif acq_type == "GP-TS":
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(X_GRID)))
            acq = mu + L @ rng.normal(size=len(X_GRID))
        next_x = X_GRID[np.argmax(acq)]
        x_obs = np.append(x_obs, next_x)
        y_obs = np.append(y_obs, true_f(next_x))
        regrets.append(GLOBAL_MAX - y_obs.max())
        history.append((x_obs.copy(), y_obs.copy()))
    return regrets, history


def plot_acquisition_function_comparison():
    x_init = np.array([1.5, 2.5, 3.5])  # 局所最適の近くから開始(探索が必要な設定)

    # ---- 反復ごとのregret(10反復、4手法)。EIの軌跡を「3反復目時点」の
    # スナップショットとして獲得関数landscapeの比較にも流用する ----
    results = {m: run_bo(m, x_init, n_iter=10, seed=1) for m in ["PI", "EI", "UCB", "GP-TS"]}
    regrets_by_method = {m: r for m, (r, _) in results.items()}
    x_snap, y_snap = results["EI"][1][2]  # EIの軌跡、2反復後(=3反復目の直前)の状態

    # ---- 同一のGP事後分布(3反復目時点)上での各獲得関数の landscape ----
    mu, sigma, cov = gp_posterior(x_snap, y_snap, X_GRID)
    f_best = y_snap.max()
    pi_vals = acq_pi(mu, sigma, f_best)
    ei_vals = acq_ei(mu, sigma, f_best)
    ucb_vals = acq_ucb(mu, sigma)

    def normalize(v):
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    pi_n, ei_n, ucb_n = normalize(pi_vals), normalize(ei_vals), normalize(ucb_vals)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.3, 1])

    ax_obj = fig.add_subplot(gs[0, 0])
    ax_obj.plot(X_GRID, true_f(X_GRID), color="black", lw=1.5, label="真の目的関数")
    ax_obj.fill_between(X_GRID, mu - 1.96 * sigma, mu + 1.96 * sigma, color=COLOR_OK, alpha=0.2)
    ax_obj.plot(X_GRID, mu, color=COLOR_OK, lw=1.5, label="GP事後平均(3反復目時点)")
    ax_obj.scatter(x_snap, y_snap, color="black", s=40, zorder=5, label="観測点(初期3+EIの2反復)")
    ax_obj.axvline(X_GRID[np.argmax(true_f(X_GRID))], color=COLOR_DIVERGENT, lw=1, ls=":",
                    label="真の大域最適")
    ax_obj.set_ylabel("f(x)")
    ax_obj.set_title("目的関数とGP事後分布(3反復目時点)")
    ax_obj.legend(fontsize=8, loc="upper left")

    ax_acq = fig.add_subplot(gs[1, 0], sharex=ax_obj)
    for label, vals, color in [("PI", pi_n, COLOR_CHAIN[1]), ("EI", ei_n, COLOR_CHAIN[0]),
                                ("UCB", ucb_n, COLOR_CHAIN[2])]:
        ax_acq.plot(X_GRID, vals, color=color, lw=1.5, label=label)
        ax_acq.axvline(X_GRID[np.argmax(vals)], color=color, lw=1, ls="--", alpha=0.7)
    ax_acq.set_xlabel("x")
    ax_acq.set_ylabel("獲得関数(正規化)")
    ax_acq.set_title("同一のGP事後分布上での獲得関数の違い(縦線=各手法の次の提案点)")
    ax_acq.legend(fontsize=8.5, loc="upper left")

    ax_regret = fig.add_subplot(gs[:, 1])
    colors = {"PI": COLOR_CHAIN[1], "EI": COLOR_CHAIN[0], "UCB": COLOR_CHAIN[2], "GP-TS": COLOR_ALT}
    for method, regrets in regrets_by_method.items():
        ax_regret.plot(np.arange(1, 11), np.array(regrets) + 1e-4, "o-", color=colors[method],
                        label=f"{method}(10反復目={regrets[-1]:.4f})")
    ax_regret.set_yscale("log")
    ax_regret.set_xlabel("反復回数")
    ax_regret.set_ylabel("regret(真の大域最適との差、log軸)")
    ax_regret.set_title("反復ごとのregretの収束")
    ax_regret.legend(fontsize=9)

    fig.suptitle("獲得関数(PI/EI/UCB/GP-TS)は同じGP事後分布でも異なる次の一手を提案する", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "acquisition_function_comparison.png")
    plt.close(fig)

    print("acquisition_function_comparison.png saved (final regret: " +
          ", ".join(f"{m}={r[-1]:.4f}" for m, r in regrets_by_method.items()) + ")")


if __name__ == "__main__":
    plot_acquisition_function_comparison()
