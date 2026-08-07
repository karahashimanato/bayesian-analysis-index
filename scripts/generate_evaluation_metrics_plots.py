"""
tools/evaluation-metrics.md に埋め込む可視化画像を生成するスクリプト。

  1. LOO: elpd_diffの絶対値だけでなく、標準誤差dseと比較して初めて
     「有意な差か」を判断できることを、実際に2種類のモデル比較で示す
  2. Brier Score vs AUC-ROC: 順位付けの良さ(AUC)と確率較正の良さ(Brier)が
     数学的に独立な性質であることを、実際のモデル予測で示す

実行方法:
    source .venv/bin/activate
    python scripts/generate_evaluation_metrics_plots.py

出力先: assets/evaluation-metrics/*.png
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from plot_style import COLOR_ALT, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "evaluation-metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def plot_loo_elpd_diff():
    """elpd_diffの絶対値だけでは判断できず、標準誤差dseと比較して
    初めて有意な差かどうかが分かることを、2種類のモデル比較で示す。"""

    rng = np.random.default_rng(7)
    n = 60
    x = rng.uniform(-2, 2, size=n)
    x_noise = rng.normal(size=n)  # 目的変数と無関係な特徴量
    true_b0, true_b1, sigma_obs = 1.5, 2.0, 1.5
    y = true_b0 + true_b1 * x + rng.normal(0, sigma_obs, size=n)

    def fit(cols):
        with pm.Model() as model:
            b0 = pm.Normal("b0", 0.0, 5.0)
            mu = b0
            for name, col in cols:
                b = pm.Normal(name, 0.0, 5.0)
                mu = mu + b * col
            sigma = pm.HalfNormal("sigma", 2.0)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(
                2000, tune=1500, chains=4, target_accept=0.9, random_seed=0,
                progressbar=False, compute_convergence_checks=False,
            )
            pm.compute_log_likelihood(idata, progressbar=False)
        return idata

    idata_a = fit([("b1", x)])                                  # 正しいモデル
    idata_b = fit([])                                            # 切片のみ(明らかに劣る)
    idata_c = fit([("b1", x), ("b2", x_noise)])                  # 無関係な特徴量を追加

    cmp_ab = az.compare({"A: x予測子あり": idata_a, "B: 切片のみ": idata_b})
    cmp_ac = az.compare({"A: x予測子のみ": idata_a, "C: 無関係な特徴量を追加": idata_c})

    def diff_and_dse(cmp, worse_name):
        row = cmp.loc[worse_name]
        return float(row["elpd_diff"]), float(row["dse"])

    diff_ab, dse_ab = diff_and_dse(cmp_ab, "B: 切片のみ")
    diff_ac, dse_ac = diff_and_dse(cmp_ac, "C: 無関係な特徴量を追加")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparisons = [
        ("A vs B\n(予測子の有無)", diff_ab, dse_ab, COLOR_DIVERGENT if abs(diff_ab) > 2 * dse_ab else COLOR_OK),
        ("A vs C\n(無関係な特徴量の追加)", diff_ac, dse_ac, COLOR_DIVERGENT if abs(diff_ac) > 2 * dse_ac else COLOR_OK),
    ]
    ys = [1, 0]
    for (label, diff, dse, color), yy in zip(comparisons, ys):
        ax.errorbar([diff], [yy], xerr=[2 * dse], fmt="o", color=color, capsize=6, markersize=9, lw=2)
        sig = "|diff| > 2×dse" if abs(diff) > 2 * dse else "|diff| < 2×dse"
        ax.text(diff, yy + 0.18, f"elpd_diff={diff:.2f}, dse={dse:.2f} ({sig})",
                ha="center", fontsize=9, color=color)

    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in comparisons])
    ax.set_xlabel("elpd_diff (誤差棒は±2×dse)")
    ax.set_title("LOOのelpd_diffは絶対値ではなくdseと比較して評価する")
    ax.set_ylim(-0.8, 1.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loo_elpd_diff.png")
    plt.close(fig)

    print(f"loo_elpd_diff.png saved (A-vs-B: elpd_diff={diff_ab:.2f} dse={dse_ab:.2f} "
          f"[{'有意' if abs(diff_ab) > 2*dse_ab else '有意でない'}], "
          f"A-vs-C: elpd_diff={diff_ac:.2f} dse={dse_ac:.2f} "
          f"[{'有意' if abs(diff_ac) > 2*dse_ac else '有意でない'}])")


def plot_brier_auc_independence():
    """順位付けの良さ(AUC-ROC)と確率較正の良さ(Brier Score)が
    独立な性質であることを、実際のベイズロジスティック回帰の予測で示す。"""

    rng = np.random.default_rng(13)
    n = 400
    x = rng.uniform(-3, 3, size=n)
    true_b0, true_b1 = -0.2, 1.0
    p_true = 1 / (1 + np.exp(-(true_b0 + true_b1 * x)))
    y = rng.binomial(1, p_true)

    with pm.Model():
        b0 = pm.Normal("b0", 0.0, 3.0)
        b1 = pm.Normal("b1", 0.0, 3.0)
        p = pm.Deterministic("p", pm.math.invlogit(b0 + b1 * x))
        pm.Bernoulli("y", p=p, observed=y)
        idata = pm.sample(
            1500, tune=1500, chains=4, target_accept=0.9, random_seed=0,
            progressbar=False, compute_convergence_checks=False,
        )

    b0_post = float(idata.posterior["b0"].values.mean())
    b1_post = float(idata.posterior["b1"].values.mean())
    logit_correct = b0_post + b1_post * x

    p_correct = 1 / (1 + np.exp(-logit_correct))                # 較正済みの基準モデル
    p_overconfident = 1 / (1 + np.exp(-3.0 * logit_correct))  # 順位は同じ、確信度だけ3倍
    p_constant = np.full(n, y.mean())                          # 全員に基準率を予測、順位付け能力ゼロ

    def auc(p_pred, y_true):
        order = np.argsort(p_pred)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        n_pos, n_neg = y_true.sum(), n - y_true.sum()
        return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    def brier(p_pred, y_true):
        return float(np.mean((p_pred - y_true) ** 2))

    auc_correct, brier_correct = auc(p_correct, y), brier(p_correct, y)
    auc_over, brier_over = auc(p_overconfident, y), brier(p_overconfident, y)
    auc_const, brier_const = auc(p_constant, y), brier(p_constant, y)

    def reliability(p_pred, y_true, n_bins=8):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p_pred, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        mean_pred, mean_obs = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() > 0:
                mean_pred.append(p_pred[mask].mean())
                mean_obs.append(y_true[mask].mean())
        return np.array(mean_pred), np.array(mean_obs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for p_pred, color, label in [
        (p_correct, COLOR_OK, f"較正済みモデル(AUC={auc_correct:.3f}, Brier={brier_correct:.3f})"),
        (p_overconfident, COLOR_DIVERGENT, f"過信モデル(順位は同じ, AUC={auc_over:.3f}, Brier={brier_over:.3f})"),
        (p_constant, COLOR_ALT, f"定数モデル(基準率のみ, AUC={auc_const:.3f}, Brier={brier_const:.3f})"),
    ]:
        mp, mo = reliability(p_pred, y)
        axes[0].plot(mp, mo, "o-", color=color, label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="理想的な較正")
    axes[0].set_xlabel("予測確率(ビン平均)")
    axes[0].set_ylabel("実際の陽性率(ビン平均)")
    axes[0].set_title("較正(Brier Scoreが見ている軸)")
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    metrics = ["AUC-ROC\n(高いほど良い)", "Brier Score\n(低いほど良い)"]
    x_pos = np.arange(2)
    width = 0.25
    axes[1].bar(x_pos - width, [auc_correct, brier_correct], width, color=COLOR_OK, label="較正済みモデル")
    axes[1].bar(x_pos, [auc_over, brier_over], width, color=COLOR_DIVERGENT, label="過信モデル")
    axes[1].bar(x_pos + width, [auc_const, brier_const], width, color=COLOR_ALT, label="定数モデル")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics)
    axes[1].axhline(0.5, color="gray", lw=0.8, ls=":")
    axes[1].set_title("較正済み vs 過信: AUC同じでBrierだけ悪化\n較正済み vs 定数: AUCだけ悪化")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("AUC-ROCとBrier Scoreは独立な性質を測る", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "brier_auc_independence.png")
    plt.close(fig)

    print(f"brier_auc_independence.png saved (較正済み: AUC={auc_correct:.3f} Brier={brier_correct:.3f}, "
          f"過信: AUC={auc_over:.3f} Brier={brier_over:.3f}, "
          f"定数: AUC={auc_const:.3f} Brier={brier_const:.3f})")


if __name__ == "__main__":
    plot_loo_elpd_diff()
    plot_brier_auc_independence()
