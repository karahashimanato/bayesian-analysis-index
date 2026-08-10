"""
tools/evaluation-metrics.md に埋め込む可視化画像を生成するスクリプト。

  1. LOO: elpd_diffの絶対値だけでなく、標準誤差dseと比較して初めて
     「有意な差か」を判断できることを、実際に2種類のモデル比較で示す
  2. Brier Score vs AUC-ROC: 順位付けの良さ(AUC)と確率較正の良さ(Brier)が
     数学的に独立な性質であることを、実際のモデル予測で示す
  3. Brier Score(IPCW): 打ち切りを無視した素朴な計算がバイアスを持ち、
     IPCW補正で打ち切りなしのオラクル値に近づくことを示す
  4. DM: 報酬モデルの誤設定度合いを連続的に変えたとき、バイアスが
     単調に拡大することを示す
  5. DR: 傾向スコア・報酬モデルの正誤2x2で、どちらか一方が正しければ
     不偏という「二重にロバスト」な性質を示す
  6. SNIPS: 傾向スコアが極端な場合にIPSの分散が爆発し、自己正規化で
     大きく抑えられることを示す
  7. SNDR: 一様な傾向スコアの特殊構造下でSNIPS・SNDR・単純平均が
     報酬モデルの正誤によらず近似的に一致することを示す
  8. Regret: 獲得関数の最大化に使うグリッドの分解能がボトルネックだと、
     regretが理論上の0まで収束しないことを示す

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

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

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


def plot_ope_bias_variance_tradeoff():
    """IPS/SNIPS/DM/DRの4つのOPE推定量について、ログデータからのシミュレーション
    を繰り返し、真の方策価値のまわりでのバイアス・分散のトレードオフを比較する。
    DMの報酬モデルは意図的に誤設定(非単調な真の報酬をarm indexへの線形回帰で
    近似)し、DMがバイアスを持つ一方、DRがそのバイアスを補正できることを示す。"""

    rng = np.random.default_rng(11)
    n_arms = 4
    p_true = np.array([0.10, 0.15, 0.45, 0.20])  # 非単調(arm2が突出)
    pi_b = np.full(n_arms, 0.25)
    pi_e = np.array([0.05, 0.10, 0.70, 0.15])
    v_true = float(np.sum(pi_e * p_true))

    n_reps = 60
    n_log = 400
    arm_idx_all = np.arange(n_arms)

    v_ips = np.zeros(n_reps)
    v_snips = np.zeros(n_reps)
    v_dm = np.zeros(n_reps)
    v_dr = np.zeros(n_reps)

    for k in range(n_reps):
        rng_k = np.random.default_rng(1000 + k)
        actions = rng_k.choice(n_arms, size=n_log, p=pi_b)
        rewards = rng_k.binomial(1, p_true[actions]).astype(float)

        w = pi_e[actions] / pi_b[actions]
        v_ips[k] = np.mean(w * rewards)
        v_snips[k] = np.sum(w * rewards) / np.sum(w)

        with pm.Model():
            b0 = pm.Normal("b0", 0.3, 0.5)
            b1 = pm.Normal("b1", 0, 0.5)
            sigma = pm.HalfNormal("sigma", 0.5)
            mu = b0 + b1 * actions
            pm.Normal("y", mu=mu, sigma=sigma, observed=rewards)
            idata = pm.sample(500, tune=500, chains=2, target_accept=0.9,
                               random_seed=k, progressbar=False,
                               compute_convergence_checks=False)
        b0_hat = float(idata.posterior["b0"].mean())
        b1_hat = float(idata.posterior["b1"].mean())
        r_hat_per_arm = b0_hat + b1_hat * arm_idx_all
        r_hat_actions = b0_hat + b1_hat * actions

        v_dm[k] = np.sum(pi_e * r_hat_per_arm)
        v_dr[k] = v_dm[k] + np.mean(w * (rewards - r_hat_actions))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    estimators = [("IPS", v_ips, COLOR_CHAIN[0]), ("SNIPS", v_snips, COLOR_CHAIN[1]),
                  ("DM\n(誤設定あり)", v_dm, COLOR_DIVERGENT), ("DR", v_dr, COLOR_OK)]
    positions = np.arange(len(estimators))
    bp = axes[0].boxplot([e[1] for e in estimators], positions=positions, widths=0.6,
                          patch_artist=True, showmeans=True)
    for patch, (_, _, color) in zip(bp["boxes"], estimators):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0].axhline(v_true, color="black", lw=1.3, ls="--", label=f"真の方策価値={v_true:.3f}")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([e[0] for e in estimators], fontsize=9)
    axes[0].set_ylabel("推定された方策価値")
    axes[0].set_title(f"{n_reps}回のログ再サンプリングでの推定値分布")
    axes[0].legend(fontsize=8)

    bias = [np.mean(e[1]) - v_true for e in estimators]
    std = [np.std(e[1]) for e in estimators]
    axes[1].bar(positions - 0.15, bias, width=0.3, color=[e[2] for e in estimators],
                alpha=0.9, label="バイアス(平均-真値)")
    ax2 = axes[1].twinx()
    ax2.bar(positions + 0.15, std, width=0.3, color=[e[2] for e in estimators],
            alpha=0.4, hatch="//", label="標準偏差")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([e[0] for e in estimators], fontsize=9)
    axes[1].set_ylabel("バイアス(平均-真値)")
    ax2.set_ylabel("標準偏差")
    axes[1].set_title("バイアス(塗り)と分散(斜線)のトレードオフ")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("OPE推定量のバイアス・分散トレードオフ(IPS/SNIPS/DM/DR)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "ope_bias_variance_tradeoff.png")
    plt.close(fig)

    print(f"ope_bias_variance_tradeoff.png saved (真値={v_true:.3f})")
    for name, v, _ in estimators:
        print(f"  {name.strip()}: mean={v.mean():.4f} bias={v.mean()-v_true:+.4f} std={v.std():.4f}")


def _km_survival(times, event_flags):
    """Kaplan-Meier推定量。ユニークなイベント時刻とそこでの生存確率を返す。"""
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = event_flags[order]
    unique_event_times = np.unique(t_sorted[e_sorted == 1])
    s = 1.0
    surv = []
    for ut in unique_event_times:
        n_risk = np.sum(t_sorted >= ut)
        d = np.sum((t_sorted == ut) & (e_sorted == 1))
        s *= (1 - d / n_risk) if n_risk > 0 else 1.0
        surv.append(s)
    return unique_event_times, np.array(surv)


def _km_eval(t_query, unique_times, surv):
    if len(unique_times) == 0 or t_query < unique_times[0]:
        return 1.0
    idx = np.searchsorted(unique_times, t_query, side="right") - 1
    return surv[idx]


def plot_brier_ipcw_censoring_correction():
    """打ち切りのある生存時間データで、打ち切りを無視した素朴なBrier Scoreが
    バイアスを持つ一方、IPCW(逆確率打ち切り重み付け)補正版が、打ち切りなしの
    オラクル値に近づくことを、複数の評価時点にわたって示す。"""

    rng = np.random.default_rng(23)
    n = 3000
    x = rng.uniform(-1, 1, size=n)
    rate = 0.5 * np.exp(0.3 + 0.7 * x)  # 真のハザード率(既知の指数分布モデルとして予測に使う)
    t_true = rng.exponential(1.0 / rate)  # 真のイベント時刻(打ち切りなしの世界)

    cmax = 3.0
    c = rng.uniform(0, cmax, size=n)  # 打ち切り時刻
    t_obs = np.minimum(t_true, c)
    delta = (t_true <= c).astype(int)  # 1=イベント観測, 0=打ち切り

    eval_times = np.linspace(0.3, 2.5, 12)

    def s_hat(t0):
        return np.exp(-rate * t0)  # 予測生存確率(真のハザードを知るモデル)

    def brier_oracle(t0):
        y = (t_true > t0).astype(float)
        return float(np.mean((s_hat(t0) - y) ** 2))

    def brier_naive(t0):
        # 素朴な誤り: 打ち切られた対象を無条件に「生存」とみなす
        y_naive = (t_obs >= t0).astype(float)
        return float(np.mean((s_hat(t0) - y_naive) ** 2))

    # 打ち切り分布G(t)のKM推定(打ち切りを「イベント」とみなして反転)
    g_times, g_surv = _km_survival(t_obs, 1 - delta)

    def brier_ipcw(t0):
        s_pred = s_hat(t0)
        total = 0.0
        for i in range(n):
            g_ti = max(_km_eval(t_obs[i], g_times, g_surv), 1e-6)
            g_t0 = max(_km_eval(t0, g_times, g_surv), 1e-6)
            if t_obs[i] <= t0 and delta[i] == 1:
                total += (0.0 - s_pred[i]) ** 2 / g_ti
            elif t_obs[i] > t0:
                total += (1.0 - s_pred[i]) ** 2 / g_t0
        return total / n

    bs_oracle = np.array([brier_oracle(t0) for t0 in eval_times])
    bs_naive = np.array([brier_naive(t0) for t0 in eval_times])
    bs_ipcw = np.array([brier_ipcw(t0) for t0 in eval_times])

    censored_frac = float(np.mean(delta == 0))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(eval_times, bs_oracle, "o-", color="black", lw=1.5, label="オラクル(打ち切りなしの真値)")
    ax.plot(eval_times, bs_naive, "s-", color=COLOR_DIVERGENT, lw=1.5,
             label="素朴な計算(打ち切りを『生存』扱い)")
    ax.plot(eval_times, bs_ipcw, "^-", color=COLOR_OK, lw=1.5, label="IPCW補正")
    ax.set_xlabel("評価時点 t")
    ax.set_ylabel("Brier Score")
    ax.set_title(f"打ち切り率{censored_frac:.1%}のデータでのBrier Score: "
                 "IPCW補正がオラクルに近づく")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "brier_ipcw_censoring_correction.png")
    plt.close(fig)

    mean_abs_err_naive = float(np.mean(np.abs(bs_naive - bs_oracle)))
    mean_abs_err_ipcw = float(np.mean(np.abs(bs_ipcw - bs_oracle)))
    print(f"brier_ipcw_censoring_correction.png saved (打ち切り率={censored_frac:.3f}, "
          f"平均絶対誤差: 素朴={mean_abs_err_naive:.4f} IPCW={mean_abs_err_ipcw:.4f})")


def plot_dm_misspecification_bias():
    """DMの報酬モデルの誤設定度合い(非単調な真の報酬から線形な報酬へ寄せる比率)
    を連続的に変えたとき、DMのバイアスが単調に拡大することを示す。"""

    rng = np.random.default_rng(29)
    n_arms = 4
    arm_idx = np.arange(n_arms)
    p_nonlinear = np.array([0.10, 0.15, 0.45, 0.20])  # 真の報酬(非単調, arm2が突出)
    p_linear = np.linspace(0.10, 0.35, n_arms)         # 線形回帰で表現可能な報酬
    pi_b = np.full(n_arms, 0.25)
    pi_e = np.array([0.05, 0.10, 0.70, 0.15])

    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # 0=線形(正しい設定) 1=非単調(誤設定)
    n_log = 2000
    n_reps = 200

    bias_mean = np.zeros(len(alphas))
    bias_std = np.zeros(len(alphas))
    v_true_list = np.zeros(len(alphas))

    for j, alpha in enumerate(alphas):
        p_alpha = (1 - alpha) * p_linear + alpha * p_nonlinear
        v_true = float(np.sum(pi_e * p_alpha))
        v_true_list[j] = v_true
        v_dm = np.zeros(n_reps)
        for k in range(n_reps):
            rng_k = np.random.default_rng(2000 + j * 1000 + k)
            actions = rng_k.choice(n_arms, size=n_log, p=pi_b)
            rewards = rng_k.binomial(1, p_alpha[actions]).astype(float)
            # 線形回帰(OLS)の報酬モデル r_hat(a) = b0 + b1*a
            design = np.column_stack([np.ones(n_log), actions.astype(float)])
            b_hat, *_ = np.linalg.lstsq(design, rewards, rcond=None)
            r_hat_per_arm = b_hat[0] + b_hat[1] * arm_idx
            v_dm[k] = np.sum(pi_e * r_hat_per_arm)
        bias_mean[j] = float(np.mean(v_dm) - v_true)
        bias_std[j] = float(np.std(v_dm))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(arm_idx, p_nonlinear, "o-", color=COLOR_DIVERGENT, label="真の報酬(alpha=1, 非単調)")
    axes[0].plot(arm_idx, p_linear, "s--", color=COLOR_OK, label="線形回帰で表現可能(alpha=0)")
    axes[0].set_xlabel("腕(arm) index")
    axes[0].set_ylabel("報酬確率")
    axes[0].set_title("報酬モデルの表現力と真の報酬の形")
    axes[0].legend(fontsize=9)

    axes[1].errorbar(alphas, bias_mean, yerr=bias_std, fmt="o-", color=COLOR_CHAIN[0],
                      capsize=4, lw=1.5)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_xlabel("誤設定度合い alpha(0=正しい設定, 1=完全に誤設定)")
    axes[1].set_ylabel("DMのバイアス(平均推定値-真値)")
    axes[1].set_title(f"{n_reps}回のログ再サンプリングでのDMバイアス\n"
                       "(誤差棒は標準偏差)")

    fig.suptitle("DMのバイアスは報酬モデルの誤設定度合いに単調に比例する", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "dm_misspecification_bias.png")
    plt.close(fig)

    print(f"dm_misspecification_bias.png saved (alpha={list(alphas)}, "
          f"bias={[round(b, 4) for b in bias_mean]})")


def plot_dr_double_robustness_grid():
    """傾向スコア(既知/誤り)と報酬モデル(正しい/誤設定)の2x2の組み合わせで、
    DRがそのどちらか一方さえ正しければ不偏になる「二重のロバスト性」を、
    IPS単体・DM単体との比較で示す。"""

    rng = np.random.default_rng(31)
    n_arms = 4
    arm_idx = np.arange(n_arms)
    p_true = np.array([0.10, 0.15, 0.45, 0.20])  # 真の報酬(非単調)
    pi_b_true = np.array([0.40, 0.30, 0.20, 0.10])  # 真のログ収集方策(偏りあり)
    pi_b_wrong = np.full(n_arms, 0.25)               # 分析者が誤って信じる傾向スコア(一様)
    pi_e = np.array([0.05, 0.10, 0.70, 0.15])
    v_true = float(np.sum(pi_e * p_true))

    n_log = 1500
    n_reps = 200

    combos = [
        ("傾向スコア正 + 報酬正", True, True),
        ("傾向スコア正 + 報酬誤設定", True, False),
        ("傾向スコア誤 + 報酬正", False, True),
        ("傾向スコア誤 + 報酬誤設定", False, False),
    ]

    dr_bias = np.zeros(len(combos))
    ips_bias = np.zeros(len(combos))
    dm_bias = np.zeros(len(combos))

    for j, (_, prop_ok, reward_ok) in enumerate(combos):
        v_dr = np.zeros(n_reps)
        v_ips = np.zeros(n_reps)
        v_dm = np.zeros(n_reps)
        for k in range(n_reps):
            rng_k = np.random.default_rng(3000 + j * 1000 + k)
            actions = rng_k.choice(n_arms, size=n_log, p=pi_b_true)
            rewards = rng_k.binomial(1, p_true[actions]).astype(float)

            pi_b_used = pi_b_true if prop_ok else pi_b_wrong
            w = pi_e[actions] / pi_b_used[actions]
            v_ips[k] = np.mean(w * rewards)

            if reward_ok:
                # 飽和モデル(腕ごとの標本平均) = 関数形の誤りがない報酬モデル
                r_hat_per_arm = np.array([
                    rewards[actions == a].mean() if np.any(actions == a) else rewards.mean()
                    for a in arm_idx
                ])
            else:
                design = np.column_stack([np.ones(n_log), actions.astype(float)])
                b_hat, *_ = np.linalg.lstsq(design, rewards, rcond=None)
                r_hat_per_arm = b_hat[0] + b_hat[1] * arm_idx
            r_hat_actions = r_hat_per_arm[actions]

            v_dm[k] = np.sum(pi_e * r_hat_per_arm)
            v_dr[k] = v_dm[k] + np.mean(w * (rewards - r_hat_actions))

        dr_bias[j] = float(np.mean(v_dr) - v_true)
        ips_bias[j] = float(np.mean(v_ips) - v_true)
        dm_bias[j] = float(np.mean(v_dm) - v_true)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(len(combos))
    width = 0.25
    ax.bar(x_pos - width, ips_bias, width, color=COLOR_CHAIN[0], label="IPS(傾向スコアのみ依存)")
    ax.bar(x_pos, dm_bias, width, color=COLOR_CHAIN[2], label="DM(報酬モデルのみ依存)")
    ax.bar(x_pos + width, dr_bias, width, color=COLOR_OK, label="DR(二重にロバスト)")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in combos], fontsize=9)
    ax.set_ylabel("バイアス(平均推定値-真値)")
    ax.set_title(f"{n_reps}回のログ再サンプリングでのバイアス: "
                 "DRはどちらか一方が正しければ不偏に近い")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dr_double_robustness_grid.png")
    plt.close(fig)

    print(f"dr_double_robustness_grid.png saved (真値={v_true:.3f})")
    for (label, _, _), db, ib, mb in zip(combos, dr_bias, ips_bias, dm_bias):
        print(f"  {label}: DR bias={db:+.4f} IPS bias={ib:+.4f} DM bias={mb:+.4f}")


def plot_snips_variance_reduction():
    """傾向スコアが極端(一部の腕がほとんど選ばれない)場合に、IPSの重みが
    爆発して分散が大きくなる一方、自己正規化したSNIPSは分散を大きく
    抑えられることを示す。"""

    rng = np.random.default_rng(37)
    n_arms = 2
    p_true = np.array([0.30, 0.60])
    pi_b = np.array([0.98, 0.02])  # 極端な傾向スコア(arm1がほぼ選ばれない)
    pi_e = np.array([0.50, 0.50])
    v_true = float(np.sum(pi_e * p_true))

    n_log = 500
    n_reps = 300

    v_ips = np.zeros(n_reps)
    v_snips = np.zeros(n_reps)
    count_rare_arm = np.zeros(n_reps)

    for k in range(n_reps):
        rng_k = np.random.default_rng(4000 + k)
        actions = rng_k.choice(n_arms, size=n_log, p=pi_b)
        rewards = rng_k.binomial(1, p_true[actions]).astype(float)
        w = pi_e[actions] / pi_b[actions]
        v_ips[k] = np.mean(w * rewards)
        v_snips[k] = np.sum(w * rewards) / np.sum(w)
        count_rare_arm[k] = np.sum(actions == 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bp = axes[0].boxplot([v_ips, v_snips], tick_labels=["IPS", "SNIPS"], widths=0.5,
                          patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], [COLOR_DIVERGENT, COLOR_OK]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0].axhline(v_true, color="black", lw=1.3, ls="--", label=f"真の方策価値={v_true:.3f}")
    axes[0].set_ylabel("推定された方策価値")
    axes[0].set_title(f"{n_reps}回のログ再サンプリングでの推定値分布\n"
                       f"標準偏差: IPS={v_ips.std():.4f} SNIPS={v_snips.std():.4f}")
    axes[0].legend(fontsize=8)

    axes[1].scatter(count_rare_arm, v_ips, color=COLOR_DIVERGENT, alpha=0.6, s=22, label="IPS")
    axes[1].scatter(count_rare_arm, v_snips, color=COLOR_OK, alpha=0.6, s=22, label="SNIPS")
    axes[1].axhline(v_true, color="black", lw=1, ls="--")
    axes[1].set_xlabel(f"1runあたりのarm1(重み{pi_e[1]/pi_b[1]:.0f}倍)の出現回数")
    axes[1].set_ylabel("推定された方策価値")
    axes[1].set_title("IPSはarm1の出現回数に大きく揺さぶられるが\nSNIPSは相対的に安定")
    axes[1].legend(fontsize=8)

    fig.suptitle("極端な傾向スコアの下でのIPS vs SNIPS: 分散の違い", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "snips_variance_reduction.png")
    plt.close(fig)

    corr_ips = float(np.corrcoef(count_rare_arm, v_ips)[0, 1])
    corr_snips = float(np.corrcoef(count_rare_arm, v_snips)[0, 1])
    print(f"snips_variance_reduction.png saved (真値={v_true:.3f}, "
          f"IPS: mean={v_ips.mean():.4f} std={v_ips.std():.4f}, "
          f"SNIPS: mean={v_snips.mean():.4f} std={v_snips.std():.4f}, "
          f"arm1出現回数との相関: IPS={corr_ips:.3f} SNIPS={corr_snips:.3f})")


def plot_sndr_identity_reward_model_invariance():
    """ログ収集方策と評価方策がどちらも一様という特殊構造の下では、
    SNIPS・SNDR・単純平均がほぼ一致し、しかもSNDRの一致は報酬モデルの
    正しさに関わらず成り立つことを示す(実装検証の回帰テストとして使える根拠)。
    一方、分析者が傾向スコアを誤って一様だと思い込んでいる(実際のログ収集方策は
    偏っている)場合にはこの一致は崩れ、SNIPS・単純平均は同程度にバイアスを持つが、
    SNDRは報酬モデルの正誤によらずDMの働きで真値に近い値を保つことを示す。"""

    n_arms = 4
    arm_idx = np.arange(n_arms)
    p_true = np.array([0.10, 0.15, 0.45, 0.20])
    pi_uniform = np.full(n_arms, 0.25)
    pi_b_true_skewed = np.array([0.55, 0.25, 0.12, 0.08])  # 実際のログ収集方策(偏り)
    pi_e = pi_uniform
    n_log = 2000
    n_reps = 150

    def reward_model(actions, rewards, correct):
        if correct:
            return np.array([
                rewards[actions == a].mean() if np.any(actions == a) else rewards.mean()
                for a in arm_idx
            ])
        design = np.column_stack([np.ones(len(actions)), actions.astype(float)])
        b_hat, *_ = np.linalg.lstsq(design, rewards, rcond=None)
        return b_hat[0] + b_hat[1] * arm_idx

    def run(pi_b_true, pi_b_assumed, correct_reward, reps):
        v_true = float(np.sum(pi_e * p_true))
        plain = np.zeros(reps)
        snips = np.zeros(reps)
        sndr = np.zeros(reps)
        for k in range(reps):
            rng_k = np.random.default_rng(5000 + k)
            actions = rng_k.choice(n_arms, size=n_log, p=pi_b_true)
            rewards = rng_k.binomial(1, p_true[actions]).astype(float)
            # 分析者は pi_b_assumed を使って重みを計算する(真のpi_b_trueと
            # 一致しているとは限らない = 傾向スコアの誤設定)
            w = pi_e[actions] / pi_b_assumed[actions]
            plain[k] = rewards.mean()
            snips[k] = np.sum(w * rewards) / np.sum(w)
            r_hat_per_arm = reward_model(actions, rewards, correct_reward)
            dm = np.sum(pi_e * r_hat_per_arm)
            correction = np.sum(w * (rewards - r_hat_per_arm[actions])) / np.sum(w)
            sndr[k] = dm + correction
        return v_true, plain, snips, sndr

    # 特殊構造が正しく成り立つ場合: 真の収集方策も分析者の想定も一様
    v_true_ok, plain_ok, snips_ok, sndr_ok_correct = run(pi_uniform, pi_uniform, True, n_reps)
    _, _, _, sndr_ok_wrong = run(pi_uniform, pi_uniform, False, n_reps)

    # 傾向スコアの誤設定: 実際の収集方策は偏っているが、分析者は一様だと誤って思い込む
    v_true_mis, plain_mis, snips_mis, sndr_mis_correct = run(pi_b_true_skewed, pi_uniform, True, n_reps)
    _, _, _, sndr_mis_wrong = run(pi_b_true_skewed, pi_uniform, False, n_reps)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    labels4 = ["単純平均", "SNIPS", "SNDR\n(正しい報酬モデル)", "SNDR\n(誤設定な報酬モデル)"]
    xp = np.arange(4)

    means_ok = [plain_ok.mean(), snips_ok.mean(), sndr_ok_correct.mean(), sndr_ok_wrong.mean()]
    axes[0].axhline(v_true_ok, color="black", lw=1, ls="--", label=f"真の評価方策価値={v_true_ok:.3f}")
    axes[0].bar(xp, means_ok, color=[COLOR_CHAIN[0], COLOR_OK, COLOR_CHAIN[2], COLOR_ALT])
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(labels4, fontsize=8)
    axes[0].set_ylabel("推定値の平均")
    axes[0].set_title("傾向スコアの想定が正しい場合(一様=一様)\n報酬モデルの正誤によらずほぼ一致")

    means_mis = [plain_mis.mean(), snips_mis.mean(), sndr_mis_correct.mean(), sndr_mis_wrong.mean()]
    axes[1].axhline(v_true_mis, color="black", lw=1, ls="--", label=f"真の評価方策価値={v_true_mis:.3f}")
    axes[1].bar(xp, means_mis, color=[COLOR_CHAIN[0], COLOR_OK, COLOR_CHAIN[2], COLOR_ALT])
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(labels4, fontsize=8)
    axes[1].set_title("傾向スコアを誤設定(実際は偏りありなのに一様と誤認)\n"
                       "SNIPS・単純平均は同程度に崩れるが、SNDRは報酬モデルの\n正誤によらず真値に近い")

    fig.suptitle("SNIPS=SNDR=単純平均の一致は「傾向スコアが正しく分かっている」ことが前提", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "sndr_identity_reward_model_invariance.png")
    plt.close(fig)

    print(f"sndr_identity_reward_model_invariance.png saved:")
    print(f"  想定正しい時(真値={v_true_ok:.4f}): 単純平均={plain_ok.mean():.4f} SNIPS={snips_ok.mean():.4f} "
          f"SNDR(正)={sndr_ok_correct.mean():.4f} SNDR(誤)={sndr_ok_wrong.mean():.4f}")
    print(f"  傾向スコア誤設定時(真値={v_true_mis:.4f}): 単純平均={plain_mis.mean():.4f} SNIPS={snips_mis.mean():.4f} "
          f"SNDR(正)={sndr_mis_correct.mean():.4f} SNDR(誤)={sndr_mis_wrong.mean():.4f}")


def _gp_posterior(x_obs, y_obs, x_query, lengthscale=1.2, noise=0.05, amplitude=1.0):
    """RBFカーネルの閉形式GP事後平均・分散(numpyのみ)。"""

    def kernel(a, b):
        d2 = (a[:, None] - b[None, :]) ** 2
        return amplitude ** 2 * np.exp(-0.5 * d2 / lengthscale ** 2)

    k_xx = kernel(x_obs, x_obs) + noise ** 2 * np.eye(len(x_obs))
    k_xq = kernel(x_obs, x_query)
    k_qq_diag = amplitude ** 2 * np.ones(len(x_query))

    l_chol = np.linalg.cholesky(k_xx)
    alpha = np.linalg.solve(l_chol.T, np.linalg.solve(l_chol, y_obs))
    mu = k_xq.T @ alpha
    v = np.linalg.solve(l_chol, k_xq)
    var = k_qq_diag - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.clip(var, 1e-12, None))


def plot_regret_grid_resolution_limit():
    """獲得関数(GP-UCB)の最大化に使うグリッドが粗いと、探索戦略自体は
    健全でもsimple regretが理論上の0まで収束しないことを、細かいグリッドと
    粗いグリッドでのBOシミュレーション比較で示す。"""

    def g(x):
        return np.sin(x) + 0.3 * np.sin(3 * x) + 0.08 * x

    x_dense = np.linspace(0, 10, 4000)
    g_dense = g(x_dense)
    g_star = float(g_dense.max())
    x_star = float(x_dense[np.argmax(g_dense)])

    n_iter = 25
    kappa = 2.0

    def run_bo(candidate_grid, seed):
        rng_k = np.random.default_rng(seed)
        x_obs = rng_k.uniform(0, 10, size=3)
        y_obs = g(x_obs)
        regrets = []
        for _ in range(n_iter):
            mu, sd = _gp_posterior(x_obs, y_obs, candidate_grid)
            ucb = mu + kappa * sd
            x_next = candidate_grid[np.argmax(ucb)]
            y_next = g(x_next)
            x_obs = np.append(x_obs, x_next)
            y_obs = np.append(y_obs, y_next)
            regrets.append(g_star - y_obs.max())
        return np.array(regrets)

    fine_grid = np.linspace(0, 10, 2000)
    coarse_grid = np.linspace(0, 10, 6)

    n_reps = 15
    regrets_fine = np.array([run_bo(fine_grid, seed=100 + s) for s in range(n_reps)])
    regrets_coarse = np.array([run_bo(coarse_grid, seed=100 + s) for s in range(n_reps)])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    iters = np.arange(1, n_iter + 1)
    for regrets, color, label in [
        (regrets_fine, COLOR_OK, f"細かいグリッド({len(fine_grid)}点)"),
        (regrets_coarse, COLOR_DIVERGENT, f"粗いグリッド({len(coarse_grid)}点)"),
    ]:
        mean_r = regrets.mean(axis=0)
        axes[0].plot(iters, mean_r, "o-", color=color, label=label, ms=4)
        axes[0].fill_between(iters, regrets.min(axis=0), regrets.max(axis=0), color=color, alpha=0.15)
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_xlabel("反復回数")
    axes[0].set_ylabel("Simple Regret")
    axes[0].set_title(f"{n_reps}回平均: 粗いグリッドは0まで収束しない")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(x_dense, g_dense, color="black", lw=1, label="真の目的関数 g(x)")
    axes[1].axvline(x_star, color="black", lw=1, ls=":", label=f"真の最適点 x*={x_star:.2f}")
    nearest_coarse = coarse_grid[np.argmin(np.abs(coarse_grid - x_star))]
    axes[1].axvline(nearest_coarse, color=COLOR_DIVERGENT, lw=1.5, ls="--",
                     label=f"粗いグリッドの最近傍={nearest_coarse:.2f}")
    for xv in coarse_grid:
        axes[1].axvline(xv, color=COLOR_DIVERGENT, lw=0.4, alpha=0.3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g(x)")
    axes[1].set_title("粗いグリッドの候補点は真の最適点を通らない")
    axes[1].legend(fontsize=8)

    fig.suptitle("獲得関数の探索失敗と計算上の解像度不足の混同に注意", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "regret_grid_resolution_limit.png")
    plt.close(fig)

    final_fine = float(regrets_fine[:, -1].mean())
    final_coarse = float(regrets_coarse[:, -1].mean())
    print(f"regret_grid_resolution_limit.png saved (g*={g_star:.4f} at x*={x_star:.3f}, "
          f"最終反復でのregret平均: 細かいグリッド={final_fine:.4f} 粗いグリッド={final_coarse:.4f}, "
          f"粗いグリッド最近傍点との距離={abs(nearest_coarse - x_star):.3f})")


if __name__ == "__main__":
    plot_loo_elpd_diff()
    plot_brier_auc_independence()
    plot_brier_ipcw_censoring_correction()
    plot_dm_misspecification_bias()
    plot_dr_double_robustness_grid()
    plot_ope_bias_variance_tradeoff()
    plot_snips_variance_reduction()
    plot_sndr_identity_reward_model_invariance()
    plot_regret_grid_resolution_limit()
