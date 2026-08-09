"""
tools/uncertainty-quantification-methods.md に埋め込む可視化画像を生成するスクリプト。

共通の合成1次元回帰データ(訓練データに意図的な空白領域[-2,2]を持たせ、
その外側[-6,-2]・[2,6]は密なデータ、さらに外側[-9,-6]・[6,9]は完全な
外挿領域とする)に対し、PyTorchで実際に5つの近似ベイズ手法
(MC Dropout, Deep Ensembles, Laplace近似(last-layer/full-network),
Bayes by Backprop, Anchored Ensembles)を学習・予測させ、
epistemic/aleatoric不確実性の挙動を比較する。

実行方法:
    source .venv/bin/activate
    python scripts/generate_uncertainty_quantification_plots.py

出力先: assets/uncertainty-quantification-methods/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plot_style import COLOR_ALT, COLOR_CHAIN, COLOR_DIVERGENT, COLOR_OK, apply_style

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "uncertainty-quantification-methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()
torch.set_num_threads(4)

X_GAP = (-2.0, 2.0)
X_TRAIN_RANGE = (-6.0, 6.0)
X_EXTRAP = (-9.0, 9.0)


def make_data(seed=0):
    """訓練データを2つの密なクラスタ([-6,-2]と[2,6])として生成し、
    その間([-2,2])を補間領域の空白、外側([-9,-6],[6,9])を外挿領域とする。"""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-6, -2, 60)
    x2 = rng.uniform(2, 6, 60)
    x_train = np.concatenate([x1, x2])

    def f_true(x):
        return np.sin(x)

    def sigma_true(x):
        return 0.1 + 0.05 * np.abs(x)

    y_train = f_true(x_train) + rng.normal(0, 1, len(x_train)) * sigma_true(x_train)
    x_grid = np.linspace(*X_EXTRAP, 400)
    return x_train, y_train, x_grid, f_true(x_grid), sigma_true(x_grid)


class MLP(nn.Module):
    """2つの隠れ層(tanh)+ヘテロスケダスティックな2ヘッド出力(mu, log_var)。"""

    def __init__(self, hidden=50, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 2)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def features(self, x):
        h = torch.tanh(self.fc1(x))
        h = self.drop1(h)
        h = torch.tanh(self.fc2(h))
        h = self.drop2(h)
        return h

    def forward(self, x):
        h = self.features(x)
        out = self.head(h)
        return out[:, 0:1], out[:, 1:2]


def hetero_nll(mu, log_var, y):
    return (0.5 * log_var + 0.5 * (y - mu) ** 2 / torch.exp(log_var)).mean()


def train_model(model, x, y, epochs=3000, lr=1e-2, anchor=None, prior_sigma=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        mu, log_var = model(x)
        loss = hetero_nll(mu, log_var, y)
        if anchor is not None:
            reg = sum(((p - p0) ** 2).sum() for p, p0 in zip(model.parameters(), anchor))
            loss = loss + reg / (2 * prior_sigma ** 2) / x.shape[0]
        loss.backward()
        opt.step()
    return model


def _to_tensor(a):
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1)


def mc_dropout_predict(x_t, y_t, x_grid, dropout_p=0.1, T=200, epochs=3000, seed=1):
    torch.manual_seed(seed)
    model = MLP(dropout_p=dropout_p)
    train_model(model, x_t, y_t, epochs=epochs)
    model.train()  # dropoutを推論時も有効にしたまま複数回forward
    xg = _to_tensor(x_grid)
    mus, sigmas = [], []
    with torch.no_grad():
        for _ in range(T):
            mu, log_var = model(xg)
            mus.append(mu.numpy().flatten())
            sigmas.append(np.exp(0.5 * log_var.numpy().flatten()))
    mus, sigmas = np.array(mus), np.array(sigmas)
    return mus.mean(0), mus.std(0), sigmas.mean(0)


def deep_ensemble_predict(x_t, y_t, x_grid, M=5, epochs=3000, seed=10):
    xg = _to_tensor(x_grid)
    mus, sigmas = [], []
    for m in range(M):
        torch.manual_seed(seed + m)
        model = MLP(dropout_p=0.0)
        train_model(model, x_t, y_t, epochs=epochs)
        model.eval()
        with torch.no_grad():
            mu, log_var = model(xg)
        mus.append(mu.numpy().flatten())
        sigmas.append(np.exp(0.5 * log_var.numpy().flatten()))
    mus, sigmas = np.array(mus), np.array(sigmas)
    return mus.mean(0), mus.std(0), sigmas.mean(0), mus


def ensemble_regularized_predict(x_t, y_t, x_grid, M=5, epochs=3000, seed=20, prior_sigma=3.0,
                                  anchor_mode="self"):
    """anchor_mode="zero"は通常の重み減衰(原点への正則化)、"self"はAnchored Ensembles
    (各メンバー自身のランダムな初期値への正則化)。正則化の強さ(prior_sigma)は同一にして
    公平に比較する。"""
    xg = _to_tensor(x_grid)
    mus, sigmas = [], []
    for m in range(M):
        torch.manual_seed(seed + m)
        model = MLP(dropout_p=0.0)
        if anchor_mode == "self":
            anchor = [p.clone().detach() for p in model.parameters()]
        else:
            anchor = [torch.zeros_like(p) for p in model.parameters()]
        train_model(model, x_t, y_t, epochs=epochs, anchor=anchor, prior_sigma=prior_sigma)
        model.eval()
        with torch.no_grad():
            mu, log_var = model(xg)
        mus.append(mu.numpy().flatten())
        sigmas.append(np.exp(0.5 * log_var.numpy().flatten()))
    mus, sigmas = np.array(mus), np.array(sigmas)
    return mus.mean(0), mus.std(0), sigmas.mean(0), mus


def laplace_last_layer_predict(x_t, y_t, x_grid, epochs=3000, seed=30, prior_precision=1.0):
    """凍結したtrunk特徴量の上に、最終層だけ閉形式のベイズ線形回帰を当てはめる
    (last-layer Laplace)。"""
    torch.manual_seed(seed)
    model = MLP(dropout_p=0.0)
    train_model(model, x_t, y_t, epochs=epochs)
    model.eval()
    with torch.no_grad():
        feat_train = model.features(x_t).numpy()
    W = model.head.weight.detach().numpy()
    b = model.head.bias.detach().numpy()
    w_mu, b_mu = W[0], b[0]

    Phi = np.concatenate([feat_train, np.ones((feat_train.shape[0], 1))], axis=1)
    y_arr = y_t.numpy().flatten()
    mu_pred_train = Phi @ np.concatenate([w_mu, [b_mu]])
    noise_var = max(np.var(y_arr - mu_pred_train), 1e-4)

    A = (Phi.T @ Phi) / noise_var + prior_precision * np.eye(Phi.shape[1])
    A_inv = np.linalg.inv(A)

    with torch.no_grad():
        feat_grid = model.features(_to_tensor(x_grid)).numpy()
    Phi_grid = np.concatenate([feat_grid, np.ones((feat_grid.shape[0], 1))], axis=1)
    w_map = np.concatenate([w_mu, [b_mu]])
    mu_grid = Phi_grid @ w_map
    var_grid = np.einsum("ij,jk,ik->i", Phi_grid, A_inv, Phi_grid)
    return mu_grid, np.sqrt(np.clip(var_grid, 0, None))


def laplace_full_network_predict(x_t, y_t, x_grid, epochs=3000, seed=30,
                                  prior_precision=1.0, T=200):
    """全パラメータについて経験的フィッシャー対角近似(対角Gauss-Newton近似の実用的な
    代替)でヘッセ行列を近似し、事後分布からサンプリングして予測する(full-network Laplace)。
    last-layer Laplaceと同じMAP解(同じseed・同じ学習)を再利用して比較しやすくする。"""
    torch.manual_seed(seed)
    model = MLP(dropout_p=0.0)
    train_model(model, x_t, y_t, epochs=epochs)
    model.eval()
    params = list(model.parameters())
    fisher_diag = [torch.zeros_like(p) for p in params]
    n = x_t.shape[0]
    for i in range(n):
        model.zero_grad()
        mu, log_var = model(x_t[i:i + 1])
        loss = hetero_nll(mu, log_var, y_t[i:i + 1])
        loss.backward()
        for fd, p in zip(fisher_diag, params):
            fd += p.grad.detach() ** 2

    # fisher_diagは既に全n訓練点にわたる二乗勾配の総和(=経験的フィッシャー情報量そのもの)
    # なので、事後精度は fisher_diag + prior_precision であり、さらにnを掛けてはいけない。
    post_std = [1.0 / torch.sqrt(fd + prior_precision) for fd in fisher_diag]
    orig_state = [p.clone().detach() for p in params]
    xg = _to_tensor(x_grid)
    mus = []
    with torch.no_grad():
        for _ in range(T):
            for p, p0, std in zip(params, orig_state, post_std):
                p.copy_(p0 + std * torch.randn_like(p0))
            mu, _ = model(xg)
            mus.append(mu.numpy().flatten())
        for p, p0 in zip(params, orig_state):
            p.copy_(p0)
    mus = np.array(mus)
    return mus.mean(0), mus.std(0)


class BBBLinear(nn.Module):
    """各重み・バイアスをNormal(mu, softplus(rho))とする変分ベイズ線形層。"""

    def __init__(self, in_f, out_f, prior_sigma=1.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.randn(out_f, in_f) * (1.0 / np.sqrt(in_f)))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        w_sigma = torch.nn.functional.softplus(self.w_rho)
        b_sigma = torch.nn.functional.softplus(self.b_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return torch.nn.functional.linear(x, w, b)

    def kl(self):
        w_sigma = torch.nn.functional.softplus(self.w_rho)
        b_sigma = torch.nn.functional.softplus(self.b_rho)

        def kl_term(mu, sigma):
            return (torch.log(self.prior_sigma / sigma) +
                    (sigma ** 2 + mu ** 2) / (2 * self.prior_sigma ** 2) - 0.5).sum()

        return kl_term(self.w_mu, w_sigma) + kl_term(self.b_mu, b_sigma)


class BBBNet(nn.Module):
    def __init__(self, hidden=50, prior_sigma=1.0):
        super().__init__()
        self.l1 = BBBLinear(1, hidden, prior_sigma)
        self.l2 = BBBLinear(hidden, hidden, prior_sigma)
        self.head = BBBLinear(hidden, 2, prior_sigma)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        h = torch.tanh(self.l2(h))
        out = self.head(h)
        return out[:, 0:1], out[:, 1:2]

    def kl(self):
        return self.l1.kl() + self.l2.kl() + self.head.kl()


def bbb_predict(x_t, y_t, x_grid, epochs=6000, lr=5e-3, seed=40, T=200, prior_sigma=1.0):
    torch.manual_seed(seed)
    model = BBBNet(prior_sigma=prior_sigma)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_t.shape[0]
    for _ in range(epochs):
        opt.zero_grad()
        mu, log_var = model(x_t)
        loss = hetero_nll(mu, log_var, y_t) + model.kl() / n
        loss.backward()
        opt.step()

    xg = _to_tensor(x_grid)
    mus, sigmas = [], []
    with torch.no_grad():
        for _ in range(T):
            mu, log_var = model(xg)
            mus.append(mu.numpy().flatten())
            sigmas.append(np.exp(0.5 * log_var.numpy().flatten()))
    mus, sigmas = np.array(mus), np.array(sigmas)
    return mus.mean(0), mus.std(0), sigmas.mean(0)


def _shade_regions(ax, x_train):
    ax.axvspan(*X_GAP, color="gray", alpha=0.15, label="補間領域の空白(in-between)")
    ax.axvspan(X_EXTRAP[0], X_TRAIN_RANGE[0], color=COLOR_DIVERGENT, alpha=0.08)
    ax.axvspan(X_TRAIN_RANGE[1], X_EXTRAP[1], color=COLOR_DIVERGENT, alpha=0.08,
               label="外挿領域(訓練範囲外)")
    ax.scatter(x_train, np.full_like(x_train, ax.get_ylim()[0]), marker="|",
               color="black", alpha=0.3, s=40)


def plot_epistemic_aleatoric_decomposition(data, mu_mean, epistemic, aleatoric):
    x_train, y_train, x_grid, f_grid, sigma_grid = (
        data["x_train"], data["y_train"], data["x_grid"], data["f_grid"], data["sigma_grid"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    ax.plot(x_grid, f_grid, color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x_train, y_train, color="black", s=10, alpha=0.4, label="訓練データ")
    ax.plot(x_grid, mu_mean, color=COLOR_OK, lw=2, label="予測平均")
    total = np.sqrt(epistemic ** 2 + aleatoric ** 2)
    ax.fill_between(x_grid, mu_mean - 2 * total, mu_mean + 2 * total, color=COLOR_OK, alpha=0.2,
                     label="全体の予測区間(±2SD)")
    ax.set_ylabel("y")
    ax.set_title("Deep Ensemblesによる予測(全体の不確実性)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)

    ax = axes[1]
    ax.plot(x_grid, epistemic, color=COLOR_DIVERGENT, lw=2, label="epistemic(アンサンブル間の平均のばらつき)")
    ax.plot(x_grid, aleatoric, color=COLOR_ALT, lw=2, label="aleatoric(各メンバーの学習済みσの平均)")
    ax.plot(x_grid, sigma_grid, color="black", lw=1, ls=":", label="真のaleatoric(σ_true(x))")
    _shade_regions(ax, x_train)
    ax.set_xlabel("x")
    ax.set_ylabel("不確実性の大きさ")
    ax.set_title("epistemicとaleatoricの分解(訓練範囲外でepistemicのみ増加する)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "epistemic_aleatoric_decomposition.png")
    plt.close(fig)
    print("epistemic_aleatoric_decomposition.png saved "
          f"(epistemic@gap={epistemic[np.argmin(np.abs(x_grid))]:.3f}, "
          f"epistemic@extrap={epistemic[-1]:.3f}, aleatoric@extrap={aleatoric[-1]:.3f}, "
          f"true_sigma@extrap={sigma_grid[-1]:.3f})")


def plot_mc_dropout_vs_deep_ensembles(data, mu_mc, epi_mc, mu_de, epi_de):
    x_train, y_train, x_grid, f_grid, sigma_grid = (
        data["x_train"], data["y_train"], data["x_grid"], data["f_grid"], data["sigma_grid"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mu, epi, label, color in [
        (axes[0], mu_mc, epi_mc, "MC Dropout(T=200)", COLOR_ALT),
        (axes[1], mu_de, epi_de, "Deep Ensembles(M=5)", COLOR_OK),
    ]:
        ax.plot(x_grid, f_grid, color="black", lw=1.2, ls="--", label="真の関数")
        ax.scatter(x_train, y_train, color="black", s=8, alpha=0.35)
        ax.plot(x_grid, mu, color=color, lw=2, label="予測平均")
        ax.fill_between(x_grid, mu - 2 * epi, mu + 2 * epi, color=color, alpha=0.25,
                         label="epistemic(±2SD)")
        ax.axvspan(*X_GAP, color="gray", alpha=0.15)
        ax.axvspan(X_EXTRAP[0], X_TRAIN_RANGE[0], color=COLOR_DIVERGENT, alpha=0.08)
        ax.axvspan(X_TRAIN_RANGE[1], X_EXTRAP[1], color=COLOR_DIVERGENT, alpha=0.08)
        ax.set_xlabel("x")
        ax.set_title(f"{label}\n外挿域(右端)でのepistemic={epi[-1]:.3f}")
        ax.legend(fontsize=8, loc="upper center", ncol=1)
    axes[0].set_ylabel("y")

    fig.suptitle("MC DropoutはDeep Ensemblesに比べ、訓練範囲外(赤帯)でのepistemicの増加が緩やか")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mc_dropout_vs_deep_ensembles.png")
    plt.close(fig)
    print("mc_dropout_vs_deep_ensembles.png saved "
          f"(MC Dropout epistemic@extrap={epi_mc[-1]:.3f}, "
          f"Deep Ensembles epistemic@extrap={epi_de[-1]:.3f})")


def plot_laplace_last_layer_vs_full(data, mu_ll, epi_ll, mu_fn, epi_fn):
    x_train, y_train, x_grid, f_grid, sigma_grid = (
        data["x_train"], data["y_train"], data["x_grid"], data["f_grid"], data["sigma_grid"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_grid, f_grid, color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x_train, y_train, color="black", s=8, alpha=0.35)
    ax.plot(x_grid, mu_ll, color=COLOR_DIVERGENT, lw=1.5)
    ax.fill_between(x_grid, mu_ll - 2 * epi_ll, mu_ll + 2 * epi_ll, color=COLOR_DIVERGENT, alpha=0.25,
                     label=f"last-layer Laplace(epistemic@外挿域={epi_ll[-1]:.3f})")
    ax.plot(x_grid, mu_fn, color=COLOR_OK, lw=1.5)
    ax.fill_between(x_grid, mu_fn - 2 * epi_fn, mu_fn + 2 * epi_fn, color=COLOR_OK, alpha=0.25,
                     label=f"full-network Laplace(対角フィッシャー近似, epistemic@外挿域={epi_fn[-1]:.3f})")
    ax.axvspan(*X_GAP, color="gray", alpha=0.15)
    ax.axvspan(X_EXTRAP[0], X_TRAIN_RANGE[0], color=COLOR_DIVERGENT, alpha=0.06)
    ax.axvspan(X_TRAIN_RANGE[1], X_EXTRAP[1], color=COLOR_DIVERGENT, alpha=0.06, label="外挿領域")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("この1次元トイ例ではlast-layerもfull-network(対角近似)も\n訓練範囲外でepistemicが同程度に増加する(last-layerの帯はより不規則)")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "laplace_last_layer_vs_full.png")
    plt.close(fig)
    print("laplace_last_layer_vs_full.png saved "
          f"(last-layer epistemic@extrap={epi_ll[-1]:.4f}, "
          f"full-network epistemic@extrap={epi_fn[-1]:.4f})")


def plot_bayes_by_backprop(data, mu_bbb, epi_bbb, mu_de, epi_de):
    x_train, y_train, x_grid, f_grid, sigma_grid = (
        data["x_train"], data["y_train"], data["x_grid"], data["f_grid"], data["sigma_grid"])

    width_bbb = float(np.mean(4 * epi_bbb))
    width_de = float(np.mean(4 * epi_de))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x_grid, f_grid, color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x_train, y_train, color="black", s=8, alpha=0.35)
    ax.plot(x_grid, mu_de, color=COLOR_OK, lw=1.5)
    ax.fill_between(x_grid, mu_de - 2 * epi_de, mu_de + 2 * epi_de, color=COLOR_OK, alpha=0.25,
                     label=f"Deep Ensembles(平均区間幅={width_de:.2f})")
    ax.plot(x_grid, mu_bbb, color=COLOR_DIVERGENT, lw=1.5)
    ax.fill_between(x_grid, mu_bbb - 2 * epi_bbb, mu_bbb + 2 * epi_bbb, color=COLOR_DIVERGENT, alpha=0.2,
                     label=f"Bayes by Backprop(平均区間幅={width_bbb:.2f})")
    ax.axvspan(*X_GAP, color="gray", alpha=0.15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("予測区間(epistemicの±2SD)の比較")
    ax.legend(fontsize=8, loc="upper center")

    ax = axes[1]
    bars = ax.bar(["Deep Ensembles", "Bayes by Backprop"], [width_de, width_bbb],
                   color=[COLOR_OK, COLOR_DIVERGENT], alpha=0.85)
    for b, v in zip(bars, [width_de, width_bbb]):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("予測区間の平均幅(全xで平均)")
    ax.set_title(f"Bayes by Backpropの区間幅は\nDeep Ensemblesの約{width_bbb / width_de:.1f}倍")

    fig.suptitle("Bayes by Backpropは平均場変分近似のKL項により、より保守的(広い)な予測区間を与える")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bayes_by_backprop_interval_width.png")
    plt.close(fig)
    print(f"bayes_by_backprop_interval_width.png saved "
          f"(width_de={width_de:.3f}, width_bbb={width_bbb:.3f}, ratio={width_bbb / width_de:.2f})")


def plot_anchored_ensembles(data, mu_wd, epi_wd, mu_an, epi_an):
    x_train, y_train, x_grid, f_grid, sigma_grid = (
        data["x_train"], data["y_train"], data["x_grid"], data["f_grid"], data["sigma_grid"])

    extrap_mask = x_grid > X_TRAIN_RANGE[1]
    gap_mask = (x_grid > X_GAP[0]) & (x_grid < X_GAP[1])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_grid, f_grid, color="black", lw=1.2, ls="--", label="真の関数")
    ax.scatter(x_train, y_train, color="black", s=8, alpha=0.35)
    ax.plot(x_grid, mu_wd, color=COLOR_OK, lw=1.5)
    ax.fill_between(x_grid, mu_wd - 2 * epi_wd, mu_wd + 2 * epi_wd, color=COLOR_OK, alpha=0.2,
                     label="通常の重み減衰(原点=0への正則化、同じ正則化強度)")
    ax.plot(x_grid, mu_an, color=COLOR_DIVERGENT, lw=1.5)
    ax.fill_between(x_grid, mu_an - 2 * epi_an, mu_an + 2 * epi_an, color=COLOR_DIVERGENT, alpha=0.2,
                     label="Anchored Ensembles(各メンバー自身の初期値への正則化、同じ正則化強度)")
    ax.axvspan(*X_GAP, color="gray", alpha=0.15, label="補間領域の空白")
    ax.axvspan(X_TRAIN_RANGE[1], X_EXTRAP[1], color="gray", alpha=0.06)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"この1次元トイ例(M=5, 同一の正則化強度)では、外挿域(x>6)のepistemicは\n"
                 f"原点への正則化{epi_wd[extrap_mask].mean():.3f}に対しAnchoredで{epi_an[extrap_mask].mean():.3f}とむしろ小さく、"
                 f"文献の傾向とは逆の結果になった")
    ax.legend(fontsize=8.5, loc="upper center", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "anchored_ensembles_extrapolation.png")
    plt.close(fig)
    print(f"anchored_ensembles_extrapolation.png saved "
          f"(extrap: weight_decay={epi_wd[extrap_mask].mean():.4f} anchored={epi_an[extrap_mask].mean():.4f}, "
          f"gap: weight_decay={epi_wd[gap_mask].mean():.4f} anchored={epi_an[gap_mask].mean():.4f})")


def plot_in_between_uncertainty(data, epi_mc, epi_de, epi_ll, epi_bbb, epi_an):
    x_grid = data["x_grid"]

    methods = [
        ("MC Dropout", epi_mc, COLOR_ALT),
        ("Deep Ensembles", epi_de, COLOR_OK),
        ("Laplace(last-layer)", epi_ll, COLOR_DIVERGENT),
        ("Bayes by Backprop", epi_bbb, COLOR_CHAIN[2]),
        ("Anchored Ensembles", epi_an, COLOR_CHAIN[3]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, epi, color in methods:
        epi_norm = epi / epi.max()
        ax.plot(x_grid, epi_norm, color=color, lw=2, label=label)
    ax.axvspan(*X_GAP, color="gray", alpha=0.15, label="補間領域の空白(in-between)")
    ax.axvspan(X_EXTRAP[0], X_TRAIN_RANGE[0], color=COLOR_DIVERGENT, alpha=0.06)
    ax.axvspan(X_TRAIN_RANGE[1], X_EXTRAP[1], color=COLOR_DIVERGENT, alpha=0.06, label="外挿領域")
    ax.set_xlabel("x")
    ax.set_ylabel("epistemic不確実性(各手法の最大値で正規化)")
    ax.set_title("Deep Ensemblesは外挿域(赤帯)と補間領域の空白(灰色帯)で明確な非対称性を示すが、\n"
                 "他手法ではこの非対称性は弱い(MC Dropout/Bayes by Backpropはどちらも高止まり、\n"
                 "Anchored Ensemblesはどちらも同程度に低い)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "in_between_uncertainty_problem.png")
    plt.close(fig)

    gap_mask = (x_grid > X_GAP[0]) & (x_grid < X_GAP[1])
    extrap_mask = x_grid > X_TRAIN_RANGE[1]
    for label, epi, _ in methods:
        epi_norm = epi / epi.max()
        print(f"  {label}: gap_norm={epi_norm[gap_mask].mean():.3f}, "
              f"extrap_norm={epi_norm[extrap_mask].mean():.3f}")
    print("in_between_uncertainty_problem.png saved")


def main():
    x_train, y_train, x_grid, f_grid, sigma_grid = make_data(seed=0)
    x_t, y_t = _to_tensor(x_train), _to_tensor(y_train)
    data = dict(x_train=x_train, y_train=y_train, x_grid=x_grid, f_grid=f_grid, sigma_grid=sigma_grid)

    print("training MC Dropout...")
    mu_mc, epi_mc, alea_mc = mc_dropout_predict(x_t, y_t, x_grid, seed=1)

    print("training Deep Ensembles...")
    mu_de, epi_de, alea_de, _ = deep_ensemble_predict(x_t, y_t, x_grid, seed=10)

    print("training Anchored Ensembles (and weight-decay-to-origin comparison)...")
    mu_an, epi_an, alea_an, _ = ensemble_regularized_predict(
        x_t, y_t, x_grid, seed=20, prior_sigma=3.0, anchor_mode="self")
    mu_wd, epi_wd, alea_wd, _ = ensemble_regularized_predict(
        x_t, y_t, x_grid, seed=20, prior_sigma=3.0, anchor_mode="zero")

    print("training Laplace (last-layer & full-network)...")
    mu_ll, epi_ll = laplace_last_layer_predict(x_t, y_t, x_grid, seed=30, prior_precision=20.0)
    mu_fn, epi_fn = laplace_full_network_predict(x_t, y_t, x_grid, seed=30, prior_precision=20.0)

    print("training Bayes by Backprop...")
    mu_bbb, epi_bbb, alea_bbb = bbb_predict(x_t, y_t, x_grid, seed=40)

    plot_epistemic_aleatoric_decomposition(data, mu_de, epi_de, alea_de)
    plot_mc_dropout_vs_deep_ensembles(data, mu_mc, epi_mc, mu_de, epi_de)
    plot_laplace_last_layer_vs_full(data, mu_ll, epi_ll, mu_fn, epi_fn)
    plot_bayes_by_backprop(data, mu_bbb, epi_bbb, mu_de, epi_de)
    plot_anchored_ensembles(data, mu_wd, epi_wd, mu_an, epi_an)
    plot_in_between_uncertainty(data, epi_mc, epi_de, epi_ll, epi_bbb, epi_an)


if __name__ == "__main__":
    main()
