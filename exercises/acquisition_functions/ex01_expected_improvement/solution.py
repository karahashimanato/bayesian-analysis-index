"""演習: EI (Expected Improvement) の実装。問題文は problem.md を参照。"""


def expected_improvement(mu: float, sigma: float, f_best: float) -> float:
    """EI(x) = (mu - f_best)*Phi(z) + sigma*phi(z), z = (mu - f_best) / sigma

    sigma == 0 の場合は 0.0 を返す。
    """
    raise NotImplementedError("TODO: 実装してください")
