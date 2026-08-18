import pytest
from scipy import stats

from exercises.acquisition_functions.ex01_expected_improvement.solution import (
    expected_improvement,
)


def _reference_ei(mu: float, sigma: float, f_best: float) -> float:
    if sigma == 0:
        return 0.0
    z = (mu - f_best) / sigma
    return (mu - f_best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)


@pytest.mark.parametrize(
    "mu,sigma,f_best",
    [
        (5.0, 1.0, 3.0),  # mu > f_best: 改善が期待できる
        (2.0, 1.0, 3.0),  # mu < f_best: 改善は不確実性頼み
        (3.0, 0.5, 3.0),  # mu == f_best: 不確実性のボーナスのみ
        (10.0, 2.5, 1.0),
    ],
)
def test_matches_closed_form(mu, sigma, f_best):
    assert expected_improvement(mu, sigma, f_best) == pytest.approx(
        _reference_ei(mu, sigma, f_best), rel=1e-9
    )


def test_zero_sigma_returns_zero():
    assert expected_improvement(mu=5.0, sigma=0.0, f_best=3.0) == 0.0


def test_ei_is_never_negative():
    for mu, sigma, f_best in [(1.0, 2.0, 5.0), (0.0, 0.1, 10.0)]:
        assert expected_improvement(mu, sigma, f_best) >= 0.0
