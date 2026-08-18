import pytest

from exercises.prior_distributions.ex01_pick_prior_family.solution import (
    pick_prior_family,
)


def test_probability():
    assert pick_prior_family("probability") == "Beta"


def test_positive_scale_avoid_explosion():
    assert (
        pick_prior_family("positive_scale", "avoid_zero_variance_explosion")
        == "Gamma"
    )


def test_positive_scale_no_constraint():
    assert (
        pick_prior_family("positive_scale", "no_constraint")
        == "HalfNormal_or_HalfCauchy"
    )


def test_changepoint_avoid_ess_drop():
    assert (
        pick_prior_family("changepoint", "avoid_ess_drop") == "Uniform_plus_sigmoid"
    )


def test_changepoint_discrete_ok():
    assert pick_prior_family("changepoint", "discrete_ok") == "DiscreteUniform"


def test_real_unconstrained():
    assert pick_prior_family("real_unconstrained") == "Normal"


def test_wide_magnitude_positive():
    assert pick_prior_family("wide_magnitude_positive") == "LogNormal"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        pick_prior_family("not_a_real_kind")


def test_missing_detail_raises():
    with pytest.raises(ValueError):
        pick_prior_family("positive_scale")


def test_invalid_detail_raises():
    with pytest.raises(ValueError):
        pick_prior_family("positive_scale", "bogus_detail")
