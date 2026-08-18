from exercises.mcmc_diagnostics.ex01_rhat_classification.solution import classify_rhat


def test_converged_values():
    result = classify_rhat({"mu": 1.00, "sigma": 1.005})
    assert result == {"mu": "converged", "sigma": "converged"}


def test_not_converged_values():
    result = classify_rhat({"mu": 1.05, "tau": 3.51})
    assert result == {"mu": "not_converged", "tau": "not_converged"}


def test_boundary_is_not_converged():
    # 「1.01未満」が基準のため、1.01ちょうどは not_converged
    result = classify_rhat({"mu": 1.01})
    assert result == {"mu": "not_converged"}


def test_mixed_values():
    result = classify_rhat({"mu": 1.00, "tau": 3.51})
    assert result == {"mu": "converged", "tau": "not_converged"}


def test_empty_input():
    assert classify_rhat({}) == {}
