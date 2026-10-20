import numpy as np
import pytest

from pd_model.monitoring.psi import make_distribution, population_stability_index


def test_identity_and_shift():
    train = np.arange(1000, dtype=float)
    reference = make_distribution(train)
    assert population_stability_index(reference, train)["psi"] == pytest.approx(
        0, abs=1e-12
    )
    assert population_stability_index(reference, train + 3000)["psi"] > 0.25


def test_constant_and_zero_frequency():
    reference = make_distribution([5, 5, 5])
    assert population_stability_index(reference, [5, 5])["psi"] == pytest.approx(0)
    assert np.isfinite(population_stability_index(reference, [9, 9])["psi"])
    assert population_stability_index(reference, [9, 9])["psi"] > 0.25


def test_duplicate_quantiles_and_unseen_category():
    reference = make_distribution([0] * 90 + [1] * 10)
    assert np.isfinite(population_stability_index(reference, [0, 1000])["psi"])
    categorical = make_distribution(["a", "a", "b"], categorical=True)
    assert population_stability_index(categorical, ["c", "c"])["psi"] > 0.25


def test_empty_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        make_distribution([])
    reference = make_distribution([1, 2, 3])
    with pytest.raises(ValueError):
        population_stability_index(reference, [])
    with pytest.raises(ValueError):
        population_stability_index(reference, [np.nan])
