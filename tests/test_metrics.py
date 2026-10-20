import pytest

from pd_model.models.metrics import classification_metrics


def test_manually_checkable_metrics():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert result["roc_auc"] == 0.75
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert result["f1"] == pytest.approx(0.8)


def test_no_positive_prediction():
    result = classification_metrics([0, 1], [0.1, 0.2])
    assert result["precision"] == result["recall"] == result["f1"] == 0


@pytest.mark.parametrize(
    "y,p", [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, 1.2]), ([0, 1], [0.1])]
)
def test_invalid_metrics_inputs(y, p):
    with pytest.raises(ValueError):
        classification_metrics(y, p)
