import math

import numpy as np
import pandas as pd
import pytest

from pd_model.config import BILL_COLUMNS, FEATURES, PAYMENT_COLUMNS, STATUS_COLUMNS
from pd_model.features.build_features import CreditFeatureEngineer


def test_manual_aggregates_and_no_mutation(raw_sample):
    frame = raw_sample[FEATURES].iloc[:1].copy()
    bills, payments, statuses = (
        [-1, 2, 0, 4, 3, 2],
        [1, 2, 3, 4, 5, 6],
        [-2, -1, 0, 1, 2, 3],
    )
    frame.loc[:, BILL_COLUMNS] = bills
    frame.loc[:, PAYMENT_COLUMNS] = payments
    frame.loc[:, STATUS_COLUMNS] = statuses
    original = frame.copy(deep=True)
    output = CreditFeatureEngineer().fit_transform(frame).iloc[0]
    assert output.max_delay == 3
    assert output.delayed_months_count == 3
    assert output.recent_delay == 0
    assert output.bill_mean == pytest.approx(10 / 6)
    variance = sum((value - 10 / 6) ** 2 for value in bills) / 6
    assert output.bill_std == pytest.approx(math.sqrt(variance))
    assert output.payment_sum == 21
    assert output.payment_to_positive_bill_sum == pytest.approx(21 / 11)
    assert output.bill_change_recent_old == -3
    assert output.limit_utilization_recent < 0
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "age,expected",
    [
        (18, "18-29"),
        (29, "18-29"),
        (30, "30-44"),
        (44, "30-44"),
        (45, "45-59"),
        (59, "45-59"),
        (60, "60-100"),
        (100, "60-100"),
    ],
)
def test_age_boundaries(raw_sample, age, expected):
    frame = raw_sample[FEATURES].iloc[:1].copy()
    frame["AGE"] = age
    result = CreditFeatureEngineer().fit_transform(frame)
    assert result.iloc[0].age_group == expected


def test_zero_denominator(raw_sample):
    frame = raw_sample[FEATURES].iloc[:1].copy()
    frame.loc[:, BILL_COLUMNS] = 0
    result = CreditFeatureEngineer().fit_transform(frame).iloc[0]
    assert np.isnan(result.payment_to_positive_bill_sum)
    assert result.zero_positive_bill_sum == 1
