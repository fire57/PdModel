import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pd_model.config import (
    BILL_COLUMNS,
    CATEGORICAL,
    FEATURES,
    PAYMENT_COLUMNS,
    STATUS_COLUMNS,
)

ENGINEERED_NUMERIC = [
    "max_delay",
    "delayed_months_count",
    "recent_delay",
    "bill_mean",
    "bill_std",
    "bill_max",
    "payment_sum",
    "payment_mean",
    "bill_change_recent_old",
    "limit_utilization_recent",
    "payment_to_positive_bill_sum",
    "zero_positive_bill_sum",
]


def normalize_categories(frame):
    result = frame.loc[:, FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].map(
            lambda value: np.nan if pd.isna(value) else str(int(value))
        )
    result["EDUCATION"] = result["EDUCATION"].replace(
        {"0": "unknown", "4": "other", "5": "other", "6": "other"}
    )
    result["MARRIAGE"] = result["MARRIAGE"].replace({"0": "unknown"})
    return result


def feature_groups(frame):
    canonical = normalize_categories(frame)
    for column in set(FEATURES) - set(CATEGORICAL):
        canonical[column] = canonical[column].astype(float)
    return canonical.apply(
        lambda row: hashlib.sha256(
            json.dumps(row.tolist(), separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        axis=1,
    )


class CreditFeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, use_aggregates=True):
        self.use_aggregates = use_aggregates

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES):
            raise ValueError("Pipeline ожидает ровно 23 raw-признака")
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if set(X.columns) != set(FEATURES):
            raise ValueError("Набор raw-признаков не соответствует обучению")
        result = normalize_categories(X)
        if not self.use_aggregates:
            return result
        delay = X[STATUS_COLUMNS].clip(lower=0)
        bills = X[BILL_COLUMNS]
        payments = X[PAYMENT_COLUMNS]
        result["max_delay"] = delay.max(axis=1)
        result["delayed_months_count"] = (X[STATUS_COLUMNS] > 0).sum(axis=1)
        result["recent_delay"] = X["PAY_0"].clip(lower=0)
        result["bill_mean"] = bills.mean(axis=1)
        result["bill_std"] = bills.std(axis=1, ddof=0)
        result["bill_max"] = bills.max(axis=1)
        result["payment_sum"] = payments.sum(axis=1)
        result["payment_mean"] = payments.mean(axis=1)
        result["bill_change_recent_old"] = X["BILL_AMT1"] - X["BILL_AMT6"]
        result["limit_utilization_recent"] = X["BILL_AMT1"] / X["LIMIT_BAL"]
        denominator = bills.clip(lower=0).sum(axis=1)
        result["payment_to_positive_bill_sum"] = payments.sum(axis=1) / (
            denominator.replace(0, np.nan)
        )
        result["zero_positive_bill_sum"] = (denominator == 0).astype(int)
        result["age_group"] = pd.cut(
            X["AGE"],
            bins=[18, 30, 45, 60, 101],
            labels=["18-29", "30-44", "45-59", "60-100"],
            right=False,
        ).astype(object)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        extra = ENGINEERED_NUMERIC + ["age_group"] if self.use_aggregates else []
        return np.asarray(FEATURES + extra, dtype=object)
