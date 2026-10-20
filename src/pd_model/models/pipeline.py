from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_model.config import CATEGORICAL, NUMERIC
from pd_model.features.build_features import CreditFeatureEngineer, ENGINEERED_NUMERIC


def create_pipeline(algorithm="logistic", aggregates=True, seed=42, params=None):
    estimators = {
        "dummy": DummyClassifier,
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    if algorithm not in estimators:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")
    classifier = estimators[algorithm](random_state=seed, **(params or {}))
    numeric = NUMERIC + (ENGINEERED_NUMERIC if aggregates else [])
    categorical = CATEGORICAL + (["age_group"] if aggregates else [])
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("features", CreditFeatureEngineer(use_aggregates=aggregates)),
            ("preprocessor", preprocess),
            ("classifier", classifier),
        ]
    )
