import pandas as pd

from pd_model.config import TARGET
from pd_model.data.prepare import split_frame
from pd_model.features.build_features import feature_groups


def test_group_split_is_reproducible(raw_sample):
    conflicting = raw_sample.iloc[:1].copy()
    conflicting["ID"] = 99999
    conflicting[TARGET] = 1 - conflicting[TARGET]
    frame = pd.concat([raw_sample, conflicting], ignore_index=True)
    train, test = split_frame(frame)
    again_train, again_test = split_frame(frame)
    assert train.ID.tolist() == again_train.ID.tolist()
    assert test.ID.tolist() == again_test.ID.tolist()
    assert len(train) + len(test) == len(frame)
    assert not set(train.ID) & set(test.ID)
    assert not set(train.feature_group) & set(test.feature_group)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.12
    groups = feature_groups(frame)
    assert groups.iloc[0] == groups.iloc[-1]


def test_merged_categories_share_group(raw_sample):
    frame = raw_sample.iloc[:1].copy()
    frame["EDUCATION"] = 4
    other = frame.copy()
    other["EDUCATION"] = 5
    assert feature_groups(frame).iloc[0] == feature_groups(other).iloc[0]
