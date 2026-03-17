from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


FeaturePayload = pd.DataFrame | Mapping[str, object] | None


def build_stack_frame(
    df: pd.DataFrame,
    *,
    context_cols: list[str],
    extra_features: Mapping[str, object],
) -> pd.DataFrame:
    frame = df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    for name, values in extra_features.items():
        arr = np.asarray(values, dtype=float).reshape(len(df), -1)
        if arr.shape[1] == 1:
            frame[name] = arr[:, 0]
            continue
        for idx in range(arr.shape[1]):
            frame[f"{name}_{idx:02d}"] = arr[:, idx]
    return frame


def _coerce_extra_frame(extra_features: FeaturePayload, row_count: int) -> pd.DataFrame:
    if extra_features is None:
        return pd.DataFrame(index=range(row_count))
    if isinstance(extra_features, pd.DataFrame):
        return extra_features.reset_index(drop=True).copy()
    frame = pd.DataFrame(index=range(row_count))
    for name, values in extra_features.items():
        arr = np.asarray(values, dtype=float).reshape(row_count, -1)
        if arr.shape[1] == 1:
            frame[name] = arr[:, 0]
            continue
        for idx in range(arr.shape[1]):
            frame[f"{name}_{idx:02d}"] = arr[:, idx]
    return frame


def _append_extra_features(base: pd.DataFrame, extra_features: FeaturePayload) -> pd.DataFrame:
    extra = _coerce_extra_frame(extra_features, len(base))
    if extra.empty:
        return base.reset_index(drop=True).copy()
    return pd.concat([base.reset_index(drop=True), extra], axis=1)


@dataclass(frozen=True)
class StackSet:
    train: pd.DataFrame
    unlabeled: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def from_context_frames(
        cls,
        *,
        train_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        test_df: pd.DataFrame,
        context_cols: list[str],
        train_features: FeaturePayload = None,
        unlabeled_features: FeaturePayload = None,
        test_features: FeaturePayload = None,
    ) -> "StackSet":
        return cls(
            train=_append_extra_features(
                train_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                train_features,
            ),
            unlabeled=_append_extra_features(
                unlabeled_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                unlabeled_features,
            ),
            test=_append_extra_features(
                test_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                test_features,
            ),
        )

    def apply(self, augmenter: "StackAugmenter") -> "StackSet":
        return augmenter.apply(self)

    def with_frames(
        self,
        *,
        train: pd.DataFrame | None = None,
        unlabeled: pd.DataFrame | None = None,
        test: pd.DataFrame | None = None,
    ) -> "StackSet":
        return StackSet(
            train=train if train is not None else self.train.copy(),
            unlabeled=unlabeled if unlabeled is not None else self.unlabeled.copy(),
            test=test if test is not None else self.test.copy(),
        )


class StackAugmenter(ABC):
    @abstractmethod
    def apply(self, stack: StackSet) -> StackSet:
        raise NotImplementedError


@dataclass(frozen=True)
class FrameAugmenter(StackAugmenter):
    train_features: FeaturePayload = None
    unlabeled_features: FeaturePayload = None
    test_features: FeaturePayload = None

    def apply(self, stack: StackSet) -> StackSet:
        return StackSet(
            train=_append_extra_features(stack.train, self.train_features),
            unlabeled=_append_extra_features(stack.unlabeled, self.unlabeled_features),
            test=_append_extra_features(stack.test, self.test_features),
        )
