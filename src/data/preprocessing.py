"""Fold-safe preprocessing for imaging and clinical features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


IMAGING_PREFIXES = (
    "GM_vol_Region_",
    "WM_vol_Region_",
    "CSF_vol_Region_",
    "PET_suvr_Region_",
)


def imaging_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return all ROI imaging columns in their existing dataframe order."""
    return [
        column
        for column in frame.columns
        if column.startswith(IMAGING_PREFIXES)
    ]


@dataclass
class _FeatureGroupScaler:
    columns: list[str]
    fill_values: pd.Series
    scaler: StandardScaler


class FoldFeatureNormalizer:
    """Fit imputers/scalers on one training split and reuse them elsewhere."""

    def __init__(
        self,
        imaging_columns: Sequence[str],
        clinical_columns: Sequence[str] = (),
    ):
        self.imaging_columns = list(imaging_columns)
        self.clinical_columns = list(clinical_columns)
        self._imaging: _FeatureGroupScaler | None = None
        self._clinical: _FeatureGroupScaler | None = None

    @staticmethod
    def _numeric_values(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns required for normalization: {missing[:5]}")
        return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")

    @classmethod
    def _fit_group(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str],
    ) -> _FeatureGroupScaler | None:
        columns = list(columns)
        if not columns:
            return None
        values = cls._numeric_values(frame, columns)
        fill_values = values.median(axis=0).fillna(0.0)
        filled = values.fillna(fill_values)
        scaler = StandardScaler().fit(filled)
        return _FeatureGroupScaler(columns, fill_values, scaler)

    @staticmethod
    def _transform_group(frame: pd.DataFrame, group: _FeatureGroupScaler | None) -> None:
        if group is None:
            return
        values = FoldFeatureNormalizer._numeric_values(frame, group.columns)
        filled = values.fillna(group.fill_values)
        transformed = group.scaler.transform(filled)
        for index, column in enumerate(group.columns):
            frame[column] = transformed[:, index]

    def fit(self, train_frame: pd.DataFrame) -> "FoldFeatureNormalizer":
        self._imaging = self._fit_group(train_frame, self.imaging_columns)
        self._clinical = self._fit_group(train_frame, self.clinical_columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._imaging is None and self.imaging_columns:
            raise RuntimeError("FoldFeatureNormalizer must be fitted before transform")
        transformed = frame.copy()
        self._transform_group(transformed, self._imaging)
        self._transform_group(transformed, self._clinical)
        return transformed

    def fit_transform_many(
        self,
        train_frame: pd.DataFrame,
        *other_frames: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ...]:
        self.fit(train_frame)
        return tuple(self.transform(frame) for frame in (train_frame, *other_frames))

    @property
    def imaging_mean_(self) -> np.ndarray:
        if self._imaging is None:
            raise RuntimeError("Imaging scaler has not been fitted")
        return self._imaging.scaler.mean_

    @property
    def imaging_scale_(self) -> np.ndarray:
        if self._imaging is None:
            raise RuntimeError("Imaging scaler has not been fitted")
        return self._imaging.scaler.scale_
