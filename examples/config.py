"""Benchmark configuration: datasets, tracks, and model registry for change-detection evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from river import metrics, stream
from river.base import DriftDetector
from river.datasets import base
from river.decomposition import OnlineDMD
from river.evaluate import Track
from river.preprocessing import Hankelizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import TYPE_CHECKING

from functions.chdsubid import SubIDChangeDetector
from functions.rolling import Rolling

if TYPE_CHECKING:
    from collections.abc import Iterator

DriftDetector.score_one = DriftDetector.drift_detected  # ty: ignore[unresolved-attribute]


FILE_DIR = str(Path(__file__).resolve().parent)
HN = 80
R = 2
INIT_SIZE = 300
REF_SIZE = 100
TEST_SIZE = 100

N_CHECKPOINTS = 0


class NPDataset(base.Dataset):
    """Base class for numpy array datasets."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Initialize dataset from numpy arrays."""
        self.X = X
        self.y = y
        if self.y is not None:
            n_classes = self.y.shape[1] if len(self.y.shape) > 1 else 1
        else:
            n_classes = None
        super().__init__(
            task=base.REG,
            n_features=self.X.shape[1],
            n_samples=self.X.shape[0],
            n_classes=n_classes,
            n_outputs=None,
            sparse=False,
        )

    def __iter__(self) -> Iterator[tuple[dict, object]]:
        """Iterate over (x, y) sample pairs from the numpy arrays."""
        return stream.iter_array(self.X, self.y)


class ChangeDetectionTrack(Track):
    """Evaluation track that bundles synthetic step-change datasets."""

    def __init__(self) -> None:
        """Initialize the change detection track with synthetic step-change datasets."""
        datasets = [
            NPDataset(
                np.loadtxt(
                    f"{FILE_DIR}/data/synthetic-steps/y{i}.txt",
                ).reshape(-1, 1),
            )
            for i in range(1)
        ]

        super().__init__(
            "Change detection",
            datasets,
            metrics.ROCAUC(),
        )


TRACKS = [ChangeDetectionTrack()]

MODELS = {
    "Change detection": {
        "SubIDChangeDetector": (
            Hankelizer(HN)
            | SubIDChangeDetector(
                Rolling(
                    OnlineDMD(
                        r=R,
                        initialize=INIT_SIZE,
                        w=1.0,
                        exponential_weighting=False,
                    ),
                    301,
                ),
                ref_size=REF_SIZE,
                test_size=TEST_SIZE,
                grace_period=INIT_SIZE + TEST_SIZE + 1,
            )
        ),
    },
}
