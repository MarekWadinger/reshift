import os
import sys

import numpy as np
from river import metrics, stream
from river.base import DriftDetector
from river.datasets import base
from river.decomposition import OnlineDMD
from river.evaluate import Track
from river.preprocessing import Hankelizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from functions.chdsubid import SubIDChangeDetector
from functions.rolling import Rolling

DriftDetector.score_one = DriftDetector.drift_detected  # type: ignore


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
HN = 80
R = 2
INIT_SIZE = 300
REF_SIZE = 100
TEST_SIZE = 100

N_CHECKPOINTS = 0


class NPDataset(base.Dataset):
    """Base class for numpy array datasets."""

    def __init__(self, X, y=None):
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

    def __iter__(self):
        return stream.iter_array(self.X, self.y)


class ChangeDetectionTrack(Track):
    def __init__(self):
        datasets = []
        # datasets = [ds.WaterFlow(), ds.WebTraffic(), ds.synth.AnomalySine()]
        for i in range(1):
            datasets.append(
                NPDataset(
                    np.loadtxt(
                        f"{FILE_DIR}/data/synthetic-steps/y{i}.txt"
                    ).reshape(-1, 1)
                )
            )

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
