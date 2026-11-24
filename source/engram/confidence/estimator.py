"""ConfidenceEstimator: combine precision proxies and (optionally) calibrate.

Produces the per-cell precision map consumed by ``EngramMemory.update`` as the
inverse-variance fusion weight (Eq.6'). The estimator is intentionally light: the
proxy weights are scalars (can be learned jointly with the side branch or set by the
calibration fit), and the calibrator is fit offline on held-out reconstruction hits.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .proxies import combine_log_precision
from .calibration import IsotonicCalibrator, TemperatureScaler, reliability_to_precision


class ConfidenceEstimator:
    def __init__(self, weights: Optional[Dict[str, float]] = None, calibrator=None):
        self.weights = weights or {}
        self.calibrator = calibrator

    def raw_log_precision(self, proxies: Dict[str, torch.Tensor]) -> torch.Tensor:
        return combine_log_precision(proxies, self.weights)

    def precision(self, proxies: Dict[str, torch.Tensor], calibrate: bool = True) -> torch.Tensor:
        """Return the per-cell fusion precision map.

        If a calibrator is attached and ``calibrate`` is True, the raw log-precision is
        treated as a logit, mapped to a calibrated reliability, and converted to a
        precision through the odds link. Otherwise the raw precision exp(log p) is used.
        """
        raw = self.raw_log_precision(proxies)
        if self.calibrator is not None and calibrate:
            if isinstance(self.calibrator, TemperatureScaler):
                rel = self.calibrator.transform(raw)
            elif isinstance(self.calibrator, IsotonicCalibrator):
                rel = self.calibrator.transform(raw)
            else:
                raise TypeError("unknown calibrator type")
            return reliability_to_precision(rel)
        return torch.exp(raw)

    def attach_calibrator(self, calibrator):
        self.calibrator = calibrator
        return self
