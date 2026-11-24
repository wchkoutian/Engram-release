"""Explicit confidence calibration (Engram Sec 3.4, H2 fix).

Raw proxy precisions are uncalibrated: a "0.9" does not mean the token is reconstructed
correctly 90% of the time. We map raw scores to calibrated reliabilities using either:
  * isotonic regression (non-parametric, monotone) fit by the Pool-Adjacent-Violators
    Algorithm (PAVA) -- implemented here with no sklearn dependency; or
  * temperature/Platt scaling (single scalar T) fit by NLL minimisation.

Both calibrators consume the *raw log-precision* produced by ``ConfidenceEstimator`` as
their score. Temperature scaling treats that log-precision as a logit, so the calibrated
reliability is ``sigmoid(log_p / T)`` and the resulting precision ``r/(1-r) = exp(log_p/T)``
is just the raw precision raised to the power ``1/T`` -- a monotone, single-parameter
recalibration. Isotonic regression fits an arbitrary monotone score->reliability map.

The calibrator is fit on held-out (raw_score, hit) pairs, where ``hit`` is 1 if the
decoded token error is below a threshold.
At inference, calibrated reliability r is turned into a fusion precision via the odds
link precision = r / (1 - r).
"""
from __future__ import annotations

import numpy as np
import torch


RELIABILITY_CLAMP_EPS = 1e-4


def _pava(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Pool-Adjacent-Violators on points sorted by x. Returns (x_thresh, y_fit)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if not x.size or x.shape != y.shape or x.shape != w.shape:
        raise ValueError("isotonic inputs must be non-empty arrays with matching shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("isotonic inputs must be finite")
    if np.any(w <= 0):
        raise ValueError("isotonic weights must be positive")

    order = np.argsort(x, kind="mergesort")
    x_s, y_s, w_s = x[order], y[order], w[order]
    unique_x, inverse = np.unique(x_s, return_inverse=True)
    unique_w = np.zeros(unique_x.shape, dtype=np.float64)
    unique_yw = np.zeros(unique_x.shape, dtype=np.float64)
    np.add.at(unique_w, inverse, w_s)
    np.add.at(unique_yw, inverse, y_s * w_s)
    unique_y = unique_yw / unique_w

    block_val = []
    block_w = []
    block_start = []
    block_end = []
    for index, value in enumerate(unique_y):
        block_val.append(float(value))
        block_w.append(float(unique_w[index]))
        block_start.append(index)
        block_end.append(index)
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            v2, w2 = block_val.pop(), block_w.pop()
            v1, w1 = block_val.pop(), block_w.pop()
            end = block_end.pop()
            block_end.pop()
            block_start.pop()
            start = block_start.pop()
            merged = (v1 * w1 + v2 * w2) / (w1 + w2)
            block_val.append(merged)
            block_w.append(w1 + w2)
            block_start.append(start)
            block_end.append(end)

    thresholds = []
    fitted = []
    for start, end, value in zip(block_start, block_end, block_val):
        thresholds.append(unique_x[start])
        fitted.append(value)
        if end != start:
            thresholds.append(unique_x[end])
            fitted.append(value)
    return np.asarray(thresholds), np.asarray(fitted)


class IsotonicCalibrator:
    def __init__(self):
        self.x_thresh = None
        self.y_fit = None

    def fit(self, scores: torch.Tensor, hits: torch.Tensor, weights: torch.Tensor | None = None):
        x = scores.detach().cpu().numpy().astype(np.float64).reshape(-1)
        y = hits.detach().cpu().numpy().astype(np.float64).reshape(-1)
        w = (weights.detach().cpu().numpy().astype(np.float64).reshape(-1)
             if weights is not None else np.ones_like(x))
        xs, ys = _pava(x, y, w)
        self.x_thresh = xs
        self.y_fit = np.clip(ys, 0.0, 1.0)
        return self

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        assert self.x_thresh is not None, "fit before transform"
        s = scores.detach().cpu().numpy().reshape(-1)
        r = np.interp(s, self.x_thresh, self.y_fit, left=self.y_fit[0], right=self.y_fit[-1])
        out = torch.from_numpy(r).to(scores.device, scores.dtype).view_as(scores)
        return out


class TemperatureScaler:
    """logits -> sigmoid(logits / T). T fit by minimising binary NLL."""

    def __init__(self, init_T: float = 1.0):
        self.log_T = torch.nn.Parameter(torch.tensor(float(np.log(init_T))))

    def fit(self, logits: torch.Tensor, hits: torch.Tensor, max_iter: int = 200, lr: float = 0.05):
        logits = logits.detach().float().reshape(-1)
        hits = hits.detach().float().reshape(-1)
        opt = torch.optim.Adam([self.log_T], lr=lr)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(max_iter):
            opt.zero_grad()
            T = torch.exp(self.log_T)
            loss = bce(logits / T, hits)
            loss.backward()
            opt.step()
        return self

    @property
    def T(self) -> float:
        return float(torch.exp(self.log_T.detach()))

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T.detach()).to(logits.device, logits.dtype)
        return torch.sigmoid(logits / T)


def reliability_to_precision(
    rel: torch.Tensor, eps: float = RELIABILITY_CLAMP_EPS
) -> torch.Tensor:
    """Odds link: precision = r / (1 - r), used as the inverse-variance fusion weight."""
    r = rel.clamp(eps, 1.0 - eps)
    return r / (1.0 - r)


def save_calibrator(calibrator, path: str, *, metadata: dict) -> None:
    if isinstance(calibrator, IsotonicCalibrator):
        payload = {"type": "isotonic", "x_thresh": np.asarray(calibrator.x_thresh),
                   "y_fit": np.asarray(calibrator.y_fit)}
    elif isinstance(calibrator, TemperatureScaler):
        payload = {"type": "temperature", "log_T": float(calibrator.log_T.detach())}
    else:
        raise TypeError(f"unknown calibrator {type(calibrator)}")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("calibrator metadata is required")
    if metadata.get("method") != payload["type"]:
        raise ValueError(
            f"calibrator metadata method {metadata.get('method')!r} does not match "
            f"payload type {payload['type']!r}"
        )
    payload["metadata"] = metadata
    torch.save(payload, path)


def _metadata_contains(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _metadata_contains(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def load_calibrator(path: str, *, expected_metadata: dict):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata")
    if metadata is None:
        raise ValueError(
            f"{path} has no calibrator provenance and is treated as a legacy artifact"
        )
    if not _metadata_contains(metadata, expected_metadata):
        raise ValueError(
            f"{path} calibrator provenance does not match the active checkpoint/config: "
            f"actual={metadata!r}, expected={expected_metadata!r}"
        )
    if payload["type"] == "isotonic":
        c = IsotonicCalibrator()
        c.x_thresh = np.asarray(payload["x_thresh"])
        c.y_fit = np.asarray(payload["y_fit"])
        c.metadata = dict(metadata)
        return c
    if payload["type"] == "temperature":
        c = TemperatureScaler()
        with torch.no_grad():
            c.log_T.copy_(torch.tensor(float(payload["log_T"])))
        c.metadata = dict(metadata)
        return c
    raise ValueError(f"unknown calibrator payload {payload.get('type')}")
