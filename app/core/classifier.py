# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.config import FEATURE_VERSION
from app.core.features import extract_features, prepare_gray


class SampleModel:
    """Distance classifier trained from operator OK/NG images.

    With few factory samples this is more stable than a deep network, and
    inference is a few milliseconds on a CPU IPC.
    """

    def __init__(self) -> None:
        self.features = np.zeros((0, 1), np.float32)
        self.labels = np.zeros((0,), np.int8)  # 1 = OK, 0 = NG
        self.ok_reference: np.ndarray | None = None
        self.ok_grays: np.ndarray | None = None
        self.ok_radius = 0.35
        self.residual_limit = 0.08
        self.trained_at = ""
        self.ok_count = 0
        self.ng_count = 0
        self.feature_version = FEATURE_VERSION

    @property
    def ready(self) -> bool:
        return self.ok_count > 0 and self.ng_count > 0 and len(self.features) > 0

    def fit(self, items: list[tuple[np.ndarray, str]], roi: dict | None = None) -> dict:
        started = time.perf_counter()
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        ok_grays: list[np.ndarray] = []

        for image, label in items:
            name = label.strip().upper()
            if name not in {"OK", "NG"}:
                continue
            vectors.append(extract_features(image, roi))
            labels.append(1 if name == "OK" else 0)
            if name == "OK":
                ok_grays.append(prepare_gray(image, roi))

        if not vectors:
            raise ValueError("No valid OK/NG samples to train.")

        feats = np.stack(vectors).astype(np.float32)
        labs = np.asarray(labels, np.int8)
        ok_count = int((labs == 1).sum())
        ng_count = int((labs == 0).sum())
        if ok_count < 1 or ng_count < 1:
            raise ValueError("Need at least one OK sample and one NG sample.")

        ok_feats = feats[labs == 1]
        centroid = ok_feats.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
        dists = np.linalg.norm(ok_feats - centroid, axis=1)
        radius = float(np.percentile(dists, 90)) if len(dists) else 0.25
        radius = max(radius * 1.8, 0.12)

        ok_stack = np.stack(ok_grays).astype(np.uint8) if ok_grays else None
        ref = np.mean(ok_stack, axis=0).astype(np.uint8) if ok_stack is not None else None
        residuals = []
        if ok_stack is not None and len(ok_stack) > 1:
            for index, gray in enumerate(ok_stack):
                residuals.append(self._residual(gray, ok_stack, skip=index))
        residual_limit = max(float(np.max(residuals)) * 1.6, 0.06) if residuals else 0.08

        self.features = feats
        self.labels = labs
        self.ok_reference = ref
        self.ok_grays = ok_stack
        self.ok_radius = radius
        self.residual_limit = residual_limit
        self.ok_count = ok_count
        self.ng_count = ng_count
        self.feature_version = FEATURE_VERSION
        self.trained_at = time.strftime("%Y-%m-%d %H:%M:%S")

        correct = 0
        for vec, lab in zip(feats, labs):
            pred, _conf, _dok, _dng = self._score_vector(vec)
            if pred == int(lab):
                correct += 1

        return {
            "ok_count": ok_count,
            "ng_count": ng_count,
            "train_accuracy": correct / len(labs),
            "ok_radius": self.ok_radius,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "trained_at": self.trained_at,
            "feature_version": FEATURE_VERSION,
        }

    def _score_vector(self, vec: np.ndarray) -> tuple[int, float, float, float]:
        ok = self.features[self.labels == 1]
        ng = self.features[self.labels == 0]
        d_ok = float(np.min(np.linalg.norm(ok - vec, axis=1)))
        d_ng = float(np.min(np.linalg.norm(ng - vec, axis=1)))
        # Closer class wins. Confidence is how much closer it is than the other class.
        total = d_ok + d_ng
        closer_ok = d_ok <= d_ng
        confidence = (d_ng / total) if closer_ok else (d_ok / total)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        pred = 1 if closer_ok else 0
        if closer_ok and d_ok > self.ok_radius:
            # Looks like neither trained class well enough. Fail closed as NG.
            pred = 0
            confidence = min(confidence, 0.5)
        return pred, confidence, d_ok, d_ng

    @staticmethod
    def _residual(gray: np.ndarray, ok_grays: np.ndarray, skip: int | None = None) -> float:
        # High percentile so a small defect still stands out from the OK look.
        best = 1.0
        for index, ref in enumerate(ok_grays):
            if skip is not None and index == skip:
                continue
            delta = cv2.blur(cv2.absdiff(gray, ref), (3, 3))
            score = float(delta.max()) / 255.0
            if score < best:
                best = score
        return best

    def predict(self, image: np.ndarray, roi: dict | None = None, min_confidence: float = 0.55) -> dict:
        if not self.ready:
            raise RuntimeError("Model is not trained. Add OK and NG samples, then train.")
        started = time.perf_counter()
        vec = extract_features(image, roi)
        pred, confidence, d_ok, d_ng = self._score_vector(vec)
        residual = 0.0
        if self.ok_grays is not None and len(self.ok_grays):
            residual = self._residual(prepare_gray(image, roi), self.ok_grays)
            if residual > self.residual_limit:
                pred = 0
                confidence = min(confidence, max(0.0, 1.0 - residual))
        label = "OK" if pred == 1 else "NG"
        passed = label == "OK" and confidence >= float(min_confidence)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "label": label,
            "judgment": "OK" if passed else "NG",
            "confidence": confidence,
            "distance_ok": d_ok,
            "distance_ng": d_ng,
            "residual": residual,
            "elapsed_ms": elapsed,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ref = self.ok_reference if self.ok_reference is not None else np.zeros((1, 1), np.uint8)
        grays = self.ok_grays if self.ok_grays is not None else np.zeros((0, 1, 1), np.uint8)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            ok_reference=ref,
            ok_grays=grays,
            ok_radius=np.array([self.ok_radius], np.float32),
            residual_limit=np.array([self.residual_limit], np.float32),
            ok_count=np.array([self.ok_count], np.int32),
            ng_count=np.array([self.ng_count], np.int32),
            feature_version=np.array([self.feature_version], np.int32),
        )
        meta = path.with_suffix(".txt")
        meta.write_text(self.trained_at, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SampleModel":
        model = cls()
        if not path.exists():
            return model
        data = np.load(path, allow_pickle=False)
        model.features = data["features"].astype(np.float32)
        model.labels = data["labels"].astype(np.int8)
        ref = data["ok_reference"]
        model.ok_reference = None if ref.size <= 1 else ref
        grays = data["ok_grays"] if "ok_grays" in data.files else np.zeros((0, 1, 1), np.uint8)
        model.ok_grays = None if grays.size == 0 else grays
        model.ok_radius = float(data["ok_radius"][0])
        if "residual_limit" in data.files:
            model.residual_limit = float(data["residual_limit"][0])
        model.ok_count = int(data["ok_count"][0])
        model.ng_count = int(data["ng_count"][0])
        model.feature_version = int(data["feature_version"][0])
        meta = path.with_suffix(".txt")
        if meta.exists():
            model.trained_at = meta.read_text(encoding="utf-8").strip()
        if model.feature_version != FEATURE_VERSION:
            model.features = np.zeros((0, 1), np.float32)
            model.labels = np.zeros((0,), np.int8)
            model.ok_count = 0
            model.ng_count = 0
        return model


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode preview image.")
    return buf.tobytes()
