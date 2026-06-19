# ============================================================
# src/monitoring/drift.py
# ============================================================
# Data drift detection for production monitoring.
#
# What is data drift?
#   Energy consumption patterns change over time:
#     - Season changes (winter baseline is higher than summer)
#     - New industrial tenants (step-change in consumption)
#     - EV adoption (new evening charging peaks)
#
#   If the live data distribution shifts significantly from
#   the training data distribution, model predictions become
#   unreliable — even if no code has changed.
#
# How we detect it:
#   Primary:  Evidently's DataDriftPreset compares statistical
#             properties (mean, std, distribution shape) between
#             a stored reference window and the current window.
#   Fallback: Simple mean-delta check when Evidently is unavailable.
#
# What happens when drift is detected?
#   The /drift endpoint returns drift_detected: true.
#   In a production system, this would trigger a Slack/PagerDuty
#   alert prompting a model retraining run.
# ============================================================

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Compares a reference (training) data window to a current
    production window to detect statistical drift.
    """

    def __init__(self):
        # reference_data stores the training distribution.
        # Set via set_reference() after the model is trained.
        self.reference_data: Optional[pd.DataFrame] = None

    def set_reference(self, readings: List[float]) -> None:
        """
        Store the reference distribution from training data.
        Called once after model initialisation.
        """
        self.reference_data = self._to_dataframe(readings)
        logger.info("drift_reference_set", n_samples=len(readings))

    def _to_dataframe(self, readings: List[float]) -> pd.DataFrame:
        """
        Convert a flat list of readings to a feature DataFrame.

        We compute simple statistical features rather than using
        raw readings — this makes the drift comparison more robust
        to slight timestamp misalignments.
        """
        arr = np.array(readings, dtype=float)
        windows = [arr[max(0, i-24):i+1] for i in range(len(arr))]
        return pd.DataFrame({
            "consumption": arr,
            "rolling_mean": [w.mean() for w in windows],
            "rolling_std": [w.std() if len(w) > 1 else 0 for w in windows],
        })

    def check(self, current_readings: List[float]) -> Dict[str, Any]:
        """
        Compare current readings against the reference distribution.

        Returns a dict with:
          drift_detected: bool — whether significant drift was found
          method:         str  — which detection method was used
          details:        dict — method-specific diagnostic values
        """
        if self.reference_data is None or len(current_readings) < 24:
            return {
                "drift_detected": False,
                "reason": "insufficient_data_for_comparison",
            }

        current_df = self._to_dataframe(current_readings)

        # ── Primary: Evidently DataDriftPreset ────────────────
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            # Build and run the Evidently report
            # reference_data = training distribution
            # current_data   = live production window
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_df,
            )
            result_dict = report.as_dict()

            # Evidently returns a nested dict; drill into the result
            drift_detected = result_dict["metrics"][0]["result"]["dataset_drift"]

            logger.info("drift_check_complete", method="evidently", drift=drift_detected)
            return {
                "drift_detected": drift_detected,
                "method": "evidently",
                "details": {},
            }

        except Exception as e:
            logger.warning("evidently_unavailable_using_fallback", error=str(e))

        # ── Fallback: statistical mean comparison ─────────────
        # If the mean consumption of the current window has
        # shifted by more than 15% from the reference mean,
        # we flag it as potential drift.
        ref_mean = float(self.reference_data["consumption"].mean())
        cur_mean = float(current_df["consumption"].mean())
        ref_std = float(self.reference_data["consumption"].std())

        # Threshold: 1.5 standard deviations from reference mean
        delta = abs(cur_mean - ref_mean)
        threshold = 1.5 * ref_std if ref_std > 0 else ref_mean * 0.15
        drift_detected = delta > threshold

        logger.info(
            "drift_check_complete",
            method="statistical_fallback",
            drift=drift_detected,
            delta=round(delta, 4),
            threshold=round(threshold, 4),
        )
        return {
            "drift_detected": drift_detected,
            "method": "statistical_fallback",
            "details": {
                "reference_mean": round(ref_mean, 4),
                "current_mean": round(cur_mean, 4),
                "delta": round(delta, 4),
                "threshold": round(threshold, 4),
            },
        }


# Module-level singleton — shared across all API requests
_drift_detector = DriftDetector()


def get_drift_detector() -> DriftDetector:
    """Return the shared DriftDetector instance."""
    return _drift_detector
