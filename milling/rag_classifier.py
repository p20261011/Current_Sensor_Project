"""
milling/rag_classifier.py
Rule-based load classifier for the Milling Machine.

States (calibrated from the observed I_Avg distribution, at the default
20 s analysis window)
-----------------------------------------------------------------------
OFF    – red    : Machine powered down / control circuitry only.
                  I_Avg <= 0.95 A (isolated low cluster, ~3% of windows).
IDLE   – yellow : Spindle/feed energised but not cutting (no-load plateau).
                  0.95 < I_Avg <= 1.20 A (tight cluster around ~1.05 A).
NORMAL – green  : Active material removal at typical load.
                  1.20 < I_Avg < 4.00 A (bulk of the running data, incl.
                  the ramp-up/light-cut transition zone).
PEAK   – pink   : Heavy cut / high load.
                  I_Avg >= 4.00 A (top ~10% of running windows).

Note: these window-level thresholds are tuned for the default 20 s
analysis window. Widening the window compresses the RMS range further
towards the mean, so the thresholds may need adjusting (via the sidebar)
for very different window sizes.

Observed current ranges (single milling data file, 20 s / 50% overlap windows)
--------------------------------------------------------------------------
  I_Avg (windowed RMS): 0.71 – 9.87 A
  Raw per-sample I_Avg : 0.06 – 34 A (peaks flattened out by windowing)

This is a direct per-window load classification with no temporal
smoothing / hysteresis — each window is judged independently on its own
current level, since a genuine load spike (PEAK) is a real signal we
want to see, not noise to be filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass
from .feature_engineering import WindowFeatures

OFF    = "OFF"
IDLE   = "IDLE"
NORMAL = "NORMAL"
PEAK   = "PEAK"

STATE_ORDER  = [OFF, IDLE, NORMAL, PEAK]
STATE_COLORS = {OFF: "#e74c3c", IDLE: "#f1c40f", NORMAL: "#27ae60", PEAK: "#e91e8c"}
STATE_LABELS = {OFF: "OFF", IDLE: "IDLE", NORMAL: "NORMAL LOAD", PEAK: "PEAK LOAD"}

# ── Defaults tuned to observed data ───────────────────────────────────────────
DEFAULT_THRESHOLDS = dict(
    off_max_rms  = 0.95,   # A – I_Avg RMS at/below this → OFF
    idle_max_rms = 1.20,   # A – I_Avg RMS at/below this (and above off_max) → IDLE
    peak_min_rms = 4.00,   # A – I_Avg RMS at/above this → PEAK LOAD
)


@dataclass
class ClassificationResult:
    state: str
    confidence: float
    reason: str
    scores: dict[str, float]


def classify(
    features: WindowFeatures,
    thresholds: dict | None = None,
) -> ClassificationResult:
    """
    Classify one window's features into OFF / IDLE / NORMAL / PEAK load.

    Decision logic (in order):
      1. RMS <= off_max_rms                         → OFF
      2. RMS <= idle_max_rms                         → IDLE
      3. RMS <  peak_min_rms                         → NORMAL LOAD
      4. RMS >= peak_min_rms                         → PEAK LOAD
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    rms = features.rms_i_avg
    var = features.variance_i_avg
    thd = features.thd

    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    scores = dict(
        rms      = round(rms, 4),
        variance = round(var, 5),
        thd      = round(thd, 3),
    )

    off_max  = thr["off_max_rms"]
    idle_max = thr["idle_max_rms"]
    peak_min = thr["peak_min_rms"]

    # ── OFF: no meaningful draw ────────────────────────────────────────────
    if rms <= off_max:
        confidence = clamp(1.0 - rms / max(off_max, 1e-9))
        return ClassificationResult(
            state      = OFF,
            confidence = round(confidence, 3),
            reason     = f"RMS={rms:.4f} A <= {off_max} A (OFF threshold)",
            scores     = scores,
        )

    # ── IDLE: energised, minimal load ──────────────────────────────────────
    if rms <= idle_max:
        span = max(idle_max - off_max, 1e-9)
        confidence = clamp((idle_max - rms) / span)
        return ClassificationResult(
            state      = IDLE,
            confidence = round(confidence, 3),
            reason     = f"RMS={rms:.4f} A in IDLE band ({off_max}, {idle_max}] A",
            scores     = scores,
        )

    # ── PEAK: high / peak load ──────────────────────────────────────────────
    if rms >= peak_min:
        span = max(peak_min * 0.15, 1e-9)
        confidence = clamp((rms - peak_min) / span)
        return ClassificationResult(
            state      = PEAK,
            confidence = round(confidence, 3),
            reason     = f"RMS={rms:.4f} A >= {peak_min} A (PEAK threshold)",
            scores     = scores,
        )

    # ── NORMAL: typical operating load ──────────────────────────────────────
    span   = max(peak_min - idle_max, 1e-9)
    centre = (idle_max + peak_min) / 2.0
    confidence = clamp(1.0 - abs(rms - centre) / (span / 2.0))
    return ClassificationResult(
        state      = NORMAL,
        confidence = round(confidence, 3),
        reason     = f"RMS={rms:.4f} A in NORMAL LOAD band ({idle_max}, {peak_min}) A",
        scores     = scores,
    )
