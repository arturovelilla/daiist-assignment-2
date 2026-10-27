"""
Decision logic shared by train.py and app.py.

Turns the model's gap detections on each shelf photo into one of three outcomes
(pass / send for human review / send a restock trip) and prices the result.
Both the training pipeline (which sweeps and saves the chosen operating point)
and the dashboard (which recomputes live as sliders move) call these same
functions, so their numbers cannot drift apart.

Definitions
-----------
- Positive photo: the shelf truly shows at least k empty facings ("needs restock").
- Image score: the confidence of the k-th most confident "missing" detection on
  the photo (0 if the model found fewer than k). "score >= t" is exactly
  "at least k gaps detected with confidence >= t", so sweeping t sweeps the
  confidence threshold the brief asks about.
- Two thresholds t_low <= t_high:
    score >= t_high          -> restock (trip sent automatically)
    t_low <= score < t_high  -> human review (manager checks the photo)
    score < t_low            -> pass (nothing happens)
  With t_low == t_high there is no review band: the model decides alone.

Cost per photo
--------------
- Restock: one trip, whether needed (TP) or not (FP).
- Pass on a positive: the missed-restock cost (FN). Pass on a negative: 0 (TN).
- Review: reviewer time, plus a trip if the shelf truly needs it. The reviewer
  is assumed to judge correctly (a stated simplification).
Rates are measured separately on true positives and true negatives of the test
set, then mixed with the base rate, so a production base rate different from
the test set's can be applied without rerunning the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COSTS_FILE = ROOT / "costs.json"

# Confidence grid used for every sweep (0.00, 0.01, ..., 1.00).
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


# ----------------------------------------------------------------------------
# Assumptions
# ----------------------------------------------------------------------------

@dataclass
class Assumptions:
    k_gaps: int
    cost_restock_trip_eur: float
    cost_missed_restock_eur: float
    base_rate: float | None
    photos_per_day: float
    review_seconds_per_photo: float
    reviewer_cost_per_hour_eur: float
    reviewer_minutes_available_per_day: float
    productive_hours_per_person_day: float

    @property
    def cost_review_eur(self) -> float:
        return self.review_seconds_per_photo / 3600 * self.reviewer_cost_per_hour_eur

    @property
    def review_capacity_per_day(self) -> float:
        if self.review_seconds_per_photo <= 0:
            return float("inf")
        return self.reviewer_minutes_available_per_day * 60 / self.review_seconds_per_photo


def load_assumptions(path: Path = COSTS_FILE) -> Assumptions:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Assumptions(**{f.name: raw[f.name] for f in fields(Assumptions)})


# ----------------------------------------------------------------------------
# From detections to an image-level decision
# ----------------------------------------------------------------------------

def image_scores(missing_confs: list[list[float]], k: int) -> np.ndarray:
    """k-th highest 'missing' confidence per photo, 0 if fewer than k detections."""
    k = max(int(k), 1)
    out = np.zeros(len(missing_confs))
    for i, confs in enumerate(missing_confs):
        if len(confs) >= k:
            out[i] = sorted(confs, reverse=True)[k - 1]
    return out


def true_labels(gt_missing: list[int], k: int) -> np.ndarray:
    """True if the photo really shows at least k empty facings."""
    return np.asarray(gt_missing) >= max(int(k), 1)


def effective_base_rate(a: Assumptions, positive: np.ndarray) -> float:
    if a.base_rate is not None:
        return float(a.base_rate)
    return float(positive.mean()) if len(positive) else 0.0


def route(scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Outcome per photo: 0 = pass, 1 = review, 2 = restock."""
    t_low = min(t_low, t_high)
    out = np.zeros(len(scores), dtype=int)
    out[scores >= t_low] = 1
    out[scores >= t_high] = 2
    return out


OUTCOMES = ("pass", "review", "restock")


def evaluate_policy(scores, positive, a: Assumptions, t_low: float, t_high: float) -> dict:
    """Everything the report and dashboard need about one pair of thresholds."""
    outcome = route(scores, t_low, t_high)
    pos, neg = positive, ~positive
    counts = {
        "needs_restock": {o: int(((outcome == i) & pos).sum()) for i, o in enumerate(OUTCOMES)},
        "does_not": {o: int(((outcome == i) & neg).sum()) for i, o in enumerate(OUTCOMES)},
    }
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    r_pos = {o: (counts["needs_restock"][o] / n_pos if n_pos else 0.0) for o in OUTCOMES}
    r_neg = {o: (counts["does_not"][o] / n_neg if n_neg else 0.0) for o in OUTCOMES}

    p = effective_base_rate(a, positive)
    trip, miss, rev = a.cost_restock_trip_eur, a.cost_missed_restock_eur, a.cost_review_eur
    cost_pos = r_pos["restock"] * trip + r_pos["review"] * (rev + trip) + r_pos["pass"] * miss
    cost_neg = r_neg["restock"] * trip + r_neg["review"] * rev
    cost_per_photo = p * cost_pos + (1 - p) * cost_neg

    review_share = p * r_pos["review"] + (1 - p) * r_neg["review"]
    reviews_per_day = review_share * a.photos_per_day
    reviewer_hours = reviews_per_day * a.review_seconds_per_photo / 3600
    return {
        "t_low": float(min(t_low, t_high)),
        "t_high": float(t_high),
        "base_rate": p,
        "counts": counts,
        "n_photos": int(len(scores)),
        "recall_needs_restock": 1 - r_pos["pass"],          # flagged for restock or review
        "auto_restock_share_of_positives": r_pos["restock"],
        "false_trip_share_of_negatives": r_neg["restock"],
        "cost_per_1000_photos": 1000 * cost_per_photo,
        "cost_per_day": cost_per_photo * a.photos_per_day,
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "reviewer_hours_per_day": reviewer_hours,
        "people_for_review": reviewer_hours / a.productive_hours_per_person_day,
        "fits_review_capacity": reviews_per_day <= a.review_capacity_per_day + 1e-9,
    }


# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------

def _share_at_or_above(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """For each threshold in THRESHOLDS, share of the masked photos scoring >= it."""
    s = scores[mask]
    if len(s) == 0:
        return np.zeros(len(THRESHOLDS))
    return (s[None, :] >= THRESHOLDS[:, None] - 1e-12).mean(axis=1)


def sweep_single_threshold(scores, positive, a: Assumptions) -> list[dict]:
    """Model decides alone (no review band): cost at every threshold."""
    rows = []
    for t in THRESHOLDS:
        r = evaluate_policy(scores, positive, a, t, t)
        c = r["counts"]
        rows.append({
            "threshold": float(t),
            "cost_per_1000_photos": r["cost_per_1000_photos"],
            "TP": c["needs_restock"]["restock"], "FN": c["needs_restock"]["pass"],
            "FP": c["does_not"]["restock"], "TN": c["does_not"]["pass"],
            "recall": r["recall_needs_restock"],
        })
    return rows


def best_single_threshold(sweep: list[dict]) -> float:
    """Cheapest threshold. On a flat minimum (common with 146 test photos) take
    the middle of the plateau rather than its edge, so a small shift in the
    data does not push the choice off the minimum."""
    costs = np.array([r["cost_per_1000_photos"] for r in sweep])
    idx = np.flatnonzero(costs <= costs.min() + 1e-9)
    return float(sweep[int(idx[len(idx) // 2])]["threshold"])


def best_band(scores, positive, a: Assumptions) -> dict:
    """Cheapest (t_low, t_high) whose review load fits the reviewer time available.

    Vectorised over all 101 x 101 threshold pairs with t_low <= t_high.
    Ties go to the pair that sends fewer photos to review.
    """
    p = effective_base_rate(a, positive)
    pos_ge = _share_at_or_above(scores, positive)
    neg_ge = _share_at_or_above(scores, ~positive)
    lo, hi = np.meshgrid(np.arange(len(THRESHOLDS)), np.arange(len(THRESHOLDS)), indexing="ij")
    valid = lo <= hi

    pos_restock, neg_restock = pos_ge[hi], neg_ge[hi]
    pos_review, neg_review = pos_ge[lo] - pos_ge[hi], neg_ge[lo] - neg_ge[hi]
    pos_pass = 1 - pos_ge[lo]

    trip, miss, rev = a.cost_restock_trip_eur, a.cost_missed_restock_eur, a.cost_review_eur
    cost = p * (pos_restock * trip + pos_review * (rev + trip) + pos_pass * miss) \
        + (1 - p) * (neg_restock * trip + neg_review * rev)
    review_share = p * pos_review + (1 - p) * neg_review
    feasible = valid & (review_share * a.photos_per_day <= a.review_capacity_per_day + 1e-9)

    cost = np.where(feasible, cost, np.inf)
    best = np.argwhere(cost <= cost.min() + 1e-9)
    i, j = min(best, key=lambda ij: (review_share[ij[0], ij[1]], -ij[0]))
    return {"t_low": float(THRESHOLDS[i]), "t_high": float(THRESHOLDS[j])}


# ----------------------------------------------------------------------------
# Alternatives the brief asks to price against
# ----------------------------------------------------------------------------

def compare_alternatives(scores, positive, a: Assumptions, t_single: float, band: dict) -> list[dict]:
    p = effective_base_rate(a, positive)
    trip, miss, rev = a.cost_restock_trip_eur, a.cost_missed_restock_eur, a.cost_review_eur
    review_all_hours = a.photos_per_day * a.review_seconds_per_photo / 3600

    def row(name, cost_per_photo, reviews_per_day, hours, note):
        return {
            "policy": name,
            "cost_per_1000_photos": 1000 * cost_per_photo,
            "cost_per_day": cost_per_photo * a.photos_per_day,
            "reviews_per_day": reviews_per_day,
            "people_for_review": hours / a.productive_hours_per_person_day,
            "note": note,
        }

    single = evaluate_policy(scores, positive, a, t_single, t_single)
    banded = evaluate_policy(scores, positive, a, band["t_low"], band["t_high"])
    return [
        row("Do nothing (wait for scheduled rounds)", p * miss, 0, 0,
            "Every shelf that needs restocking waits: all positives are missed."),
        row("Person checks every photo", rev + p * trip, a.photos_per_day, review_all_hours,
            "No model. Assumes the reviewer is always right."),
        row("Always send a trip", trip, 0, 0,
            "No model, no review. Every photo triggers a trip."),
        row(f"Model alone (t = {t_single:.2f})", single["cost_per_1000_photos"] / 1000, 0, 0,
            "Single threshold, no human in the loop."),
        row(f"Model + review band ({band['t_low']:.2f} to {band['t_high']:.2f})",
            banded["cost_per_1000_photos"] / 1000, banded["reviews_per_day"],
            banded["reviewer_hours_per_day"], "Chosen design."),
    ]


def load_test_predictions(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "images": [d["image"] for d in data],
        "gt_missing": [d["gt_missing"] for d in data],
        "missing_confs": [d["missing_conf"] for d in data],
    }
