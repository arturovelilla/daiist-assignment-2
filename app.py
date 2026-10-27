"""
Assignment 2: Gradio dashboard for shelf-gap triage.

    uv run python main.py app

Loads the committed weights in weights/ and the outputs the training pipeline
saved in artifacts/. Never trains anything.

Tabs
- Check a shelf photo: upload a photo, see detected gaps and the pass / review /
  restock decision at the current thresholds.
- Costs and review band: move the two thresholds (and, if you want, the
  assumptions) and watch the outcome table, cost per 1,000 photos and review
  share update; cost-against-threshold curve with the chosen point marked.
- Model performance: per-class results for the finetuned model and both
  baselines, and the box-level confusion matrix.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import shelf_policy as sp

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("SHELF_WEIGHTS", ROOT / "weights" / "shelf_gaps_yolo11s.pt"))
ARTIFACTS = Path(os.environ.get("SHELF_ARTIFACTS", ROOT / "artifacts"))
EXAMPLES = Path(os.environ.get("SHELF_EXAMPLES", ROOT / "examples"))
CLASSES = ["missing", "product"]
IMG_SIZE = 640          # training images were stretched to 640x640; uploads get the same treatment


def _load_json(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


METRICS = _load_json("metrics.json")
BOX_CM = _load_json("box_confusion.json")
OP = _load_json("operating_point.json")
TRAIN_INFO = _load_json("training_info.json") or {}
DATA_STATS = _load_json("dataset_stats.json") or {}
PREDS = sp.load_test_predictions(ARTIFACTS / "test_predictions.json") \
    if (ARTIFACTS / "test_predictions.json").exists() else None
# Validation photos: where thresholds are chosen. Costs shown are always on test.
VALID = sp.load_test_predictions(ARTIFACTS / "valid_predictions.json") \
    if (ARTIFACTS / "valid_predictions.json").exists() else None
DEFAULTS = sp.load_assumptions()
COSTS_RAW = json.loads((ROOT / "costs.json").read_text(encoding="utf-8"))
IS_PLACEHOLDER = "PLACEHOLDER" in str(COSTS_RAW.get("_status", ""))

SAVED_BAND = OP["band"] if OP else {"t_low": 0.2, "t_high": 0.5}
SAVED_SINGLE = OP["single_threshold"] if OP else None
DEFAULT_BASE = OP["base_rate_used"] if OP else (DEFAULTS.base_rate or 0.5)

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(WEIGHTS))
    return _MODEL


# ----------------------------------------------------------------------------
# Tab 1: single photo
# ----------------------------------------------------------------------------

def detect(image: np.ndarray | None):
    """Run the model once per upload; slider moves re-render from the stored result."""
    if image is None:
        return None
    if image.ndim == 2:                       # greyscale -> 3 channels
        image = np.stack([image] * 3, axis=-1)
    image = np.ascontiguousarray(image[..., :3])   # drop a PNG alpha channel if present
    h, w = image.shape[:2]
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    r = model().predict(resized, imgsz=IMG_SIZE, conf=0.001, iou=0.7, max_det=1000, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    boxes[:, [0, 2]] *= w / IMG_SIZE
    boxes[:, [1, 3]] *= h / IMG_SIZE
    return {"image": image, "boxes": boxes, "cls": r.boxes.cls.cpu().numpy().astype(int),
            "conf": r.boxes.conf.cpu().numpy()}


def analyze(image, t_low, t_high, k, show_products):
    """Upload handler: detect + render in one step, so the result panel shows a
    loading indicator while the model runs."""
    det = detect(image)
    img, md = render(det, t_low, t_high, k, show_products)
    return det, img, md


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render(det, t_low, t_high, k, show_products):
    if det is None:
        return None, "Upload a shelf photo, or pick one of the examples below."
    t_low = min(t_low, t_high)
    k = max(int(k), 1)
    img = det["image"].copy()
    thick = max(1, round(max(img.shape[:2]) / 500))
    miss = det["cls"] == CLASSES.index("missing")
    if show_products:
        for b, c in zip(det["boxes"][~miss], det["conf"][~miss]):
            if c >= 0.25:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (160, 160, 160), 1)
    gap_conf = np.sort(det["conf"][miss])[::-1]
    for b, c in zip(det["boxes"][miss], det["conf"][miss]):
        if c >= t_low:
            color = (220, 38, 38) if c >= t_high else (245, 158, 11)
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), color, thick + 1)
    score = float(gap_conf[k - 1]) if len(gap_conf) >= k else 0.0
    outcome = sp.OUTCOMES[sp.route(np.array([score]), t_low, t_high)[0]]
    label = {"pass": "PASS: no action", "review": "SEND TO REVIEW: manager checks the photo",
             "restock": "RESTOCK: send an associate"}[outcome]
    md = (
        f"### {label}\n"
        f"- Gaps detected at confidence >= {t_high:.2f} (red): **{int((gap_conf >= t_high).sum())}**\n"
        f"- Gaps detected at confidence >= {t_low:.2f} (red + amber): **{int((gap_conf >= t_low).sum())}**\n"
        f"- Photo score = confidence of the {_ordinal(k)} most confident gap: **{score:.3f}**\n"
        f"- Rule: score >= {t_high:.2f} restock, {t_low:.2f} to {t_high:.2f} review, below {t_low:.2f} pass."
    )
    return img, md


# ----------------------------------------------------------------------------
# Tab 2: costs and review band
# ----------------------------------------------------------------------------

def _assumptions(k, trip, miss, base, photos, rsec, rcost, rmin) -> sp.Assumptions:
    return sp.Assumptions(
        k_gaps=int(k), cost_restock_trip_eur=float(trip), cost_missed_restock_eur=float(miss),
        base_rate=float(base), photos_per_day=float(photos), review_seconds_per_photo=float(rsec),
        reviewer_cost_per_hour_eur=float(rcost), reviewer_minutes_available_per_day=float(rmin),
        productive_hours_per_person_day=DEFAULTS.productive_hours_per_person_day,
    )


def policy_view(t_low, t_high, k, trip, miss, base, photos, rsec, rcost, rmin):
    if PREDS is None:
        msg = "No artifacts/test_predictions.json yet: run the training pipeline first."
        return msg, None, None, None
    a = _assumptions(k, trip, miss, base, photos, rsec, rcost, rmin)
    scores = sp.image_scores(PREDS["missing_confs"], a.k_gaps)
    positive = sp.true_labels(PREDS["gt_missing"], a.k_gaps)
    swapped = t_low > t_high
    t_low = min(t_low, t_high)
    r = sp.evaluate_policy(scores, positive, a, t_low, t_high)
    sweep = sp.sweep_single_threshold(scores, positive, a)
    sweep_val = None
    if VALID is not None:          # choose on validation, cost on test
        v_scores = sp.image_scores(VALID["missing_confs"], a.k_gaps)
        v_pos = sp.true_labels(VALID["gt_missing"], a.k_gaps)
        sweep_val = sp.sweep_single_threshold(v_scores, v_pos, a)
        t_best = sp.best_single_threshold(sweep_val)
    else:
        t_best = sp.best_single_threshold(sweep)
    alts = sp.compare_alternatives(scores, positive, a, t_best, {"t_low": t_low, "t_high": t_high})

    cap = a.review_capacity_per_day
    kpis = (
        f"### {r['cost_per_1000_photos']:,.0f} EUR per 1,000 photos  ·  "
        f"{r['review_share']:.1%} sent to review\n"
        f"- {r['reviews_per_day']:,.0f} reviews/day vs capacity {cap:,.0f}/day "
        f"({'fits' if r['fits_review_capacity'] else '**exceeds capacity**'}), "
        f"{r['reviewer_hours_per_day']:.1f} reviewer hours/day = {r['people_for_review']:.2f} people\n"
        f"- Shelves needing restock that get acted on (restock or review): {r['recall_needs_restock']:.1%}\n"
        f"- Expected cost per day: {r['cost_per_day']:,.0f} EUR at {a.photos_per_day:,.0f} photos/day\n"
        f"- Costs measured on {r['n_photos']} unique test photos ({int(positive.sum())} need restocking at "
        f"k = {a.k_gaps}), weighted to a base rate of {a.base_rate:.1%}. "
        f"{'Thresholds are chosen on the validation photos.' if VALID is not None else ''}"
    )
    if swapped:
        kpis += "\n\n*Low edge was above the high edge; treated as equal (no review band).*"

    c = r["counts"]
    table = pd.DataFrame(
        [[c["needs_restock"]["pass"], c["needs_restock"]["review"], c["needs_restock"]["restock"]],
         [c["does_not"]["pass"], c["does_not"]["review"], c["does_not"]["restock"]]],
        index=[f"Needs restock (>= {a.k_gaps} gaps)", "Does not"],
        columns=["Pass", "Review", "Restock"],
    ).reset_index(names="Actual / Decision")

    t = [s["threshold"] for s in sweep]
    cost = [s["cost_per_1000_photos"] for s in sweep]
    ymax = max(max(cost), alts[0]["cost_per_1000_photos"],
               max(x["cost_per_1000_photos"] for x in sweep_val) if sweep_val else 0) * 1.05
    fig = go.Figure()
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="#f59e0b", opacity=0.18, line_width=0)
    fig.add_trace(go.Scatter(x=t, y=cost, mode="lines", name="Model alone: cost at each threshold (test)",
                             line=dict(color="#2563eb", width=3)))
    if sweep_val is not None:
        fig.add_trace(go.Scatter(x=t, y=[s_["cost_per_1000_photos"] for s_ in sweep_val], mode="lines",
                                 name="Same on validation (used to choose)",
                                 line=dict(color="#2563eb", width=1.5, dash="dash"), opacity=0.6))

    def vline(x, name, color, dash):
        fig.add_trace(go.Scatter(x=[x, x], y=[0, ymax], mode="lines", name=name,
                                 line=dict(color=color, dash=dash, width=2)))

    def hline(y, name, color):
        fig.add_trace(go.Scatter(x=[0, 1], y=[y, y], mode="lines", name=name,
                                 line=dict(color=color, dash="dot", width=2)))

    vline(t_best, f"Cheapest threshold on validation with these assumptions ({t_best:.2f})", "#dc2626", "dash")
    if SAVED_SINGLE is not None:
        vline(SAVED_SINGLE, f"Threshold chosen by the pipeline ({SAVED_SINGLE:.2f})", "#111827", "dot")
    hline(r["cost_per_1000_photos"], f"Current sliders with review band ({t_low:.2f} to {t_high:.2f}, shaded)", "#f59e0b")
    hline(alts[0]["cost_per_1000_photos"], "Do nothing", "#9ca3af")
    fig.update_layout(xaxis_title="Confidence threshold (k-th most confident gap)",
                      yaxis_title="Expected cost per 1,000 photos (EUR)", height=460,
                      xaxis_range=[0, 1], yaxis_range=[0, ymax],
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))

    alt_df = pd.DataFrame([{
        "Policy": x["policy"].replace(f"({t_low:.2f} to {t_high:.2f})", "(current sliders)"),
        "EUR per 1,000 photos": round(x["cost_per_1000_photos"]),
        "EUR per day": round(x["cost_per_day"]),
        "Reviews per day": round(x["reviews_per_day"]),
        "People for review": round(x["people_for_review"], 2),
    } for x in alts])
    return kpis, table, fig, alt_df


# ----------------------------------------------------------------------------
# Tab 3: model performance
# ----------------------------------------------------------------------------

def performance_table() -> pd.DataFrame:
    if METRICS is None:
        return pd.DataFrame({"info": ["Run the training pipeline to produce artifacts/metrics.json"]})
    nice = {"finetuned_yolo11s": "Finetuned yolo11s", "coco_zero_shot_yolo11s": "COCO yolo11s zero-shot",
            "predict_nothing": "Predict nothing"}
    rows = []
    for key, m in METRICS["models"].items():
        for cls, v in m["per_class"].items():
            rows.append({"Model": nice.get(key, key), "Class": cls + (" (positive)" if cls == "missing" else ""),
                         "AP50": round(v["ap50"], 3), "AP50-95": round(v["ap50_95"], 3),
                         "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                         "GT boxes": v["n_gt"]})
        o = m["overall"]
        rows.append({"Model": nice.get(key, key), "Class": "all (mean)", "AP50": round(o["map50"], 3),
                     "AP50-95": round(o["map50_95"], 3), "Precision": round(o["precision"], 3),
                     "Recall": round(o["recall"], 3), "GT boxes": None})
    df = pd.DataFrame(rows)
    df["GT boxes"] = df["GT boxes"].astype("Int64")
    return df


def confusion_figure():
    if BOX_CM is None:
        return None
    m = np.array(BOX_CM["matrix"])
    labels = BOX_CM["labels"]
    row_tot = m.sum(axis=1, keepdims=True)
    pct = np.divide(m, row_tot, out=np.zeros_like(m, dtype=float), where=row_tot > 0)
    text = [[f"{m[i, j]:,}<br>({pct[i, j]:.0%} of row)" for j in range(m.shape[1])] for i in range(m.shape[0])]
    text[-1][-1] = "n/a<br>(no true negatives in detection)"
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed",
                      height=420, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Box-level confusion matrix (conf >= {BOX_CM['conf']}, IoU >= {BOX_CM['iou']})")
    return fig


def info_markdown() -> str:
    parts = []
    if METRICS:
        parts.append(f"Evaluated on **{METRICS['test_photos']} unique test photos** (exact duplicates removed). "
                     f"Positive class: **{METRICS['positive_class']}**. Precision/recall at each class's best-F1 confidence.")
    if TRAIN_INFO:
        a = TRAIN_INFO.get("train_args", {})
        parts.append(f"Trained: {TRAIN_INFO.get('base_model')} on {TRAIN_INFO.get('device')}, "
                     f"{TRAIN_INFO.get('epochs_completed')} epochs (max {a.get('epochs')}), imgsz {a.get('imgsz')}, "
                     f"batch {a.get('batch')}, {TRAIN_INFO.get('train_minutes')} min. Dataset {TRAIN_INFO.get('dataset')}.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

with gr.Blocks(title="Shelf gap triage") as demo:
    gr.Markdown(
        "# Shelf gap triage\n"
        "Fixed cameras photograph each shelf section. The model finds empty facings; a photo with at least "
        "**k gaps** needs a restock trip. Two thresholds split photos into **pass**, **review** by the shift "
        "manager, and **restock**."
    )
    if IS_PLACEHOLDER:
        gr.Markdown("> **Costs in costs.json are placeholders.** Numbers on this page are illustrative "
                    "until they are replaced with justified figures.")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=SAVED_BAND["t_low"], step=0.01, label="Pass | review edge (t_low)")
        t_high = gr.Slider(0, 1, value=SAVED_BAND["t_high"], step=0.01, label="Review | restock edge (t_high)")
        reset = gr.Button("Reset to chosen band", scale=0)

    with gr.Tabs():
        with gr.Tab("Check a shelf photo"):
            det_state = gr.State(None)
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Shelf photo")
                img_out = gr.Image(type="numpy", label="Detected gaps (red >= t_high, amber in review band)")
            show_products = gr.Checkbox(False, label="Also draw detected products (conf >= 0.25)")
            verdict = gr.Markdown()
            examples = sorted(str(p) for p in EXAMPLES.glob("*.jpg")) if EXAMPLES.exists() else []
            if examples:
                gr.Examples(examples=examples, inputs=img_in, label="Test-set photos")

        with gr.Tab("Costs and review band"):
            with gr.Accordion("Assumptions (defaults from costs.json; changes here are not saved)", open=False):
                with gr.Row():
                    k_in = gr.Number(DEFAULTS.k_gaps, precision=0, minimum=1, label="k: gaps that make a shelf need restocking")
                    trip_in = gr.Number(DEFAULTS.cost_restock_trip_eur, label="Restock trip cost (EUR) [FP and TP]")
                    miss_in = gr.Number(DEFAULTS.cost_missed_restock_eur, label="Missed restock cost (EUR) [FN]")
                    base_in = gr.Number(round(DEFAULT_BASE, 3), minimum=0, maximum=1, label="Base rate (share needing restock)")
                with gr.Row():
                    photos_in = gr.Number(DEFAULTS.photos_per_day, label="Photos per day")
                    rsec_in = gr.Number(DEFAULTS.review_seconds_per_photo, label="Seconds per review")
                    rcost_in = gr.Number(DEFAULTS.reviewer_cost_per_hour_eur, label="Reviewer cost per hour (EUR)")
                    rmin_in = gr.Number(DEFAULTS.reviewer_minutes_available_per_day, label="Review minutes available per day")
            kpis = gr.Markdown()
            outcome_table = gr.Dataframe(label="Unique test photos by actual need and decision", interactive=False)
            curve = gr.Plot(label="Cost against threshold")
            alt_table = gr.Dataframe(label="Priced against the alternatives", interactive=False)

        with gr.Tab("Model performance"):
            gr.Markdown(info_markdown())
            gr.Dataframe(value=performance_table(), label="Per-class results on the test split", interactive=False)
            gr.Plot(value=confusion_figure())
            if (ARTIFACTS / "training_curves.png").exists():
                gr.Image(str(ARTIFACTS / "training_curves.png"), label="Training curves", interactive=False)

    policy_inputs = [t_low, t_high, k_in, trip_in, miss_in, base_in, photos_in, rsec_in, rcost_in, rmin_in]
    policy_outputs = [kpis, outcome_table, curve, alt_table]
    render_inputs = [det_state, t_low, t_high, k_in, show_products]

    img_in.change(analyze, [img_in, t_low, t_high, k_in, show_products], [det_state, img_out, verdict])
    for comp in (t_low, t_high, k_in, show_products):
        comp.change(render, render_inputs, [img_out, verdict])
    for comp in policy_inputs:
        comp.change(policy_view, policy_inputs, policy_outputs)
    reset.click(lambda: (SAVED_BAND["t_low"], SAVED_BAND["t_high"]), None, [t_low, t_high])
    demo.load(policy_view, policy_inputs, policy_outputs)


if WEIGHTS.exists():
    # Load the model and run it once on a blank image at startup, so the first
    # real photo is answered in about a second instead of 10-30 s on a CPU.
    model().predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), imgsz=IMG_SIZE, verbose=False)


if __name__ == "__main__":
    # Open the dashboard in the default browser, except on automated runners (CI) with no browser.
    demo.launch(inbrowser=not os.environ.get("CI"))
