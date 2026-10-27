"""
Assignment 2: training pipeline.

    uv run python main.py train          # full pipeline (what grading runs)
    python train.py --eval-only          # re-evaluate the saved weights, no training
    python train.py --policy-only        # recompute costs/thresholds from saved predictions (seconds)
    python train.py --valid-only         # predict on the validation photos with the saved weights, then --policy-only
    python train.py --smoke              # tiny CPU run to check the code works end to end

Stages
1. Download Parth "Empty shelf detection" v3 from Roboflow Universe.
2. Build a clean copy: keep the two real classes (missing, product) and drop the
   empty "-" class (1 box in the whole dataset); remove exact duplicate images
   from valid and test (test: 403 files -> 146 unique photos). Train is kept as
   downloaded.
3. Finetune yolo11s from its COCO checkpoint. Best weights -> weights/.
4. Evaluate on the deduplicated test split: finetuned model, COCO zero-shot
   yolo11s, and a predict-nothing baseline. Per-class metrics + confusion matrix.
5. Image-level policy: score every photo, sweep the confidence threshold
   against the costs in costs.json, and pick the cheapest threshold and the
   cheapest pass/review/restock band that fits the review time available.
   Thresholds are CHOSEN on the validation photos and their cost is REPORTED on
   the test photos, so the reported cost is not tuned on the data it is
   measured on.
6. Save everything app.py needs to artifacts/, and bundle weights + artifacts
   into results.zip (handy when training on Colab).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

import shelf_policy as sp
from box_metrics import confusion_matrix, evaluate

ROOT = Path(__file__).resolve().parent

# Dataset (Roboflow Universe, public, used as published)
WORKSPACE = "parth-vfviv"
PROJECT = "empty-shelf-detection-nipzt"
VERSION = 3
RAW_DIR = ROOT / "datasets" / "parth-v3"
CLEAN_DIR = ROOT / "datasets" / "parth-v3-clean"
CLASSES = ["missing", "product"]           # positive class for the business problem: missing
POSITIVE_CLASS = "missing"

# Training settings (recorded in artifacts/training_info.json)
BASE_MODEL = "yolo11s.pt"
TRAIN_ARGS = dict(
    epochs=60,
    patience=15,          # stop if validation fitness has not improved for 15 epochs
    imgsz=640,            # images were already resized to 640x640 by Roboflow; bigger only upsamples
    batch=16,
    max_det=1000,         # photos hold up to ~400 boxes; the default 300 would silently drop some
    seed=0,
    deterministic=True,
    cache="ram",
    workers=2,
    plots=True,
)
PRED_ARGS = dict(imgsz=640, conf=0.001, iou=0.7, max_det=1000, verbose=False)


def out_dirs(smoke: bool) -> tuple[Path, Path]:
    base = ROOT / "smoke_output" if smoke else ROOT
    return base / "weights", base / "artifacts"


# ----------------------------------------------------------------------------
# 1-2. Data
# ----------------------------------------------------------------------------

def download_dataset() -> None:
    if (RAW_DIR / "data.yaml").exists():
        print(f"Dataset already present at {RAW_DIR.relative_to(ROOT)}, skipping download.")
        return
    from roboflow import Roboflow

    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(RAW_DIR), overwrite=False
    )


def _raw_class_names() -> list[str]:
    for line in (RAW_DIR / "data.yaml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            inner = line.split(":", 1)[1].strip().strip("[]")
            return [n.strip().strip("'\"") for n in inner.split(",")]
    raise RuntimeError("Could not read class names from data.yaml")


def build_clean_dataset() -> dict:
    """Two-class copy with deduplicated valid/test. Returns dataset statistics."""
    raw_names = _raw_class_names()
    remap = {i: CLASSES.index(n) for i, n in enumerate(raw_names) if n in CLASSES}
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    stats = {"raw_class_names": raw_names, "dropped_boxes": 0, "splits": {}}

    for split in ("train", "valid", "test"):
        src_img = RAW_DIR / split / "images"
        dst_img, dst_lbl = CLEAN_DIR / split / "images", CLEAN_DIR / split / "labels"
        dst_img.mkdir(parents=True)
        dst_lbl.mkdir(parents=True)
        seen, files, kept = set(), 0, 0
        inst = {c: 0 for c in CLASSES}
        per_image_missing, box_area = [], {c: [] for c in CLASSES}
        for img in sorted(src_img.iterdir()):
            files += 1
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if split != "train" and digest in seen:
                continue                        # exact duplicate: keep one copy
            seen.add(digest)
            kept += 1
            lines, n_missing = [], 0
            lbl = RAW_DIR / split / "labels" / f"{img.stem}.txt"
            text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                if c not in remap:
                    stats["dropped_boxes"] += 1
                    continue
                name = CLASSES[remap[c]]
                inst[name] += 1
                box_area[name].append(float(parts[3]) * float(parts[4]))
                n_missing += name == "missing"
                lines.append(" ".join([str(remap[c])] + parts[1:5]))
            per_image_missing.append(n_missing)
            shutil.copy2(img, dst_img / img.name)
            (dst_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                    encoding="utf-8")
        pim = np.array(per_image_missing)
        stats["splits"][split] = {
            "files": files,
            "unique_images_kept": kept,
            "instances": inst,
            "images_with_at_least_one_gap": int((pim > 0).sum()),
            "gaps_per_image_median": float(np.median(pim)) if len(pim) else 0.0,
            "gaps_per_image_p90": float(np.percentile(pim, 90)) if len(pim) else 0.0,
            "box_area_pct_of_image_median": {
                c: (100 * float(np.median(v)) if v else None) for c, v in box_area.items()
            },
        }
    (CLEAN_DIR / "data.yaml").write_text(
        f"path: {CLEAN_DIR.as_posix()}\ntrain: train/images\nval: valid/images\n"
        f"test: test/images\nnc: {len(CLASSES)}\nnames: {CLASSES}\n",
        encoding="utf-8",
    )
    return stats


# ----------------------------------------------------------------------------
# 3. Train
# ----------------------------------------------------------------------------

def train(weights_dir: Path, smoke: bool) -> dict:
    from ultralytics import YOLO

    args = dict(TRAIN_ARGS)
    base = BASE_MODEL
    if smoke:
        base = "yolo11n.pt"
        args.update(epochs=1, imgsz=320, batch=4, cache=False, workers=0, plots=False)
    model = YOLO(base)
    started = time.time()
    model.train(data=str(CLEAN_DIR / "data.yaml"), project=str(ROOT / "runs"),
                name="smoke" if smoke else "shelf", exist_ok=True, **args)
    run_dir = Path(model.trainer.save_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    final = weights_dir / "shelf_gaps_yolo11s.pt"
    shutil.copy2(run_dir / "weights" / "best.pt", final)

    import torch
    info = {
        "base_model": base,
        "train_args": args,
        "epochs_completed": int(getattr(model.trainer, "epoch", -1)) + 1,
        "train_minutes": round((time.time() - started) / 60, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
        "run_dir": str(run_dir),
        "weights": final.relative_to(ROOT).as_posix(),
    }
    return info


# ----------------------------------------------------------------------------
# 4. Evaluate
# ----------------------------------------------------------------------------

def _load_ground_truth(split_dir: Path, limit: int | None) -> list[dict]:
    from PIL import Image

    items = []
    for img in sorted((split_dir / "images").iterdir())[:limit]:
        w, h = Image.open(img).size
        rows = [l.split() for l in (split_dir / "labels" / f"{img.stem}.txt").read_text().splitlines() if l.split()]
        arr = np.array([[float(x) for x in r[:5]] for r in rows]).reshape(-1, 5)
        cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
        items.append({
            "path": img,
            "gt_cls": arr[:, 0].astype(int),
            "gt_box": np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1),
        })
    return items


def _predict(model_path: str, items: list[dict], class_map) -> list[dict]:
    """Run a model over the test photos. class_map(model_class_id) -> our class id or None."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    out = []
    for item in items:                      # one photo at a time keeps memory flat on small machines
        r = model.predict(str(item["path"]), **PRED_ARGS)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        mapped = np.array([class_map(c) for c in cls], dtype=object)
        keep = np.array([m is not None for m in mapped], dtype=bool)
        out.append({
            **item,
            "pred_cls": mapped[keep].astype(int) if keep.any() else np.zeros(0, int),
            "pred_conf": r.boxes.conf.cpu().numpy()[keep],
            "pred_box": r.boxes.xyxy.cpu().numpy()[keep].reshape(-1, 4),
        })
    return out


def _empty_predictions(items: list[dict]) -> list[dict]:
    return [{**i, "pred_cls": np.zeros(0, int), "pred_conf": np.zeros(0),
             "pred_box": np.zeros((0, 4))} for i in items]


def evaluate_models(weights_path: Path, artifacts: Path, smoke: bool) -> dict:
    items = _load_ground_truth(CLEAN_DIR / "test", limit=6 if smoke else None)
    print(f"Evaluating on {len(items)} unique test photos ...")

    finetuned = _predict(str(weights_path), items, lambda c: int(c))
    # COCO has no "empty shelf" class: map every COCO detection to "product"
    # (the most generous reading) and nothing can ever become "missing".
    coco = _predict("yolo11n.pt" if smoke else BASE_MODEL, items, lambda c: CLASSES.index("product"))
    nothing = _empty_predictions(items)

    metrics = {
        "test_photos": len(items),
        "positive_class": POSITIVE_CLASS,
        "models": {
            "finetuned_yolo11s": evaluate(finetuned, CLASSES),
            "coco_zero_shot_yolo11s": evaluate(coco, CLASSES),
            "predict_nothing": evaluate(nothing, CLASSES),
        },
        "notes": {
            "coco_zero_shot_yolo11s": "Every COCO detection counted as 'product'; COCO has no gap class, so 'missing' is never predicted.",
            "precision_recall": "At the confidence that maximises each class's F1 at IoU 0.50.",
        },
    }
    cm = confusion_matrix(finetuned, len(CLASSES), conf=0.25, iou_thr=0.45)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts / "box_confusion.json").write_text(json.dumps({
        "labels": CLASSES + ["background"],
        "rows": "true class", "columns": "predicted class",
        "conf": 0.25, "iou": 0.45,
        "matrix": cm.tolist(),
    }, indent=2), encoding="utf-8")

    per_photo = _per_photo_records(finetuned)
    (artifacts / "test_predictions.json").write_text(json.dumps(per_photo), encoding="utf-8")
    save_valid_predictions(weights_path, artifacts, smoke)

    # Ultralytics' own numbers, printed only as a sanity check on our evaluator.
    try:
        from ultralytics import YOLO
        ref = YOLO(str(weights_path)).val(data=str(CLEAN_DIR / "data.yaml"), split="test",
                                          imgsz=PRED_ARGS["imgsz"], max_det=1000, verbose=False,
                                          plots=False, project=str(ROOT / "runs"), name="val_check",
                                          exist_ok=True)
        print(f"[sanity] Ultralytics model.val() on test: mAP50 {ref.box.map50:.3f}, mAP50-95 {ref.box.map:.3f} "
              "(not reported: val() uses multi-label NMS, so its boxes differ slightly from predict(); "
              "metrics.json applies Ultralytics' AP maths to predict() boxes for all three models)")
    except Exception as e:  # noqa: BLE001
        print(f"[sanity] Ultralytics val skipped: {e}")

    _copy_examples(per_photo, artifacts, smoke)
    return metrics


def _per_photo_records(preds: list[dict]) -> list[dict]:
    """Per-photo data the policy and the dashboard need: true gap count and every
    'missing' confidence. Any k can be scored later without rerunning the model."""
    miss_id = CLASSES.index("missing")
    out = []
    for im in preds:
        confs = np.sort(im["pred_conf"][im["pred_cls"] == miss_id])[::-1]
        out.append({
            "image": im["path"].name,
            "gt_missing": int((im["gt_cls"] == miss_id).sum()),
            "gt_product": int((im["gt_cls"] != miss_id).sum()),
            "missing_conf": [round(float(c), 4) for c in confs],
        })
    return out


def save_valid_predictions(weights_path: Path, artifacts: Path, smoke: bool) -> None:
    """Same per-photo records for the deduplicated validation photos, where the
    thresholds are chosen. The validation split was only used for early
    stopping/best-epoch selection during training, never for fitting weights."""
    items = _load_ground_truth(CLEAN_DIR / "valid", limit=6 if smoke else None)
    print(f"Scoring {len(items)} unique validation photos (used to choose thresholds) ...")
    preds = _predict(str(weights_path), items, lambda c: int(c))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "valid_predictions.json").write_text(json.dumps(_per_photo_records(preds)), encoding="utf-8")


def _copy_examples(per_photo: list[dict], artifacts: Path, smoke: bool) -> None:
    """A few test photos for the dashboard (low / typical / high gap counts)."""
    dst = (artifacts.parent / "examples") if smoke else ROOT / "examples"
    dst.mkdir(parents=True, exist_ok=True)
    ranked = sorted(per_photo, key=lambda d: d["gt_missing"])
    picks = {ranked[0]["image"], ranked[len(ranked) // 2]["image"],
             ranked[(3 * len(ranked)) // 4]["image"], ranked[-1]["image"]}
    for name in picks:
        shutil.copy2(CLEAN_DIR / "test" / "images" / name, dst / name)


# ----------------------------------------------------------------------------
# 5. Policy
# ----------------------------------------------------------------------------

def run_policy(artifacts: Path) -> dict:
    a = sp.load_assumptions()
    valid_path = artifacts / "valid_predictions.json"
    if not valid_path.exists():
        raise SystemExit("artifacts/valid_predictions.json is missing: run `python train.py --valid-only` first.")
    val = sp.load_test_predictions(valid_path)
    test = sp.load_test_predictions(artifacts / "test_predictions.json")
    s_val, y_val = sp.image_scores(val["missing_confs"], a.k_gaps), sp.true_labels(val["gt_missing"], a.k_gaps)
    s_test, y_test = sp.image_scores(test["missing_confs"], a.k_gaps), sp.true_labels(test["gt_missing"], a.k_gaps)

    # 1) Choose on validation.
    sweep_val = sp.sweep_single_threshold(s_val, y_val, a)
    t_single = sp.best_single_threshold(sweep_val)
    band = sp.best_band(s_val, y_val, a)
    # 2) Report on test, with the thresholds fixed.
    sweep_test = sp.sweep_single_threshold(s_test, y_test, a)
    alone = sp.evaluate_policy(s_test, y_test, a, t_single, t_single)
    chosen = sp.evaluate_policy(s_test, y_test, a, band["t_low"], band["t_high"])
    alternatives = sp.compare_alternatives(s_test, y_test, a, t_single, band)
    # For transparency: what tuning directly on test would have given (optimistic).
    t_test_opt = sp.best_single_threshold(sweep_test)
    band_test_opt = sp.best_band(s_test, y_test, a)

    with (artifacts / "cost_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "cost_per_1000_valid", "cost_per_1000_test", "TP_test", "FN_test", "FP_test",
                    "TN_test", "recall_test"])
        for v, t in zip(sweep_val, sweep_test):
            w.writerow([t["threshold"], round(v["cost_per_1000_photos"], 2), round(t["cost_per_1000_photos"], 2),
                        t["TP"], t["FN"], t["FP"], t["TN"], round(t["recall"], 4)])

    op = {
        "assumptions_file": "costs.json",
        "assumptions": json.loads((ROOT / "costs.json").read_text(encoding="utf-8")),
        "thresholds_chosen_on": f"validation ({len(s_val)} unique photos)",
        "costs_reported_on": f"test ({len(s_test)} unique photos)",
        "test_photos": int(len(s_test)),
        "test_positive_share": float(y_test.mean()) if len(y_test) else 0.0,
        "valid_positive_share": float(y_val.mean()) if len(y_val) else 0.0,
        "base_rate_used": sp.effective_base_rate(a, y_test),
        "single_threshold": t_single,
        "band": band,
        "model_alone": alone,
        "chosen": chosen,
        "alternatives": alternatives,
        "validation": {
            "model_alone_cost_per_1000": sp.evaluate_policy(s_val, y_val, a, t_single, t_single)["cost_per_1000_photos"],
            "band_cost_per_1000": sp.evaluate_policy(s_val, y_val, a, band["t_low"], band["t_high"])["cost_per_1000_photos"],
        },
        "in_sample_test_optimum": {
            "note": "Thresholds tuned directly on test. Optimistic, shown only to measure how much the choice depends on the sample.",
            "single_threshold": t_test_opt,
            "single_cost_per_1000": sp.evaluate_policy(s_test, y_test, a, t_test_opt, t_test_opt)["cost_per_1000_photos"],
            "band": band_test_opt,
            "band_cost_per_1000": sp.evaluate_policy(s_test, y_test, a, band_test_opt["t_low"], band_test_opt["t_high"])["cost_per_1000_photos"],
        },
    }
    (artifacts / "operating_point.json").write_text(json.dumps(op, indent=2), encoding="utf-8")
    _plot_cost_curve(sweep_val, sweep_test, t_single, band, artifacts / "cost_curve.png", a.k_gaps)

    print(f"\nPolicy (k = {a.k_gaps}, base rate {op['base_rate_used']:.1%}). Chosen on validation, costed on test:")
    print(f"  Model alone: threshold {t_single:.2f}, {alone['cost_per_1000_photos']:.0f} EUR per 1,000 photos")
    print(f"  Band {band['t_low']:.2f}-{band['t_high']:.2f}: {chosen['cost_per_1000_photos']:.0f} EUR per 1,000 photos, "
          f"{chosen['review_share']:.1%} to review ({chosen['reviews_per_day']:.0f}/day, "
          f"{'fits' if chosen['fits_review_capacity'] else 'EXCEEDS'} capacity)")
    for alt in alternatives:
        print(f"  {alt['policy']:<45} {alt['cost_per_1000_photos']:8.0f} EUR / 1,000 photos")
    ins = op["in_sample_test_optimum"]
    print(f"  (Tuned on test instead: threshold {ins['single_threshold']:.2f} / band "
          f"{ins['band']['t_low']:.2f}-{ins['band']['t_high']:.2f}, {ins['band_cost_per_1000']:.0f} EUR per 1,000 photos)")
    return op


def _plot_cost_curve(sweep_val, sweep_test, t_single, band, path: Path, k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["threshold"] for r in sweep_test]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, [r["cost_per_1000_photos"] for r in sweep_test], color="#2563eb", lw=2, label="test (reported)")
    ax.plot(t, [r["cost_per_1000_photos"] for r in sweep_val], color="#2563eb", lw=1.5, ls="--", alpha=0.6,
            label="validation (used to choose)")
    ax.axvline(t_single, color="#dc2626", ls="--", label=f"chosen single threshold = {t_single:.2f}")
    ax.axvspan(band["t_low"], band["t_high"], color="#f59e0b", alpha=0.2,
               label=f"review band {band['t_low']:.2f}-{band['t_high']:.2f}")
    ax.set_xlabel("Confidence threshold (k-th most confident gap)")
    ax.set_ylabel("Expected cost per 1,000 photos (EUR)")
    ax.set_title(f"Cost against threshold, model deciding alone (k = {k})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------

def bundle_results(weights_dir: Path, artifacts: Path, smoke: bool) -> None:
    target = (artifacts.parent / "results") if smoke else ROOT / "results"
    staging = ROOT / "_results_staging"
    if staging.exists():
        shutil.rmtree(staging)
    # Only our model: Ultralytics can drop helper checkpoints (e.g. its AMP check model) into weights/.
    (staging / "weights").mkdir(parents=True)
    shutil.copy2(weights_dir / "shelf_gaps_yolo11s.pt", staging / "weights" / "shelf_gaps_yolo11s.pt")
    for d in (artifacts, (artifacts.parent / "examples") if smoke else ROOT / "examples"):
        if d.exists():
            shutil.copytree(d, staging / d.name)
    shutil.make_archive(str(target), "zip", staging)
    shutil.rmtree(staging)
    print(f"\nBundled weights, artifacts and examples into {Path(str(target) + '.zip').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true", help="skip training, evaluate saved weights")
    mode.add_argument("--policy-only", action="store_true", help="only recompute costs and thresholds")
    mode.add_argument("--valid-only", action="store_true",
                      help="score the validation photos with the saved weights, then recompute the policy")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run into smoke_output/")
    args = parser.parse_args()

    weights_dir, artifacts = out_dirs(args.smoke)
    weights_path = weights_dir / "shelf_gaps_yolo11s.pt"

    if args.policy_only:
        run_policy(artifacts)
        return

    download_dataset()
    stats = build_clean_dataset()
    if args.valid_only:
        if not weights_path.exists():
            raise SystemExit(f"--valid-only needs {weights_path.relative_to(ROOT)}; train first.")
        save_valid_predictions(weights_path, artifacts, args.smoke)
        run_policy(artifacts)
        return
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Clean dataset:", {s: v["unique_images_kept"] for s, v in stats["splits"].items()},
          f"(dropped {stats['dropped_boxes']} box(es) of unused classes)")

    info_path = artifacts / "training_info.json"
    if not args.eval_only:
        info = train(weights_dir, args.smoke)
        info["dataset"] = f"{WORKSPACE}/{PROJECT}/{VERSION}"
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        results_png = Path(info["run_dir"]) / "results.png"
        if results_png.exists():
            shutil.copy2(results_png, artifacts / "training_curves.png")
    elif not weights_path.exists():
        raise SystemExit(f"--eval-only needs {weights_path.relative_to(ROOT)}; train first.")

    evaluate_models(weights_path, artifacts, args.smoke)
    run_policy(artifacts)
    bundle_results(weights_dir, artifacts, args.smoke)


if __name__ == "__main__":
    main()
