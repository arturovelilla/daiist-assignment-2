"""
Box-level detection metrics, computed the same way for every model compared.

One evaluator for the finetuned model, the COCO zero-shot model and the
predict-nothing baseline, so the comparison table cannot mix two different
definitions of "AP".

- Matching (same rule as Ultralytics): per class and IoU threshold, all
  prediction/ground-truth pairs above the threshold are sorted by IoU and
  matched one-to-one, highest IoU first. Matched predictions are true positives.
- AP (same as Ultralytics): area under the precision envelope, sampled at 101
  recall points and integrated with the trapezoid rule, at IoU 0.50 (AP50) and
  averaged over IoU 0.50:0.05:0.95 (AP50-95). Using Ultralytics' definition keeps
  these numbers comparable with Roboflow's reference model on the same data.
- Precision / recall per class: at the confidence that maximises that class's
  F1 at IoU 0.50 (the convention Ultralytics prints).
- Confusion matrix: rows = true class, columns = predicted class, last row and
  column = background. Built at a fixed confidence and IoU (defaults match
  Ultralytics: conf 0.25, IoU 0.45), class-agnostic matching so a gap predicted
  as a product shows up as a confusion rather than as two separate errors.
"""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), boxes as x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _match_one_image(pred_box, pred_conf, gt_box) -> np.ndarray:
    """True-positive flags (n_pred, n_iou_thresholds) for one image and one class."""
    tp = np.zeros((len(pred_box), len(IOU_THRESHOLDS)), dtype=bool)
    if len(gt_box) == 0 or len(pred_box) == 0:
        return tp
    ious = iou_matrix(gt_box, pred_box)                       # (n_gt, n_pred)
    for t, thr in enumerate(IOU_THRESHOLDS):
        gi, pj = np.nonzero(ious >= thr)
        if len(gi) == 0:
            continue
        pairs = np.stack([gi, pj], axis=1)
        if len(pairs) > 1:
            pairs = pairs[ious[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]   # one GT per prediction
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]   # one prediction per GT
        tp[pairs[:, 1], t] = True
    return tp


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    integrate = getattr(np, "trapezoid", None) or np.trapz      # name changed in numpy 2.0
    return float(integrate(np.interp(x, mrec, mpre), x))


def evaluate(images: list[dict], class_names: list[str]) -> dict:
    """
    images: one dict per image with numpy arrays
        gt_cls (n,), gt_box (n,4), pred_cls (m,), pred_conf (m,), pred_box (m,4)
    Returns per-class and overall metrics.
    """
    per_class = {}
    for c, name in enumerate(class_names):
        confs, tps, n_gt = [], [], 0
        for im in images:
            g = im["gt_box"][im["gt_cls"] == c]
            pm = im["pred_cls"] == c
            n_gt += len(g)
            if pm.any():
                confs.append(im["pred_conf"][pm])
                tps.append(_match_one_image(im["pred_box"][pm], im["pred_conf"][pm], g))
        if not confs or n_gt == 0:
            per_class[name] = {"ap50": 0.0, "ap50_95": 0.0, "precision": 0.0, "recall": 0.0,
                               "best_f1_conf": None, "n_gt": n_gt,
                               "n_pred": int(sum(len(x) for x in confs))}
            continue
        conf = np.concatenate(confs)
        tp = np.concatenate(tps)
        order = np.argsort(-conf, kind="stable")
        conf, tp = conf[order], tp[order]
        ctp = np.cumsum(tp, axis=0)
        cfp = np.cumsum(~tp, axis=0)
        recall = ctp / n_gt
        precision = ctp / (ctp + cfp)
        aps = [_ap(recall[:, t], precision[:, t]) for t in range(len(IOU_THRESHOLDS))]
        f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + 1e-9)
        b = int(f1.argmax())
        per_class[name] = {
            "ap50": aps[0],
            "ap50_95": float(np.mean(aps)),
            "precision": float(precision[b, 0]),
            "recall": float(recall[b, 0]),
            "best_f1_conf": float(conf[b]),
            "n_gt": n_gt,
            "n_pred": int(len(conf)),
        }
    vals = list(per_class.values())
    return {
        "per_class": per_class,
        "overall": {
            "map50": float(np.mean([v["ap50"] for v in vals])),
            "map50_95": float(np.mean([v["ap50_95"] for v in vals])),
            "precision": float(np.mean([v["precision"] for v in vals])),
            "recall": float(np.mean([v["recall"] for v in vals])),
        },
    }


def confusion_matrix(images: list[dict], n_classes: int, conf: float = 0.25,
                     iou_thr: float = 0.45) -> np.ndarray:
    """Rows = true class (+ background), columns = predicted class (+ background)."""
    bg = n_classes
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for im in images:
        keep = im["pred_conf"] >= conf
        pc, pb = im["pred_cls"][keep], im["pred_box"][keep]
        gc, gb = im["gt_cls"], im["gt_box"]
        ious = iou_matrix(gb, pb)
        gi, pj = np.nonzero(ious > iou_thr)
        order = np.argsort(-ious[gi, pj], kind="stable")
        used_g, used_p = set(), set()
        for k in order:
            g, p = int(gi[k]), int(pj[k])
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            m[gc[g], pc[p]] += 1
        for g in range(len(gc)):
            if g not in used_g:
                m[gc[g], bg] += 1
        for p in range(len(pc)):
            if p not in used_p:
                m[bg, pc[p]] += 1
    return m
