#!/usr/bin/env python
"""Print a comparison table across rollout eval runs in a RUN_DIR.

Discovers full-frame rollout dirs (rollout/, rollout_sdedit_t*/, rollout_constrained_t*_a*/)
and segment rollout dirs (segment_rollout/, segment_rollout_sdedit_t*/).

Segment summaries use composite_* keys; we remap them to the canonical names below.
Usage:
    python scripts/compare_rollout_runs.py --run_dir outputs/paradigm_a_visible_v1
"""
import argparse
import glob
import json
import os
import sys


# Canonical metric names -> (display label, direction)
METRICS = [
    ("composite_image_mse",              "Image MSE",              "↓"),
    ("composite_crop_mse",               "Crop MSE (diffusion)",   "↓"),
    ("deterministic_crop_mse",           "Crop MSE (mean sprite)", "↓"),
    ("dynamics_gt_mse",                  "Dynamics GT MSE",        "↓"),
    ("constraint_detection_rate",        "Lander detection",       "↑"),
    ("constraint_centroid_err_vs_pred_px", "Centroid err/pred (px)", "↓"),
    ("constraint_centroid_err_vs_true_px", "Centroid err/true (px)", "↓"),
]
CONFIG_KEYS = ["sdedit_t_start", "constraint_alpha", "constraint_steps", "use_detected_position", "use_current_theta", "num_triplets", "pipeline"]

# Full-frame rollout uses different key names — remap to canonical
_KEY_ALIASES = {
    "generated_image_mse":    "composite_image_mse",
    "generated_crop_mse":     "composite_crop_mse",
    "deterministic_dp_image_mse": "deterministic_crop_mse",
    "deterministic_dp_crop_mse":  "deterministic_crop_mse",
}


def load_summary(path, pipeline_label):
    with open(path) as f:
        d = json.load(f)
    # Remap aliased keys to canonical
    for old, new in _KEY_ALIASES.items():
        if old in d and new not in d:
            d[new] = d.pop(old)
    d["pipeline"] = pipeline_label
    return d


def run_label(summary):
    pipeline = summary.get("pipeline", "")
    t = summary.get("sdedit_t_start", -1)
    a = summary.get("constraint_alpha", 0.0)
    det = summary.get("use_detected_position", False)
    cur_th = summary.get("use_current_theta", False)
    prefix = f"[{pipeline}] " if pipeline else ""
    if t is None or t <= 0:
        t_str = ""
    else:
        t_str = f" SDEdit t={t}"
    if a and a > 0:
        t_str += f" α={a}"
    if det:
        t_str += " +detPos"
    if cur_th:
        t_str += " +curTheta"
    label = (t_str.strip() or "pure noise")
    return f"{prefix}{label}"


def fmt(val):
    if val is None:
        return "  —  "
    if isinstance(val, float):
        if abs(val) < 0.001:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Output run directory to scan")
    args = parser.parse_args()

    candidates = []  # (path, pipeline_label)

    # Full-frame rollout dirs
    std = os.path.join(args.run_dir, "rollout", "summary.json")
    if os.path.exists(std):
        candidates.append((std, "full-frame"))
    for p in sorted(glob.glob(os.path.join(args.run_dir, "rollout_sdedit_t*", "summary.json"))):
        candidates.append((p, "full-frame"))
    for p in sorted(glob.glob(os.path.join(args.run_dir, "rollout_constrained_t*", "summary.json"))):
        candidates.append((p, "full-frame"))

    # Segment rollout dirs
    seg = os.path.join(args.run_dir, "segment_rollout", "summary.json")
    if os.path.exists(seg):
        candidates.append((seg, "segment"))
    for p in sorted(glob.glob(os.path.join(args.run_dir, "segment_rollout_*", "summary.json"))):
        candidates.append((p, "segment"))

    if not candidates:
        print(f"No rollout summary.json files found under {args.run_dir}")
        sys.exit(1)

    summaries = [load_summary(p, lbl) for p, lbl in candidates]
    labels = [run_label(s) for s in summaries]

    col_w = max(24, max(len(l) for l in labels) + 2)
    name_w = 28

    header = f"{'Metric':<{name_w}}" + "".join(f"  {l:>{col_w}}" for l in labels)
    sep = "-" * len(header)
    print(f"\n{'ROLLOUT COMPARISON':^{len(header)}}")
    print(f"{'run: ' + args.run_dir:^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for key, label, arrow in METRICS:
        vals = [s.get(key) for s in summaries]
        numeric = [v for v in vals if v is not None]
        if not numeric:
            continue
        best = min(numeric) if arrow == "↓" else max(numeric)
        row = f"{label + ' ' + arrow:<{name_w}}"
        for v in vals:
            f = fmt(v)
            marker = "*" if (v is not None and v == best and len(numeric) > 1) else " "
            row += f"  {marker}{f:>{col_w - 1}}"
        print(row)

    print(sep)
    print(f"{'Config':<{name_w}}" + "".join(f"  {'':>{col_w}}" for _ in summaries))
    for key in CONFIG_KEYS:
        row = f"  {key:<{name_w - 2}}"
        for s in summaries:
            val = s.get(key, "—")
            row += f"  {fmt(val):>{col_w}}"
        print(row)
    print(sep)
    print("* = best in row\n")

    for label, (path, _) in zip(labels, candidates):
        print(f"  [{label}]  {path}")
    print()


if __name__ == "__main__":
    main()
