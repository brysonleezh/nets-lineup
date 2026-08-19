"""
Phase 6 Step 4 - prediction freeze. Writes data/projections/predictions_frozen.json
and guards data/projections/nets_rookies_2026.csv against silent overwrite.

Run: python3 src/translator/phase6_step4_freeze.py [--refreeze "reason"]
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTIONS_DIR = REPO_ROOT / "data" / "projections"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
CSV_PATH = PROJECTIONS_DIR / "nets_rookies_2026.csv"
FROZEN_PATH = PROJECTIONS_DIR / "predictions_frozen.json"
CONFIG_PATH = REPO_ROOT / "config.yaml"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(refreeze_reason: str | None = None) -> None:
    if FROZEN_PATH.exists() and refreeze_reason is None:
        raise SystemExit(
            f"{FROZEN_PATH} already exists - predictions are frozen. Pass "
            f"--refreeze \"<reason>\" if this is an intentional, disclosed re-freeze."
        )

    csv_hash = sha256_of(CSV_PATH)
    config_hash = sha256_of(CONFIG_PATH)
    deploy_manifest_hash = sha256_of(TRANSLATOR_DIR / "deployment" / "manifest.json")
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip() or "uncommitted"

    holdout_metrics = pd.read_csv(TRANSLATOR_DIR / "holdout_metrics.csv").iloc[0]
    phase5_metrics = {
        "jsd_mean": float(holdout_metrics["jsd_mean"]),
        "l1_mean": float(holdout_metrics["l1_mean"]),
        "top1_hit_mean": float(holdout_metrics["top1_hit_mean"]),
        "top1_within_top2_mean": float(holdout_metrics["top1_within_top2_mean"]),
        "n_holdout": 36, "holdout_class": 2025,
        "beats_t4": bool(holdout_metrics["beats_t4"]),
        "beats_t5i": bool(holdout_metrics["beats_t5i"]),
        "beats_t5ii": bool(holdout_metrics["beats_t5ii"]),
    }

    preds = pd.read_csv(CSV_PATH)
    rookie_summaries = []
    for _, r in preds.iterrows():
        rookie_summaries.append({
            "display_name": r["display_name"],
            "predicted_top3": ast.literal_eval(r["y_top3"]),  # list of (archetype_idx, weight) tuples
            "predicted_argmax": int(r["y_pred_argmax"]),
            "saturated": bool(r["saturated"]),
        })

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "nets_rookies_2026_csv_sha256": csv_hash,
        "config_yaml_sha256": config_hash,
        "deployment_manifest_sha256": deploy_manifest_hash,
        "phase5_holdout_metrics_at_freeze_time": phase5_metrics,
        "rookies": rookie_summaries,
        "note": "Frozen before the 2026-27 NBA season began. This file's own sha256 references "
                "prove exactly which model version and which config produced these predictions. "
                "See src/eval/review_2026_predictions.py for the pre-committed evaluation plan.",
        "refreeze_log": [],
    }

    if refreeze_reason is not None and FROZEN_PATH.exists():
        old = json.loads(FROZEN_PATH.read_text())
        record["refreeze_log"] = old.get("refreeze_log", [])
        record["refreeze_log"].append({
            "refrozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": refreeze_reason,
            "previous_csv_sha256": old.get("nets_rookies_2026_csv_sha256"),
        })
        print(f"REFREEZE: {refreeze_reason}")

    FROZEN_PATH.write_text(json.dumps(record, indent=2, default=str))
    print(f"Wrote {FROZEN_PATH}")
    print(f"  csv sha256: {csv_hash}")
    print(f"  git hash: {git_hash}")
    for r in rookie_summaries:
        print(f"  {r['display_name']}: top3={r['predicted_top3']}")


if __name__ == "__main__":
    reason = None
    if "--refreeze" in sys.argv:
        idx = sys.argv.index("--refreeze")
        reason = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "no reason given"
    freeze(refreeze_reason=reason)
