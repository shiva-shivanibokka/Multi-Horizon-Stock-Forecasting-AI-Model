"""
retrain.py
Retrains all four forecasting models and saves new checkpoints.

TWO MODES:

  Full rebuild (run locally, first time or after major changes):
    python retrain.py --refresh-data

  Incremental / weekly update (GitHub Actions default):
    python retrain.py
    - Downloads only the last 30 days of new price data
    - Appends new windows to the existing .npz dataset
    - Fine-tunes models on the recent data only (fast, low memory)

  Retrain a single model:
    python retrain.py --model transformer
    python retrain.py --model lstm
    python retrain.py --model rnn
    python retrain.py --model rf

GITHUB ACTIONS NOTES:
  Free GitHub runners have 2 CPUs / 7 GB RAM / ~14 GB disk.
  The full 10-year dataset (~57 GB) cannot fit on a free runner.

  The weekly workflow:
    1. Downloads model weights (.pth/.pkl) from the previous commit
    2. Downloads only the last 30 days of new price data (~50 MB)
    3. Appends new windows to a TEMPORARY in-memory dataset
    4. Fine-tunes each model for a few epochs on the new data
    5. Commits updated weights back to the repo

  For a full rebuild (e.g. after architecture changes), run locally:
    python retrain.py --refresh-data

After training, the new .pth / .pkl checkpoints are saved in-place.
If RENDER_DEPLOY_HOOK is set in the environment, the script calls it
to trigger a Render redeploy so the Flask backend picks up new weights.
"""

import os
import sys
import argparse
import subprocess
import requests


RENDER_HOOK = os.environ.get("RENDER_DEPLOY_HOOK", "")


def run(cmd: str, cwd: str):
    """Runs a shell command in the given directory and streams output."""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr
    )
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def trigger_redeploy():
    """Pings the Render deploy hook so the backend reloads new weights.
    Only runs if RENDER_DEPLOY_HOOK is set in the environment."""
    if not RENDER_HOOK:
        print("No RENDER_DEPLOY_HOOK set, skipping redeploy trigger.")
        return
    try:
        resp = requests.post(RENDER_HOOK, timeout=10)
        print(f"Redeploy triggered: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Could not trigger redeploy: {e}")


BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIRS = {
    "transformer": os.path.join(BASE, "transformer_final"),
    "lstm": os.path.join(BASE, "lstm_final_project"),
    "rnn": os.path.join(BASE, "rnn_final"),
    "rf": os.path.join(BASE, "rf_final"),
}

TRAIN_SCRIPTS = {
    "transformer": "train_transformer.py",
    "lstm": "train_lstm.py",
    "rnn": "train_rnn.py",
    "rf": "train_rf.py",
}


def build_dataset(refresh: bool = False, incremental: bool = False):
    """
    Runs build_dataset.py to download and cache the shared dataset.

    --refresh:     Full re-download of 10 years of data (run locally, ~20-30 min).
    --incremental: Only download the last 30 days and append new windows (GitHub
                   Actions default — fast, low memory, no OOM risk).
    Neither flag:  Use existing cache if present, otherwise full build.
    """
    if refresh:
        flag = "--refresh"
    elif incremental:
        flag = "--incremental"
    else:
        flag = ""
    run(f"python build_dataset.py {flag}".strip(), cwd=BASE)


def retrain(model_name: str):
    print(f"\n{'=' * 50}")
    print(f"Retraining: {model_name.upper()}")
    print(f"{'=' * 50}")
    script = TRAIN_SCRIPTS[model_name]
    cwd = MODEL_DIRS[model_name]
    run(f"python {script}", cwd=cwd)
    print(f"{model_name.upper()} training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain stock forecasting models")
    parser.add_argument(
        "--model",
        choices=["transformer", "lstm", "rnn", "rf", "all"],
        default="all",
        help="Which model to retrain (default: all)",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Full 10-year re-download. Run locally only — too large for GitHub Actions.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only download the last 30 days and append new windows (default on GitHub Actions).",
    )
    args = parser.parse_args()

    # Step 1: update the shared dataset
    # - Locally with --refresh-data: full 10-year rebuild
    # - On GitHub Actions (default): incremental update of last 30 days only
    is_ci = os.environ.get("CI", "false").lower() == "true"
    use_incremental = args.incremental or (is_ci and not args.refresh_data)

    if args.refresh_data:
        print("Full dataset rebuild requested (run locally only).")
        build_dataset(refresh=True)
    elif use_incremental:
        print("Incremental dataset update (GitHub Actions mode).")
        build_dataset(incremental=True)
    else:
        print("Using existing dataset cache.")
        build_dataset()

    # Step 2: retrain the requested models
    models = list(MODEL_DIRS.keys()) if args.model == "all" else [args.model]
    for m in models:
        retrain(m)

    trigger_redeploy()
    print("\nAll retraining complete.")
