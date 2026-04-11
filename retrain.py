"""
retrain.py
Retrains all four forecasting models and saves new checkpoints.

This script is designed to be called by the GitHub Actions cron workflow
every week. It can also be run manually at any time:

    python retrain.py                     # retrain all 4 models
    python retrain.py --model transformer # retrain only transformer
    python retrain.py --model lstm
    python retrain.py --model rnn
    python retrain.py --model rf

After training, the new .pth / .pkl checkpoints are saved in-place.
If RENDER_DEPLOY_HOOK is set in the environment, the script calls it
to trigger a Render redeploy so the Flask backend picks up new weights.

MLflow logs each run to ./mlruns/ so you can compare across weeks:
    mlflow ui
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


def build_dataset(refresh: bool = False):
    """
    Runs build_dataset.py to download and cache the shared dataset.
    All four training scripts load from this cache — data is downloaded
    once rather than four times.
    """
    flag = "--refresh" if refresh else ""
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
        help="Force re-download of the dataset even if a cache already exists",
    )
    args = parser.parse_args()

    # Step 1: build / refresh the shared dataset cache
    # This downloads S&P 500 data once for all four models to share
    build_dataset(refresh=args.refresh_data)

    # Step 2: retrain the requested models
    models = list(MODEL_DIRS.keys()) if args.model == "all" else [args.model]

    for m in models:
        retrain(m)

    trigger_redeploy()

    print("\nAll retraining complete.")
