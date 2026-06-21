"""
Sequential experiment runner. Reads experiments.yaml and runs each experiment in order.
Per-run logs go to runs/<run_name>/train.log.
Sends ntfy.sh notification after each run.

Usage (on server):
  cd ~/solafune/code
  python run_all.py                       # run all experiments
  python run_all.py --start_from v13_ir12ch_focal  # skip earlier experiments
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml


NTFY_TOPIC = "solafune_luiz_train"
TRAIN_SCRIPT = Path(__file__).parent / "src" / "train.py"


def send_ntfy(title: str, message: str):
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode(),
            headers={"Title": title, "Priority": "default"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[ntfy] Failed to send notification: {e}")


def recompute_stats(data_dir: str, csv_train: str):
    """Delete and recompute stats.json using train.py's compute_stats."""
    stats_path = Path(data_dir) / "stats.json"
    if stats_path.exists():
        print(f"[stats] Removing old {stats_path}")
        stats_path.unlink()
    print("[stats] Recomputing stats.json ...")
    src_dir = Path(__file__).parent / "src"
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{src_dir}'); "
         f"from train import compute_stats; from pathlib import Path; "
         f"compute_stats(Path('{csv_train}'), Path('{data_dir}'), Path('{stats_path}')); "
         f"print('[stats] Done.')"],
        check=True,
    )
    return result.returncode == 0


def run_experiment(run_name: str, params: dict) -> bool:
    """Run a single experiment. Returns True if successful (exit code 0)."""
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    cmd = [sys.executable, str(TRAIN_SCRIPT), "--run_name", run_name]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    print(f"\n{'='*60}")
    print(f"Starting: {run_name}")
    print(f"Command:  {' '.join(cmd)}")
    print(f"Log:      {log_path}")
    print(f"{'='*60}\n")

    send_ntfy(f"[START] {run_name}", f"Starting experiment {run_name}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,   # tqdm writes to stderr; let it go directly to terminal
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        last_rmse = "n/a"
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
            # Track best val RMSE from saved-best lines
            if "Saved best model" in line:
                try:
                    last_rmse = line.split("RMSE=")[1].split(")")[0].strip()
                except Exception:
                    pass
        proc.wait()

    success = proc.returncode == 0
    status = "DONE" if success else f"FAILED (exit {proc.returncode})"
    msg = f"{status} | best_val_rmse={last_rmse} | log={log_path}"
    print(f"\n[{run_name}] {msg}\n")
    send_ntfy(f"[{status}] {run_name}", msg)
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments.yaml")
    parser.add_argument("--start_from", default=None,
                        help="Skip all experiments before this run_name.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them.")
    parser.add_argument("--auto_stop", action="store_true",
                        help="Stop Vast.ai instance after all experiments finish (requires vastai CLI).")
    parser.add_argument("--instance_id", default=None,
                        help="Vast.ai instance ID to stop. If omitted, reads $VAST_CONTAINERLABEL env var.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    experiments = cfg.get("experiments", [])

    skip = args.start_from is not None
    results = []

    for exp in experiments:
        run_name = exp["run_name"]

        if skip:
            if run_name == args.start_from:
                skip = False
            else:
                print(f"Skipping {run_name}")
                continue

        # Merge defaults + experiment-specific (experiment overrides defaults)
        params = {**defaults}
        do_recompute = False
        for k, v in exp.items():
            if k == "run_name":
                continue
            if k == "recompute_stats":
                do_recompute = bool(v)
            else:
                params[k] = v

        if args.dry_run:
            cmd = [f"--{k} {v}" for k, v in params.items()]
            recompute_note = " [recompute_stats]" if do_recompute else ""
            print(f"DRY RUN{recompute_note}: python src/train.py --run_name {run_name} {' '.join(cmd)}")
            continue

        if do_recompute:
            recompute_stats(
                data_dir=str(params.get("data_dir", defaults.get("data_dir", "~/solafune/data"))),
                csv_train=str(params.get("csv_train", defaults.get("csv_train", "~/solafune/data/train_dataset.csv"))),
            )

        success = run_experiment(run_name, params)
        results.append((run_name, "OK" if success else "FAILED"))

        if not success:
            print(f"\nExperiment {run_name} failed. Stopping queue.")
            break

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"  {name}: {status}")

    if args.auto_stop:
        import os
        instance_id = args.instance_id or os.environ.get("VAST_CONTAINERLABEL", "")
        if not instance_id:
            print("\n[auto_stop] Could not find instance ID. "
                  "Pass --instance_id or set $VAST_CONTAINERLABEL.")
        else:
            print(f"\n[auto_stop] Stopping instance {instance_id}...")
            send_ntfy("Stopping instance", f"All experiments done. Stopping {instance_id}.")
            os.system(f"vastai stop instance {instance_id}")


if __name__ == "__main__":
    main()
