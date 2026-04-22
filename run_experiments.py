"""
run_experiments.py
──────────────────
Reads experiments.yaml and launches each experiment as an independent
background process on its assigned GPU.

Usage:
    python run_experiments.py                    # launch all experiments
    python run_experiments.py --dry_run          # print commands without running
    python run_experiments.py --list             # show experiments and GPU plan
    python run_experiments.py --kill             # stop all running experiments

Requires PyYAML:
    pip install pyyaml

All output goes to <output_base>/<experiment_name>/stdout.log
Metrics go to   <output_base>/<experiment_name>/logs/training_log.csv
"""

import argparse
import os
import signal
import subprocess
import sys
import time

try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Run:  pip install pyyaml")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Stability check
# ──────────────────────────────────────────────────────────────────────────────

STABILITY_THRESHOLD = 5e-3   # lr × pos_weight × (batch/256) must be < this

def _stability_score(exp: dict) -> float:
    lr         = exp.get("lr",         1e-3)
    pos_weight = exp.get("pos_weight", 10.0)
    batch_size = exp.get("batch_size", 256)
    return lr * pos_weight * (batch_size / 256)


def _check_stability(exp: dict) -> tuple[bool, str]:
    score = _stability_score(exp)
    safe  = score < STABILITY_THRESHOLD
    msg   = (
        f"lr={exp.get('lr',1e-3):.0e} × "
        f"pos_weight={exp.get('pos_weight',10)} × "
        f"(batch={exp.get('batch_size',256)}/256) = "
        f"{score:.2e}  {'✓ stable' if safe else '✗ LIKELY EXPLOSION'}"
    )
    return safe, msg


# ──────────────────────────────────────────────────────────────────────────────
# CLI builder
# ──────────────────────────────────────────────────────────────────────────────

# Map from YAML key → train.py CLI flag (only where name differs)
_YAML_TO_CLI = {
    "pos_weight":          "--pos_weight",
    "lr":                  "--lr",
    "batch_size":          "--batch_size",
    "grad_clip":           "--grad_clip",
    "epochs":              "--epochs",
    "patience":            "--patience",
    "num_workers":         "--num_workers",
    "dataset_cache_size":  "--dataset_cache_size",
    "window_size":         "--window_size",
    "stride":              "--stride",
    "checkpoint_every":    "--checkpoint_every",
    "data_root":           "--data_root",
    "inp_file":            "--inp_file",
    "lstm_hidden":         "--lstm_hidden",
    "embedding_dim":       "--embedding_dim",
    "gnn_hidden":          "--gnn_hidden",
    "gnn_layers":          "--gnn_layers",
    "gnn_dropout":         "--gnn_dropout",
    "lstm_dropout":        "--lstm_dropout",
}

# Keys handled specially — not passed as plain --flag value
_SPECIAL_KEYS = {"name", "gpu", "resume", "output_base", "use_demand",
                 "use_flows", "use_pressure", "val_split", "test_split",
                 "seed", "inp_file"}


def _build_command(python: str, exp: dict, shared: dict,
                   output_dir: str) -> list[str]:
    """Build the train.py subprocess command for one experiment."""
    merged = {**shared, **exp}   # experiment overrides shared

    cmd = [python, "train.py", "--output_dir", output_dir]

    for yaml_key, cli_flag in _YAML_TO_CLI.items():
        val = merged.get(yaml_key)
        if val is None:
            continue
        cmd += [cli_flag, str(val)]

    # Boolean flags
    if merged.get("resume"):
        cmd.append("--resume")
    if not merged.get("use_flows", True):
        cmd.append("--no_flows")
    if merged.get("use_demand", False):
        cmd.append("--use_demand")

    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Launcher
# ──────────────────────────────────────────────────────────────────────────────

def launch_experiments(cfg_path: str, dry_run: bool = False) -> None:
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    shared     = cfg.get("shared", {})
    experiments = cfg.get("experiments", [])
    output_base = shared.get("output_base",
                             "/media/ailab/LandslideJH/LeakDB/LeakDB/outputs")

    python = sys.executable

    pid_file = os.path.join(os.path.dirname(cfg_path), "experiment_pids.txt")
    pid_lines = []

    print(f"\n{'━'*65}")
    print(f"  LeakDB Experiment Launcher  —  {len(experiments)} experiment(s)")
    print(f"{'━'*65}\n")

    for exp in experiments:
        name       = exp["name"]
        gpu        = exp.get("gpu", 0)
        output_dir = os.path.join(output_base, name)
        os.makedirs(output_dir, exist_ok=True)

        # Stability check
        safe, stability_msg = _check_stability({**shared, **exp})
        status_icon = "✓" if safe else "⚠"
        print(f"  [{status_icon}] {name}  →  GPU {gpu}")
        print(f"      {stability_msg}")

        if not safe:
            print(
                f"\n  ❌  {name} FAILED stability check. "
                "Adjust lr, batch_size, or pos_weight in experiments.yaml.\n"
                "  Hint: lr × pos_weight × (batch/256) must be < 5e-3\n"
            )
            continue

        cmd = _build_command(python, exp, shared, output_dir)
        stdout_log = os.path.join(output_dir, "stdout.log")

        print(f"      output → {output_dir}")
        print(f"      log    → {stdout_log}")

        if dry_run:
            env_str = f"CUDA_VISIBLE_DEVICES={gpu}"
            print(f"      cmd    → {env_str} {' '.join(cmd)}\n")
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        with open(stdout_log, "w") as log_fh:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT
            )

        pid_lines.append(f"{proc.pid}  {name}  gpu={gpu}")
        print(f"      PID    → {proc.pid}\n")

        # Wait for this experiment's dataset index before starting the next.
        # This prevents both processes hammering the same 1000 Parquet files
        # simultaneously during the schema scan.
        if len(experiments) > 1:
            print(f"      Waiting for {name} to finish dataset indexing…")
            for _ in range(24):   # up to 2 min
                time.sleep(5)
                try:
                    with open(stdout_log) as lf:
                        content = lf.read()
                    if "LeakWindowDataset:" in content:
                        # Print the key lines
                        for line in content.splitlines():
                            if any(k in line for k in
                                   ["LeakWindowDataset:", "Split →",
                                    "DataLoader:", "Model :"]):
                                print(f"      {line.strip()}")
                        break
                except Exception:
                    pass
                print("      .", end="", flush=True)
            print()

    if not dry_run and pid_lines:
        with open(pid_file, "w") as fh:
            fh.write("\n".join(pid_lines) + "\n")

        print(f"\n{'━'*65}")
        print(f"  All experiments launched.  PIDs saved to: {pid_file}")
        print(f"\n  Monitor:   python monitor.py")
        print(f"  Stop all:  python run_experiments.py --kill")
        print(f"{'━'*65}\n")


def kill_experiments(cfg_path: str) -> None:
    pid_file = os.path.join(os.path.dirname(cfg_path), "experiment_pids.txt")
    if not os.path.exists(pid_file):
        print("No experiment_pids.txt found — nothing to kill.")
        return

    with open(pid_file) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    for line in lines:
        parts = line.split()
        pid   = int(parts[0])
        name  = parts[1] if len(parts) > 1 else "?"
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to {name} (PID {pid})")
        except ProcessLookupError:
            print(f"  {name} (PID {pid}) already stopped.")

    print("\nAll experiments sent SIGTERM (graceful save + exit).")


def list_experiments(cfg_path: str) -> None:
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    shared      = cfg.get("shared", {})
    experiments = cfg.get("experiments", [])

    print(f"\n{'━'*65}")
    print(f"  Experiments defined in {cfg_path}")
    print(f"{'━'*65}")

    for exp in experiments:
        merged = {**shared, **exp}
        safe, stability_msg = _check_stability(merged)
        icon = "✓" if safe else "⚠ UNSTABLE"
        print(f"\n  {icon}  {exp['name']}  (GPU {exp.get('gpu', 0)})")
        print(f"     pos_weight={merged.get('pos_weight')}  "
              f"lr={merged.get('lr')}  "
              f"batch={merged.get('batch_size')}  "
              f"grad_clip={merged.get('grad_clip')}")
        print(f"     {stability_msg}")
        print(f"     resume={exp.get('resume', False)}  "
              f"epochs={merged.get('epochs')}  "
              f"patience={merged.get('patience')}")

    print(f"\n{'━'*65}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch hyperparameter experiments from experiments.yaml"
    )
    p.add_argument(
        "--config", type=str, default="experiments.yaml",
        help="Path to experiments.yaml (default: ./experiments.yaml)"
    )
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    p.add_argument("--list",    action="store_true",
                   help="List all experiments and their stability scores")
    p.add_argument("--kill",    action="store_true",
                   help="Send SIGTERM to all running experiments")
    args = p.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    if args.list:
        list_experiments(args.config)
    elif args.kill:
        kill_experiments(args.config)
    else:
        launch_experiments(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
