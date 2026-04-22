"""
monitor.py
──────────
Live terminal dashboard for parallel training runs.

Shows for each run:
  • Current tqdm batch progress (live, from stdout.log)
  • Last 5 completed epochs with all metrics (from training_log.csv)
  • Estimated time remaining
  • Heartbeat / alive status

Usage:
    python monitor.py                     # uses default paths
    python monitor.py --refresh 20        # refresh every 20 seconds
    python monitor.py --output_base /media/ailab/LandslideJH/LeakDB/LeakDB/outputs

Quit with Ctrl-C.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE = "/media/ailab/LandslideJH/LeakDB/LeakDB/outputs"

RUNS = [
    {
        "name":       "Run A  │  pos_weight=10  │  GPU 1",
        "subdir":     "run_A_pw10",
        "color_code": "\033[36m",   # cyan
    },
    {
        "name":       "Run B  │  pos_weight=5   │  GPU 0",
        "subdir":     "run_B_pw5",
        "color_code": "\033[33m",   # yellow
    },
]

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
DIM    = "\033[2m"
CLEAR  = "\033[2J\033[H"   # clear screen + move cursor to top


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv_tail(path: str, n: int = 5) -> list[dict]:
    """Return the last n rows of training_log.csv as dicts."""
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows[-n:]


def _read_heartbeat(path: str) -> dict:
    """Parse heartbeat.txt into a dict."""
    result = {}
    if not os.path.exists(path):
        return result
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def _read_stdout_tail(path: str, n_lines: int = 6) -> list[str]:
    """
    Return the last n_lines of stdout.log.
    Used to show the live tqdm progress bar and epoch prints.
    """
    if not os.path.exists(path):
        return []
    try:
        # Efficient tail: seek from end
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            # Read last 4 KB — more than enough for a few lines
            fh.seek(max(0, size - 4096))
            raw = fh.read().decode("utf-8", errors="replace")
        lines = raw.splitlines()
        # Strip blank lines, keep last n
        lines = [l for l in lines if l.strip()]
        return lines[-n_lines:]
    except Exception:
        return []


def _eta(rows: list[dict]) -> str:
    """Estimate remaining training time from the last two epoch rows."""
    if len(rows) < 2:
        return "—"
    try:
        last   = rows[-1]
        second = rows[-2]
        epoch_secs = float(last["elapsed_s"]) - float(second["elapsed_s"])
        if epoch_secs <= 0:
            return "—"
        # patience is not stored in CSV; assume a ceiling of 300 epochs
        # and estimate from the last epoch's elapsed time
        current_epoch = int(last["epoch"])
        # Rough upper bound: assume 300 epochs max
        max_epoch = 300
        remaining_epochs = max_epoch - current_epoch
        eta_secs = remaining_epochs * epoch_secs
        eta = timedelta(seconds=int(eta_secs))
        return str(eta)
    except Exception:
        return "—"


def _alive_status(hb: dict, color: str, log_path: str) -> str:
    """
    Return a coloured alive/dead string.
    Priority: heartbeat timestamp (post-epoch) → stdout.log mtime (mid-epoch)
    This means the monitor shows 'alive' even before the first epoch completes.
    """
    # Mid-epoch detection: stdout.log was modified in the last 3 minutes
    if os.path.exists(log_path):
        try:
            log_age = time.time() - os.path.getmtime(log_path)
            if log_age < 180:   # modified within last 3 minutes
                ts_str = hb.get("timestamp", "")
                if not ts_str:
                    return (f"{GREEN}● training epoch 1{RESET}  "
                            f"(stdout.log active {int(log_age)}s ago, "
                            f"no epoch completed yet)")
        except Exception:
            pass

    # Post-epoch detection: use heartbeat timestamp
    ts_str = hb.get("timestamp", "")
    if not ts_str:
        # stdout.log exists but is old, or doesn't exist
        if os.path.exists(log_path):
            try:
                log_age = time.time() - os.path.getmtime(log_path)
                if log_age > 300:
                    return f"{RED}● stalled or not started{RESET}  (stdout.log last touched {int(log_age/60)}m ago)"
            except Exception:
                pass
        return f"{RED}● not started{RESET}"

    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        age_secs = (datetime.now() - ts).total_seconds()
        if age_secs < 120:
            return f"{GREEN}● alive{RESET}  (heartbeat {int(age_secs)}s ago)"
        elif age_secs < 1800:   # within 30 min — probably mid-epoch
            return f"{color}● epoch in progress{RESET}  ({int(age_secs/60)}m since last completed epoch)"
        else:
            return f"{RED}● stalled or stopped{RESET}  (last seen {ts_str})"
    except Exception:
        return f"{DIM}● unknown{RESET}"


def _fmt_row(row: dict) -> str:
    """Format one CSV row as a single readable line."""
    try:
        ep   = int(row["epoch"])
        tl   = float(row["train_loss"])
        vl   = float(row["val_loss"])
        f1   = float(row["val_f1"])
        auc  = float(row["val_roc_auc"])
        prec = float(row["val_precision"])
        rec  = float(row["val_recall"])
        ela  = float(row["elapsed_s"])
        h    = int(ela // 3600)
        m    = int((ela % 3600) // 60)
        return (
            f"  Ep {ep:>4}  "
            f"TrLoss {tl:.4f}  VaLoss {vl:.4f}  │  "
            f"F1 {f1:.4f}  AUC {auc:.4f}  "
            f"P {prec:.4f}  R {rec:.4f}  │  "
            f"{h}h{m:02d}m elapsed"
        )
    except Exception:
        return f"  {row}"


# ── Render ────────────────────────────────────────────────────────────────────

def render(base_dir: str, refresh: int) -> None:
    """Single render pass — clears screen and redraws everything."""
    now = datetime.now().strftime("%H:%M:%S")
    width = 100

    lines = []
    lines.append(CLEAR)
    lines.append(f"{BOLD}{'─' * width}{RESET}")
    lines.append(
        f"{BOLD}  LeakDB Training Monitor"
        f"{'':>30}refreshes every {refresh}s   {now}{RESET}"
    )
    lines.append(f"{BOLD}{'─' * width}{RESET}")

    for run in RUNS:
        c    = run["color_code"]
        path = os.path.join(base_dir, run["subdir"])

        csv_path  = os.path.join(path, "logs", "training_log.csv")
        hb_path   = os.path.join(path, "logs", "heartbeat.txt")
        log_path  = os.path.join(path, "stdout.log")

        rows = _read_csv_tail(csv_path, n=5)
        hb   = _read_heartbeat(hb_path)
        tail = _read_stdout_tail(log_path, n_lines=5)

        best_f1 = hb.get("best_val_f1", "—")
        epoch   = hb.get("epoch",       "—")
        gpus    = hb.get("gpus",        "1")

        lines.append("")
        lines.append(f"{c}{BOLD}  {run['name']}{RESET}")
        lines.append(f"  Status   : {_alive_status(hb, c, log_path)}")
        lines.append(f"  Progress : epoch {epoch}   best val F1 = {best_f1}   GPUs = {gpus}")
        lines.append(f"  ETA (≤300 ep ceiling) : {_eta(rows)}")

        # Last 5 epochs
        lines.append(f"{DIM}  {'─' * (width - 2)}{RESET}")
        if rows:
            lines.append(f"{DIM}  Last {len(rows)} completed epoch(s):{RESET}")
            for row in rows:
                lines.append(f"{DIM}{_fmt_row(row)}{RESET}")
        else:
            lines.append(f"{DIM}  No epochs completed yet — first epoch takes ~20 min.{RESET}")

        # Live tqdm / stdout tail
        lines.append(f"{DIM}  {'─' * (width - 2)}{RESET}")
        lines.append(f"{DIM}  Live output (stdout.log tail):{RESET}")
        if tail:
            for tl in tail:
                # Trim very long tqdm lines to fit terminal width
                display = tl[:width - 4] if len(tl) > width - 4 else tl
                lines.append(f"{DIM}  {display}{RESET}")
        else:
            if os.path.exists(log_path):
                lines.append(f"{DIM}  (stdout.log exists but is empty){RESET}")
            else:
                lines.append(f"{DIM}  (stdout.log not found — run not started){RESET}")

    lines.append("")
    lines.append(f"{BOLD}{'─' * width}{RESET}")
    lines.append(f"{DIM}  Ctrl-C to exit   │   output dirs: {base_dir}{RESET}")
    lines.append(f"{BOLD}{'─' * width}{RESET}")

    print("\n".join(lines), end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Live monitor for parallel training runs")
    p.add_argument(
        "--output_base", type=str, default=DEFAULT_BASE,
        help="Parent directory containing run_A_pw10/ and run_B_pw5/"
    )
    p.add_argument(
        "--refresh", type=int, default=30,
        help="Seconds between screen refreshes (default 30)"
    )
    args = p.parse_args()

    print(f"Starting monitor... refreshing every {args.refresh}s. Ctrl-C to quit.")
    time.sleep(1)

    try:
        while True:
            render(args.output_base, args.refresh)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        # Move cursor below the dashboard before exiting
        print("\n\nMonitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
