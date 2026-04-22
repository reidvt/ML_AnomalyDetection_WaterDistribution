"""
preprocess.py
─────────────
Stage 0 of the pipeline.  Scans the raw LeakDB directory tree and converts
each scenario into a single wide-format Parquet file.

Run once before training:
    # Use the default paths defined below
    python preprocess.py

    # Or override via CLI (recommended for the AI Lab)
    python preprocess.py \
        --raw_dir /media/ailab/LandslideJH/LeakDB/LeakDB \
        --out_dir /media/ailab/LandslideJH/LeakDB/data_master

BUG FIXED: the original script parsed --raw_dir / --out_dir from CLI but then
used the hardcoded globals ROOT_DATA_DIR / CLEAN_PARQUET_DIR in the actual
scan loop, making the CLI flags completely ineffective.
"""

import os
import argparse
import warnings

import pandas as pd
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# ── Default paths (edit for your machine) ─────────────────────────────────────
_DEFAULT_RAW_DIR = r"C:\Users\reidv\SDX\SEM2\MergedTraining\5scenarios-1000"
_DEFAULT_OUT_DIR = r"C:\Users\reidv\SDX\SEM2\MergedTraining\data_master"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_read_csv(file_path: str, col_name: str) -> "pd.DataFrame | None":
    """
    Read a single-column CSV robustly:
      - Coerces index and values to numeric (drops header rows / junk rows).
      - Sorts and de-duplicates on index.
    Returns a single-column DataFrame named `col_name`, or None on failure.
    """
    try:
        df            = pd.read_csv(file_path, index_col=0)
        df.index      = pd.to_numeric(df.index,       errors="coerce")
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0],  errors="coerce")
        df            = df.dropna()
        df            = df.sort_index()
        df            = df[~df.index.duplicated(keep="last")]
        df            = df.iloc[:, [0]]
        df.columns    = [col_name]
        return df
    except Exception:
        return None


# ── Main scan ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw LeakDB CSV scenarios to Parquet master files."
    )
    parser.add_argument(
        "--raw_dir", type=str, default=_DEFAULT_RAW_DIR,
        help="Root of the raw LeakDB directory tree (contains Scenario-N folders)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=_DEFAULT_OUT_DIR,
        help="Output directory for Parquet master files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process scenarios even if the output file already exists",
    )
    args = parser.parse_args()

    # ── Resolve to absolute paths ─────────────────────────────────────────
    # FIX: use args.raw_dir / args.out_dir everywhere — never the old globals.
    raw_dir = os.path.abspath(os.path.expanduser(args.raw_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔍 Scanning  : {raw_dir}")
    print(f"📥 Output    : {out_dir}")
    print(f"   Force re-process: {args.force}\n")

    # ── Identify scenario directories ─────────────────────────────────────
    scenario_dirs: set = set()
    for root, dirs, _ in os.walk(raw_dir):
        if "pressures" in [d.lower() for d in dirs]:
            scenario_dirs.add(root)

    print(f"📦 Found {len(scenario_dirs)} scenarios.\n")
    if not scenario_dirs:
        print("❌ No scenarios found.  Check that --raw_dir points to the "
              "correct location and that each scenario has a 'Pressures' sub-folder.")
        return

    skipped = processed = errors = 0

    for scenario_path in tqdm(sorted(scenario_dirs), desc="Processing"):
        scenario_name = os.path.basename(scenario_path).replace(" ", "_")
        network_name  = os.path.basename(os.path.dirname(scenario_path)).replace(" ", "_")
        out_name      = f"{network_name}_{scenario_name}.parquet"
        out_path      = os.path.join(out_dir, out_name)   # FIX: use out_dir

        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue

        try:
            data_frames: list = []
            used_columns: set = set()

            for root, _, files in os.walk(scenario_path):
                folder_type = os.path.basename(root).lower()

                for fname in sorted(files):
                    if not fname.lower().endswith(".csv"):
                        continue

                    element_name = fname[:-4]
                    file_path    = os.path.join(root, fname)

                    if   folder_type == "leaks":     base_col = "Label"
                    elif folder_type == "pressures": base_col = f"{element_name}_pressure"
                    elif folder_type == "demands":   base_col = f"{element_name}_demand"
                    elif folder_type == "flows":     base_col = f"{element_name}_flow"
                    else:
                        continue

                    # De-duplicate column names within this scenario
                    final_col = base_col
                    counter   = 1
                    while final_col in used_columns:
                        final_col = f"{base_col}_{counter}"
                        counter  += 1
                    used_columns.add(final_col)

                    df = _safe_read_csv(file_path, final_col)
                    if df is not None:
                        data_frames.append(df)

            if not data_frames:
                continue

            df_final = pd.concat(data_frames, axis=1)

            label_cols = [c for c in df_final.columns if c.startswith("Label")]
            df_final["Label_Combined"] = (
                df_final[label_cols].max(axis=1) if label_cols else 0.0
            )

            df_final = df_final.ffill().fillna(0.0)
            df_final.to_parquet(out_path, engine="pyarrow", compression="snappy")
            processed += 1

        except Exception as exc:
            print(f"\n❌ Error on {out_name}: {exc}")
            errors += 1

    print(
        f"\n🎉 Preprocessing complete. "
        f"Processed: {processed} | Skipped (exists): {skipped} | Errors: {errors}"
    )


if __name__ == "__main__":
    main()
