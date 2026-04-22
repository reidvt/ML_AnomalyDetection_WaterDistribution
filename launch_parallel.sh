#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_parallel.sh
#
# Starts two independent training jobs simultaneously, one per GPU.
# Each job writes to its own output directory so nothing is overwritten.
#
# Run from the project directory:
#   chmod +x launch_parallel.sh
#   ./launch_parallel.sh
#
# Monitor:
#   tail -f /media/ailab/LandslideJH/LeakDB/LeakDB/outputs/run_A_pw10/logs/training_log.csv
#   tail -f /media/ailab/LandslideJH/LeakDB/LeakDB/outputs/run_B_pw5/logs/training_log.csv
#
#   cat /media/ailab/LandslideJH/LeakDB/LeakDB/outputs/run_A_pw10/logs/heartbeat.txt
#   cat /media/ailab/LandslideJH/LeakDB/LeakDB/outputs/run_B_pw5/logs/heartbeat.txt
#
#   watch -n 30 nvidia-smi
#
# Stop both jobs cleanly (saves checkpoint before exit):
#   kill -TERM $(cat run_A.pid) $(cat run_B.pid)
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: 'set -e' intentionally removed.
# The INP-file-not-found warning writes to stderr during startup.
# With set -e, that stderr output (non-zero exit from some subshells)
# caused the script to abort after Run A, leaving Run B never started.
# This produced only 499 batches for Run A — it was seeing a partial
# dataset because its own startup was interrupted mid-index.
# We handle critical failures explicitly below instead.

BASE_DIR="/media/ailab/LandslideJH/LeakDB/LeakDB"
DATA_ROOT="${BASE_DIR}/data_master"
OUTPUT_A="${BASE_DIR}/outputs/run_A_pw10"   # pos_weight=10  high recall baseline
OUTPUT_B="${BASE_DIR}/outputs/run_B_pw5"    # pos_weight=5   balanced precision/recall

CONDA_ENV="leak_pipeline"
CONDA_BASE="/home/ailab/Downloads/enter"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

# ── Batch sizes tuned to saturate each GPU ───────────────────────────────────
#
# The model is only 280 KB of weights.  At batch=256 the GPU was drawing
# 80-129W vs a 260W cap — clear data-loading starvation.  Larger batches
# give the GPU more work per data-load cycle and push utilisation higher.
#
# GPU 1 (clean, 24.5 GB free)  →  batch 1024 safe
# GPU 0 (competing cvpr job using ~5.5 GB, ~14 GB free for training)  →  batch 512 safe
#
# If either job OOMs on startup, halve its batch size and relaunch.
BATCH_A=1024
BATCH_B=512

EPOCHS=150
PATIENCE=15    # 15 gives ~14-22h worst case; 25 risks not finishing by tomorrow
WORKERS=12
CACHE=128      # Parquet DataFrames in RAM per process

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LeakDB parallel training launcher"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Run A  GPU 1 (clean)   batch=${BATCH_A}  pos_weight=10  ${OUTPUT_A}"
echo "  Run B  GPU 0 (shared)  batch=${BATCH_B}  pos_weight=5   ${OUTPUT_B}"
echo ""

mkdir -p "${OUTPUT_A}" "${OUTPUT_B}"

# ── Run A: GPU 1, pos_weight=10 (high recall) ────────────────────────────────
echo "▶  Starting Run A on GPU 1 (batch ${BATCH_A})..."
CUDA_VISIBLE_DEVICES=1 \
  nohup "${PYTHON}" train.py \
    --output_dir          "${OUTPUT_A}" \
    --data_root           "${DATA_ROOT}" \
    --pos_weight          10 \
    --batch_size          "${BATCH_A}" \
    --epochs              "${EPOCHS}" \
    --patience            "${PATIENCE}" \
    --num_workers         "${WORKERS}" \
    --dataset_cache_size  "${CACHE}" \
    > "${OUTPUT_A}/stdout.log" 2>&1 &

PID_A=$!
echo "${PID_A}" > run_A.pid
echo "   PID: ${PID_A}   log: ${OUTPUT_A}/stdout.log"

PID_A=$!
echo "${PID_A}" > run_A.pid
echo "   PID: ${PID_A}   log: ${OUTPUT_A}/stdout.log"

# Wait until Run A has finished indexing the dataset before starting Run B.
# This prevents both processes from hammering the same 1000 Parquet files
# simultaneously during the metadata scan, which can cause partial indexing.
echo "   Waiting for Run A dataset index (up to 90s)..."
for i in $(seq 1 18); do
    sleep 5
    if grep -q "LeakWindowDataset:" "${OUTPUT_A}/stdout.log" 2>/dev/null; then
        echo "   ✓ Run A dataset ready:"
        grep -E "LeakWindowDataset:|Split →|DataLoader:" "${OUTPUT_A}/stdout.log" \
            2>/dev/null | sed 's/^/     /'
        break
    fi
    echo -n "   ."
done
echo ""

# ── Run B: GPU 0, pos_weight=5 (balanced) ────────────────────────────────────
echo "▶  Starting Run B on GPU 0 (batch ${BATCH_B})..."
CUDA_VISIBLE_DEVICES=0 \
  nohup "${PYTHON}" train.py \
    --output_dir          "${OUTPUT_B}" \
    --data_root           "${DATA_ROOT}" \
    --pos_weight          5 \
    --batch_size          "${BATCH_B}" \
    --epochs              "${EPOCHS}" \
    --patience            "${PATIENCE}" \
    --num_workers         "${WORKERS}" \
    --dataset_cache_size  "${CACHE}" \
    > "${OUTPUT_B}/stdout.log" 2>&1 &

PID_B=$!
echo "${PID_B}" > run_B.pid
echo "   PID: ${PID_B}   log: ${OUTPUT_B}/stdout.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Both jobs running.  PIDs: A=${PID_A}  B=${PID_B}"
echo ""
echo "  Live metrics:"
echo "    tail -f ${OUTPUT_A}/logs/training_log.csv"
echo "    tail -f ${OUTPUT_B}/logs/training_log.csv"
echo ""
echo "  GPU utilisation (updates every 2s):"
echo "    watch -n 2 nvidia-smi"
echo ""
echo "  Alive checks:"
echo "    cat ${OUTPUT_A}/logs/heartbeat.txt"
echo "    cat ${OUTPUT_B}/logs/heartbeat.txt"
echo ""
echo "  Stop both cleanly:"
echo "    kill -TERM \$(cat run_A.pid) \$(cat run_B.pid)"
echo ""
echo "  Generate plots after each finishes:"
echo "    python evaluate.py --output_dir ${OUTPUT_A}"
echo "    python evaluate.py --output_dir ${OUTPUT_B}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
