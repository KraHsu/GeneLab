#!/usr/bin/env bash
# Auto eval + export once the go2w 6k training run finishes.
# Waits on the training PID, picks the highest-numbered checkpoint, runs a deterministic
# eval, and exports TorchScript + ONNX (history-aware obs schema) next to the checkpoint.
set -uo pipefail
cd /home/charles/workspace/dev/GeneLab

TRAIN_PID="${1:?usage: go2w_posttrain.sh <train_pid>}"
TASK="Genelab-Velocity-Flat-Unitree-Go2W-v0"
RUN_ROOT="logs/rsl_rl/go2w_velocity_flat"
export PYTHONPATH="examples/unitree/src${PYTHONPATH:+:$PYTHONPATH}"
GENELAB=(.venv/bin/genelab --import genelab_unitree.tasks)

echo "[posttrain] waiting for training PID $TRAIN_PID to exit..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[posttrain] training process exited at $(date -u +%H:%M:%SZ)"

# Newest run dir, then the highest model_<N>.pt in it.
RUN_DIR="$(ls -dt "$RUN_ROOT"/*/ 2>/dev/null | head -1)"
RUN_DIR="${RUN_DIR%/}"
if [[ -z "$RUN_DIR" ]]; then echo "[posttrain] FATAL: no run dir under $RUN_ROOT"; exit 1; fi
CKPT="$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "$CKPT" ]]; then echo "[posttrain] FATAL: no model_*.pt in $RUN_DIR"; exit 1; fi
echo "[posttrain] using checkpoint: $CKPT"

echo "[posttrain] === eval ==="
"${GENELAB[@]}" eval "$TASK" "$CKPT" --num-envs 64 --episodes 50 \
  --out "$RUN_DIR/eval.json" 2>&1 || echo "[posttrain] eval FAILED (continuing to export)"

echo "[posttrain] === export torchscript ==="
"${GENELAB[@]}" export "$TASK" "$CKPT" -f torchscript --out "$RUN_DIR/policy.ts" 2>&1 \
  || echo "[posttrain] torchscript export FAILED"

echo "[posttrain] === export onnx ==="
"${GENELAB[@]}" export "$TASK" "$CKPT" -f onnx --out "$RUN_DIR/policy.onnx" 2>&1 \
  || echo "[posttrain] onnx export FAILED"

echo "[posttrain] done. artifacts in $RUN_DIR:"
ls -la "$RUN_DIR"/eval.json "$RUN_DIR"/policy.ts* "$RUN_DIR"/policy.onnx* 2>&1
