#!/bin/bash
# Run any registered model on SPOT-Bench tasks, one shard per GPU, then merge + score.
# usage: MODEL=MageVL [RUN=results/MageVL NGPU=8 PY=python ARGS=--no_repeat] bash scripts/run_sharded.sh [TASK ...]
# (default tasks: ABD PNR). Re-running resumes unfinished shards from their checkpoints.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=${PY:-python}
MODEL=${MODEL:-MageVL}
RUN=${RUN:-results/$MODEL}
NGPU=${NGPU:-8}
TASKS=${@:-ABD PNR}

for TASK in $TASKS; do
  key=$(echo $TASK | tr A-Z a-z)
  [ -f $RUN/shard0/$key.json ] || $PY scripts/shard.py shard $TASK $NGPU $RUN
  pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    [ -f $RUN/shard$i/${key}_$MODEL.json ] && continue
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
      $PY inference.py --task $TASK --model $MODEL --resume $ARGS \
        --annotation_path $RUN/shard$i/$key.json --result_dir $RUN/shard$i \
        > $RUN/shard$i/$key.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || echo "[WARN] shard pid $p failed"; done
  $PY scripts/shard.py merge $TASK $NGPU $RUN $MODEL
done
$PY score.py --tasks $TASKS --model $MODEL --result_dir $RUN
