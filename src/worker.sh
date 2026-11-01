#!/bin/bash
# worker: pops "task scheme fold" lines from a shared queue (flock) and runs run_fold.py on its own GPU
R=$1; Q=$2; MODE=$3
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ]; then export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; fi
cd "$(dirname "$0")"
while true; do
  line=$(flock $Q.lock bash -c "head -n1 $Q; sed -i '1d' $Q")
  [ -z "$line" ] && break
  set -- $line
  if [ "$MODE" = train ]; then
    $PY run_fold.py --task $1 --scheme $2 --fold $3 --out $R >> $R/logs/train_$1_$2_$3.log 2>&1 || echo "FAIL $line" >> $R/logs/failures.txt
  else
    $PY evaluate.py --runs $R --task $1 --scheme $2 --out $R/eval --B 1000 --Bc 1000 --Bsub 200 --Bcenter 200 >> $R/logs/eval_$1_$2.log 2>&1 || echo "FAIL eval $line" >> $R/logs/failures.txt
  fi
  echo "$(date +%T) $(hostname) gpu=$CUDA_VISIBLE_DEVICES done $line" >> $R/logs/progress.txt
done
