#!/bin/bash
# worker2: queue lines are either "retrain TASK SCHEME FOLD MODELS" or "eval TASK SCHEME MODELS"
R=$1; Q=$2
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ]; then export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; fi
cd "$(dirname "$0")"
while true; do
  line=$(flock $Q.lock bash -c "head -n1 $Q; sed -i '1d' $Q")
  [ -z "$line" ] && break
  set -- $line
  if [ "$1" = retrain ]; then
    SKIP_DONE=0 $PY run_fold.py --task $2 --scheme $3 --fold $4 --models $5 --merge --out $R >> $R/logs/retrain_$2_$3_$4.log 2>&1 || echo "FAIL $line" >> $R/logs/failures2.txt
  else
    $PY evaluate.py --runs $R --task $2 --scheme $3 --models $4 --out $R/eval2 --B 1000 --Bc 1000 --Bsub 200 --Bcenter 200 >> $R/logs/eval2_$2_$3_$4.log 2>&1 || echo "FAIL $line" >> $R/logs/failures2.txt
  fi
  echo "$(date +%T) $(hostname) gpu=$CUDA_VISIBLE_DEVICES done $line" >> $R/logs/progress2.txt
done
