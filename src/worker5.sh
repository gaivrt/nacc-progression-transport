#!/bin/bash
# worker5: each queue line is an argument string for $PY (scripts in the job's code dir); lines popped under flock
Q=$1; LOG=$2
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ]; then export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; fi
cd "$(dirname "$0")"
while true; do
  line=$(flock $Q.lock bash -c "head -n1 $Q; sed -i '1d' $Q")
  [ -z "$line" ] && break
  h=$(echo "$line" | md5sum | cut -c1-10)
  $PY $line >> $LOG/job_$h.log 2>&1 || echo "FAIL $line" >> $LOG/failures.txt
  echo "$(date +%T) $(hostname) gpu=$CUDA_VISIBLE_DEVICES done $line" >> $LOG/progress.txt
done
