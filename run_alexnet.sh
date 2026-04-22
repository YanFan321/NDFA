#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1


MAX_JOBS=2

cmds=(

"python train_alexnet.py --cos_lr --train_mode NDFA --lr 5e-4 --name alex_NDFA_cifar10 --dataset cifar10"
"python train_alexnet.py --cos_lr --train_mode NDFA --lr 5e-4 --name alex_NDFA_cifar100 --dataset cifar100"

)

LOG_TO_FILE=0

run_one () {
  local cmd="$1"
  local tag
  tag="$(echo "$cmd" | sed 's/[^A-Za-z0-9._-]/_/g')"
  echo "[START] $cmd"
  if (( LOG_TO_FILE == 1 )); then
    mkdir -p logs
    bash -lc "$cmd" >"logs/${tag}.log" 2>&1 &
  else
    bash -lc "$cmd" &
  fi
}


running=0
for c in "${cmds[@]}"; do
  run_one "$c"
  running=$((running+1))
  if (( running >= MAX_JOBS )); then
    wait -n
    running=$((running-1))
  fi
done
wait
echo "All done."

