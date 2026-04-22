#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MAX_JOBS=6

cmds=(

"python train_vit.py --model vits --batch_size 256 --train_mode NDFA --name vit_NDFA12_cifar10 --dataset cifar10"
"python train_vit.py --model vits --batch_size 256 --train_mode NDFA --name vit_NDFA12_cifar100 --dataset cifar100"

"python train_vit.py --model vits --batch_size 256 --dfa_num 4 --train_mode NDFA --name vit_NDFA4_cifar10 --dataset cifar10"
"python train_vit.py --model vits --batch_size 256 --dfa_num 4 --train_mode NDFA --name vit_NDFA4_cifar100 --dataset cifar100"

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
