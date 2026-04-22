#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1


MAX_JOBS=6

cmds=(
"python train.py --dataset svhn --model c2f2 --cos_lr --epochs 100  --train_mode NDFA --name c2f2_NDFA_svhn"
"python train.py --dataset cifar10 --model c2f2 --cos_lr --epochs 100  --train_mode NDFA --name c2f2_NDFA_cifar10"
"python train.py --dataset cifar100 --model c2f2 --cos_lr --epochs 100  --train_mode NDFA --name c2f2_NDFA_cifar100"

#"python train.py --dataset svhn --model resnet --cos_lr --dfa_num 16 --epochs 200 --train_mode NDFA --name Res32_NDFA16_svhn"
#"python train.py --dataset cifar10 --model resnet --cos_lr --dfa_num 16 --epochs 200 --train_mode NDFA --name Res32_NDFA16_cifar10"
#"python train.py --dataset cifar100 --model resnet --cos_lr --dfa_num 16 --epochs 200 --train_mode NDFA --name Res32_NDFA16_cifar100"

"python train.py --dataset svhn --model resnet --cos_lr --dfa_num 4 --epochs 200 --train_mode NDFA --name Res32_NDFA4_svhn"
"python train.py --dataset cifar10 --model resnet --cos_lr --dfa_num 4 --epochs 200 --train_mode NDFA --name Res32_NDFA4_cifar10"
"python train.py --dataset cifar100 --model resnet --cos_lr --dfa_num 4 --epochs 200 --train_mode NDFA --name Res32_NDFA4_cifar100"
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

