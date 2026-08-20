#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR"

SCENARIO="agent_only_simple"
TOTAL_STEPS=10000000
GRU_PILOT_STEPS=6000000

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Each of the 30 environment workers should use one BLAS/OpenMP thread.
# This also overrides invalid thread values occasionally injected by AutoDL.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ALGOS=(
  sagi_ppo_mlp
  sagi_ppo_gru
  ppo_lagrangian_mlp
  ppo_lagrangian_gru
)
SEEDS=(0 42 123 1337 3407)

run_one() {
  local algo="$1"
  local seed="$2"
  local total_steps="$3"
  local n_envs="$4"
  local n_steps="$5"
  local n_epochs="$6"
  local batch_size="$7"
  local output_group="$8"
  local resume_from="${9:-}"

  local log_dir="console_logs/${output_group}/seed_${seed}"
  local log_file="${log_dir}/${SCENARIO}_${algo}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$log_dir"

  echo "----------------------------------------"
  echo "场景: $SCENARIO"
  echo "算法: $algo"
  echo "种子: $seed"
  echo "步数: $total_steps"
  echo "日志: $log_file"
  echo "----------------------------------------"

  local train_args=(
    --algo "$algo"
    --scenario "$SCENARIO"
    --total-timesteps "$total_steps"
    --n-envs "$n_envs"
    --n-steps "$n_steps"
    --batch-size "$batch_size"
    --n-epochs "$n_epochs"
    --target-kl 0.03
    --lr 3e-5
    --gamma 0.99
    --gae-lambda 0.95
    --clip-range 0.2
    --hidden-dim 256
    --rnn-hidden-dim 64
    --seed "$seed"
    --initial-cost-limit 650
    --final-cost-limit 8
    --cost-warmup-fraction 0.10
    --cost-anneal-fraction 0.40
    --lambda-lr 0.035
    --cost-vf-coef 0.5
    --save-freq 2000000
    --model-save-root "models/${output_group}/seed_${seed}"
    --tensorboard-log-dir "../tf-logs/${output_group}/seed_${seed}"
  )
  if [[ -n "$resume_from" ]]; then
    echo "续训模型: $resume_from"
    train_args+=(--resume-from "$resume_from")
  fi

  python -u train.py "${train_args[@]}" 2>&1 | tee "$log_file"
}

latest_pilot_model() {
  local pilot_dir="models/p0_2_constraint_fix/seed_42/${SCENARIO}"
  local candidates=()
  local candidate
  local latest=""

  shopt -s nullglob
  candidates=(
    "$pilot_dir"/${SCENARIO}_sagi_ppo_mlp_*/sagi_ppo_mlp_final_model.zip
  )
  shopt -u nullglob

  for candidate in "${candidates[@]}"; do
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done

  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

latest_gru_pilot_model() {
  local algo="$1"
  local pilot_dir="models/p0_3_gru_mask_pilot/seed_42/${SCENARIO}"
  local candidates=()
  local candidate
  local latest=""

  shopt -s nullglob
  candidates=(
    "$pilot_dir"/${SCENARIO}_${algo}_*/${algo}_final_model.zip
  )
  shopt -u nullglob

  for candidate in "${candidates[@]}"; do
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done

  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

model_target_steps() {
  local model_path="$1"
  local config_path="${model_path%/*}/training_config.json"
  python -c \
    'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["total_timesteps"]))' \
    "$config_path"
}

MODE="${1:-}"

case "$MODE" in
  smoke)
    # Verifies the complete collection/update/save path. It is not a result run.
    run_one sagi_ppo_mlp 42 1024 1 1024 2 512 constraint_fix_smoke
    ;;
  gru-smoke)
    # Exercises both constrained GRU code paths after feature-extractor changes.
    for algo in sagi_ppo_gru ppo_lagrangian_gru; do
      run_one "$algo" 42 1024 1 1024 2 512 p0_3_gru_mask_smoke
    done
    ;;
  gru-pilot)
    # Seed 42 exposed failures in both GRU variants in the 10M experiment.
    # Six million steps still preserve the same 10%/40%/50% schedule and give
    # three million steps at the fixed final budget before a full retrain.
    for algo in sagi_ppo_gru ppo_lagrangian_gru; do
      run_one \
        "$algo" 42 "$GRU_PILOT_STEPS" 30 2048 10 512 \
        p0_3_gru_mask_pilot
    done
    ;;
  extend-gru-pilot)
    # Diagnostic continuation only: the original 6M curriculum reached the
    # final budget earlier than a fresh 10M run, so these models are not final
    # experiment results even after reaching 10M total timesteps.
    for algo in sagi_ppo_gru ppo_lagrangian_gru; do
      if ! gru_model="$(latest_gru_pilot_model "$algo")"; then
        echo "未找到 ${algo} 的 GRU pilot 最终模型。"
        echo "请确认已完成：bash run_p01.sh gru-pilot"
        exit 1
      fi

      gru_steps="$(model_target_steps "$gru_model")"
      if (( gru_steps >= TOTAL_STEPS )); then
        echo "${algo} 的最新 pilot 已达到 ${gru_steps} 步，跳过。"
        continue
      fi

      echo "将 ${algo} 从约 ${gru_steps} 步诊断性续训到 ${TOTAL_STEPS} 步。"
      run_one \
        "$algo" 42 "$TOTAL_STEPS" 30 2048 10 512 \
        p0_3_gru_mask_pilot "$gru_model"
    done
    ;;
  pilot)
    # Fresh 10M-step pilot. Existing 8M pilots should use extend-pilot instead.
    run_one sagi_ppo_mlp 42 "$TOTAL_STEPS" 30 2048 10 512 p0_2_constraint_fix
    ;;
  extend-pilot)
    if ! pilot_model="$(latest_pilot_model)"; then
      echo "未找到 pilot 最终模型。请先运行：bash run_p01.sh pilot"
      exit 1
    fi
    pilot_steps="$(model_target_steps "$pilot_model")"
    if (( pilot_steps >= TOTAL_STEPS )); then
      echo "最新 pilot 已达到 ${pilot_steps} 步，无需再次续训。"
      exit 0
    fi
    echo "将 pilot 从约 ${pilot_steps} 步续训到 ${TOTAL_STEPS} 步。"
    run_one \
      sagi_ppo_mlp 42 "$TOTAL_STEPS" 30 2048 10 512 \
      p0_2_constraint_fix "$pilot_model"
    ;;
  train)
    if ! pilot_model="$(latest_pilot_model)"; then
      echo "未找到合格的完整 pilot。请先运行：bash run_p01.sh pilot"
      exit 1
    fi
    pilot_steps="$(model_target_steps "$pilot_model")"
    if (( pilot_steps != TOTAL_STEPS )); then
      echo "最新 pilot 的目标步数为 ${pilot_steps}，正式实验要求 ${TOTAL_STEPS}。"
      echo "请先运行：bash run_p01.sh extend-pilot"
      exit 1
    fi

    for seed in "${SEEDS[@]}"; do
      for algo in "${ALGOS[@]}"; do
        if [[ "$seed" == "42" && "$algo" == "sagi_ppo_mlp" ]]; then
          echo "复用已经完成的 pilot：seed=42, algo=sagi_ppo_mlp"
          continue
        fi
        run_one "$algo" "$seed" "$TOTAL_STEPS" 30 2048 10 512 p0_2_constraint_fix
      done
    done
    ;;
  *)
    echo "使用方法："
    echo "  bash run_p01.sh smoke           # MLP 1024 步代码短测试"
    echo "  bash run_p01.sh gru-smoke       # 两个 GRU 各运行 1024 步"
    echo "  bash run_p01.sh gru-pilot       # 两个 GRU seed=42 各运行 600 万步"
    echo "  bash run_p01.sh extend-gru-pilot # 两个 GRU 从 600 万诊断续训到 1000 万步"
    echo "  bash run_p01.sh pilot          # 从头跑 1000 万步 pilot"
    echo "  bash run_p01.sh extend-pilot   # 将已有 800 万步 pilot 续训到 1000 万步"
    echo "  bash run_p01.sh train          # pilot 合格后训练剩余 19 个模型"
    exit 1
    ;;
esac
