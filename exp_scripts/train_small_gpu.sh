#!/usr/bin/env bash
set -euo pipefail

# Conservative single-node LUFFY recipe for 2 x NVIDIA A40 (48 GiB).
# It preserves the standard 1 off-policy + 7 on-policy group and the
# 8192-token response limit. CPU offload trades speed for GPU memory, so the
# node should have at least ~128 GiB of available system RAM.

export HF_ENDPOINT="${LUFFY_HF_ENDPOINT:-https://huggingface.co}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

LUFFY_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LUFFY_ROOT="${LUFFY_ROOT:-$(cd -- "$LUFFY_SCRIPT_DIR/.." && pwd)}"

# If the caller already activated the luffy environment, leave it alone.
# Otherwise, LUFFY_CONDA_ENV may be either an environment name or full path.
if [[ "${CONDA_DEFAULT_ENV:-}" != "luffy" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${LUFFY_CONDA_ENV:-luffy}"
fi

LUFFY_MODEL_PATH="${MODEL_PATH:-Elliott/Qwen2.5-Math-7B-16k-think}"
LUFFY_DATA_DIR="${DATA_DIR:-$LUFFY_ROOT/data}"
LUFFY_EXP_NAME="${EXP_NAME:-LUFFY_2XA40}"
LUFFY_WANDB_PROJECT="${WANDB_PROJECT:-luffy-math-2xa40}"

LUFFY_VISIBLE_GPUS="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
if (( LUFFY_VISIBLE_GPUS < 2 )); then
    echo "LUFFY requires 2 visible GPUs for this recipe; PyTorch sees $LUFFY_VISIBLE_GPUS." >&2
    echo "Check the scheduler allocation and CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

if [[ ! -f "$LUFFY_DATA_DIR/openr1.parquet" ]]; then
    echo "Missing training data: $LUFFY_DATA_DIR/openr1.parquet" >&2
    exit 1
fi
if [[ ! -f "$LUFFY_DATA_DIR/valid.parquet" ]]; then
    echo "Missing validation data: $LUFFY_DATA_DIR/valid.parquet" >&2
    exit 1
fi

ray stop >/dev/null 2>&1 || true
cd "$LUFFY_ROOT/luffy/verl"

set -x
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=$LUFFY_DATA_DIR/openr1.parquet" \
    "data.val_files=$LUFFY_DATA_DIR/valid.parquet" \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    "actor_rollout_ref.model.path=$LUFFY_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    data.reward_impl_version=3 \
    data.shuffle=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    "trainer.project_name=$LUFFY_WANDB_PROJECT" \
    "trainer.experiment_name=$LUFFY_EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.max_optim_to_keep=2 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=500 \
    trainer.total_epochs=30 \
    "$@"

