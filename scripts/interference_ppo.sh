#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/mix-grpo-qwen2.5-3b.out
#SBATCH -e log/error-mix-grpo-qwen2.5-3b.out
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=grpo-qwen2.5-3b
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4
set -x
# 
nvidia-smi
source ~/.shadow1
conda activate llm_churn

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# Parse command line arguments
cl
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
    # MODEL_PATH="Qwen/Qwen2.5-7B"
# MODEL_PATH="models/Qwen2.5-3B-Instruct"
# model_name=Qwen2.5-3B-Instruct

MODEL_PATH="models/Qwen2.5-7B-Instruct-1M"
model_name=Qwen2.5-1.5B

python3 -m clipping_analysis.main \
    algorithm.adv_estimator=gae \
    data.train_files=data//train.parquet \
    data.val_files=data/reasoning_gym/test.parquet \
    data.buffer_files=data/reasoning_gym/Qwen2.5-7B-Instruct-1M_buffer_flatten.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=models/${model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64\
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    reward_model.launch_reward_fn_async=true\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.remove_truncated=False\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=models/${model_name} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=64 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.reward_manager=deepscaler \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rlvr-interfere' \
    trainer.experiment_name=${model_name}-Deepscaler-PPO\
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50000 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=50 "${@:1}"