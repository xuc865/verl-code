set -x
# Optional first arg: engine name (vllm|sglang|...). Must be shifted off $@ so Hydra
# never sees a bare 'vllm' override (Error parsing override 'vllm').
ENGINE=${1:-vllm}
if [ "$#" -gt 0 ] && [ "${1}" != "--" ] && [[ "${1}" != *=* ]]; then
  ENGINE=$1
  shift
fi
export VLLM_ATTENTION_BACKEND=XFORMERS

# =========================================================================== #
# GiGPO -- Group-in-Group Policy Optimization for multi-turn coding agents.
#   adv_estimator = gigpo  (episode A^E + w · step A^S grouped by anchor state)
#   anchor        = post-step workspace code (no history)
#   policy loss   = vanilla PPO-clip (same as GRPO path)
# Hydra overrides from launch_gigpo_* scripts are appended via $@.
# =========================================================================== #

num_cpus_per_env_worker=1.0

train_data_size=${TRAIN_BATCH_SIZE:-8}
val_data_size=${VAL_DATA_SIZE:-${TRAIN_BATCH_SIZE:-8}}
group_size=${GROUP_SIZE:-8}
mode="mean_norm"              # GiGPO group baseline: "mean_norm" or "mean_std_norm"

# One-line benchmark switch (resolves dataset + backend automatically):
#   local              -> synthetic CPU self-repair bed (no Docker, runnable now)
#   apps_train_coderl  -> CodeRL APPS train (common RL recipe)
#   swe_bench_verified / swe_bench_lite / r2e_gym_* -> need r2egym + Docker
benchmark="local"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=False \
    algorithm.gigpo.similarity_thresh=0.95 \
    env.env_name=swebench \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.swebench.benchmark=$benchmark \
    env.swebench.max_turns=30 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gigpo_swebench' \
    trainer.experiment_name='gigpo_qwen2.5_coder_7b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
