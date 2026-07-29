set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# =========================================================================== #
# DAPO -- Decoupled clip & Dynamic sAmpling Policy Optimization on SWE-bench.
#   https://arxiv.org/abs/2503.14476
#
#   Active in this fork (really wired):
#     * clip-higher        : asymmetric clip_ratio_low/high  (decoupled clip)
#     * token-mean loss    : actor.loss_agg_mode=token-mean
#     * (overlong handling) : data.truncation='error' guards over-long prompts
#
#   !!! CAVEAT -- dynamic sampling is NOT wired in this fork !!!
#   algorithm.filter_groups.{enable,max_num_gen_batches} are declared in
#   ppo_trainer.yaml but the training loop in verl/trainer/ppo/ray_trainer.py
#   does NOT consume them (no group-resampling loop). The flags below are kept
#   for forward-compatibility and to match the repo convention, but setting
#   enable=True currently has NO effect. To get true DAPO dynamic sampling the
#   resampling loop must be added to ray_trainer.py first (GPU-tested).
# =========================================================================== #

num_cpus_per_env_worker=1.0

train_data_size=8
val_data_size=8
group_size=8

clip_ratio_low=0.2
clip_ratio_high=0.28
enable_filter_groups=False   # see CAVEAT above: currently a no-op in this fork
max_num_gen_batches=10

# One-line benchmark switch (see ../grpo_trainer/run_swebench.sh for presets).
benchmark="local"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
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
    algorithm.filter_groups.enable=$enable_filter_groups \
    algorithm.filter_groups.max_num_gen_batches=$max_num_gen_batches \
    env.env_name=swebench \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.swebench.benchmark=$benchmark \
    env.swebench.max_turns=30 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='dapo_swebench' \
    trainer.experiment_name='dapo_qwen2.5_coder_7b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
