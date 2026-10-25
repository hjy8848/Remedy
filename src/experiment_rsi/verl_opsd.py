import json
import subprocess
import sys
from pathlib import Path

VERL_COMMIT = "9268aad322df93cc61e9c168d66d395256dbe87a"


def _model_path(config):
    model_path = config["model"]["path"]
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path.resolve())

    from huggingface_hub import snapshot_download

    revision = config["model"].get("revision", "local")
    return snapshot_download(
        repo_id=model_path,
        revision=None if revision in (None, "local", "unspecified") else revision,
    )


def _merge_teacher_snapshot(config, model_path, adapter_path, output_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config["model"]["dtype"]]
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    tokenizer.save_pretrained(output_path)


def _dataset_rows(samples):
    group_indices = {group_id: index for index, group_id in enumerate(dict.fromkeys(s["group_id"] for s in samples))}
    rows = []
    for sample in samples:
        metrics = sample["task_metrics"]
        prompt_ids = sample["prompt_ids"]
        response_ids = sample["generated_ids"]
        response_logprobs = sample["behavior_logprobs"]
        if not prompt_ids or not response_ids:
            raise ValueError("VeRL replay samples must contain prompt and response tokens")
        if len(response_ids) != len(response_logprobs):
            raise ValueError("sample response tokens and behavior log-probabilities must have equal lengths")
        rows.append(
            {
                "prompt": metrics["prompt_messages"],
                "data_source": "rsi_opsd",
                "reward_model": {
                    "ground_truth": json.dumps(metrics["post_feedback"], ensure_ascii=False, sort_keys=True)
                },
                "extra_info": {
                    "index": group_indices[sample["group_id"]],
                    "problem": json.dumps(metrics["prompt_messages"], ensure_ascii=False),
                },
                "agent_name": "rsi_research_replay",
                "replay_prompt_ids": [int(token) for token in prompt_ids],
                "replay_response_ids": [int(token) for token in response_ids],
                "replay_response_logprobs": [float(value) for value in response_logprobs],
                "rsi_reward": float(sample["reward"]),
            }
        )
    return rows


def build_verl_overrides(config, dataset_path, model_path, teacher_model_path, adapter_path,
                         sample_count, prompt_length, response_length, checkpoint_dir, agent_loop_config):
    token_budget = prompt_length + response_length
    lora = config["model"]["lora"]
    target_modules = "[{}]".format(",".join(lora["target_modules"]))
    model_dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[config["model"]["dtype"]]
    train_dtype = {"float16": "float16", "bfloat16": "bfloat16", "float32": "bfloat16"}[config["model"]["dtype"]]

    return [
        "data.train_files={}".format(dataset_path),
        "data.val_files={}".format(dataset_path),
        "data.train_batch_size={}".format(sample_count),
        "data.val_batch_size={}".format(sample_count),
        "data.max_prompt_length={}".format(prompt_length),
        "data.max_response_length={}".format(response_length),
        "data.filter_overlong_prompts=False",
        "data.shuffle=False",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.model.path={}".format(model_path),
        "actor_rollout_ref.model.lora_adapter_path={}".format(adapter_path or "null"),
        "actor_rollout_ref.model.lora_rank={}".format(lora["r"]),
        "actor_rollout_ref.model.lora_alpha={}".format(lora["alpha"]),
        "actor_rollout_ref.model.target_modules={}".format(target_modules),
        "actor_rollout_ref.model.enable_gradient_checkpointing={}".format(
            str(config["training"]["gradient_checkpointing"]).lower()
        ),
        "actor_rollout_ref.actor.fsdp_config.model_dtype={}".format(model_dtype),
        "actor_rollout_ref.actor.fsdp_config.dtype={}".format(train_dtype),
        "actor_rollout_ref.actor.ppo_mini_batch_size={}".format(sample_count),
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu={}".format(token_budget),
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.optim.lr={}".format(config["training"]["learning_rate"]),
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]",
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.dtype={}".format(train_dtype),
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        "actor_rollout_ref.rollout.agent.default_agent_loop=rsi_research_replay",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path={}".format(agent_loop_config),
        "distillation.enabled=True",
        "distillation.n_gpus_per_node=1",
        "distillation.nnodes=1",
        "distillation.teacher_key=data_source",
        "distillation.self_distillation=True",
        "distillation.privileged_solution_key=reward_model.ground_truth",
        "distillation.privileged_problem_key=extra_info.problem",
        "distillation.privileged_mode=chat_turn",
        "distillation.privileged_user_template='Research task and instructions: {problem} | Verified environment feedback from executing the candidate: {solution} | Use the feedback to improve the candidate.'",
        "distillation.privileged_enable_thinking=True",
        "distillation.distillation_loss.loss_mode=forward_kl_topk",
        "distillation.distillation_loss.topk={}".format(config["training"]["opsd_top_k"]),
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.use_policy_gradient=False",
        "distillation.teacher_models.teacher_model.key=rsi_opsd",
        "distillation.teacher_models.teacher_model.model_path={}".format(teacher_model_path),
        "distillation.teacher_models.teacher_model.num_replicas=1",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.dtype={}".format(train_dtype),
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3",
        "distillation.teacher_models.teacher_model.inference.max_num_seqs=32",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.save_freq=1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.resume_mode=disable",
        "trainer.logger=[console]",
        "trainer.project_name=experiment_rsi",
        "trainer.experiment_name=opsd",
        "trainer.default_local_dir={}".format(checkpoint_dir),
    ]


def train_opsd_with_verl(config, samples, out_dir, adapter_path=None):
    if not samples or any(sample["role"] != "edit" for sample in samples):
        raise ValueError("OPSD trains edit-role samples only")
    if any("post_feedback" not in sample["task_metrics"] or "prompt_messages" not in sample["task_metrics"] for sample in samples):
        raise ValueError("OPSD samples must contain the original prompt and post-execution feedback")
    adapter_version = str(adapter_path) if adapter_path else "base"
    if {sample["adapter_version"] for sample in samples} != {adapter_version}:
        raise ValueError("training adapter must match the adapter used to collect these samples")
    model_revision = str(config["model"].get("revision", "unspecified"))
    if {sample["model_revision"] for sample in samples} != {model_revision}:
        raise ValueError("training model revision must match the revision used to collect these samples")
    if config["model"]["lora"]["dropout"] != 0:
        raise ValueError("VeRL FSDP LoRA does not expose the configured LoRA dropout setting")
    if adapter_path:
        adapter_path = str(Path(adapter_path).resolve())

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "opsd_samples.parquet"
    pq.write_table(pa.Table.from_pylist(_dataset_rows(samples)), dataset_path)

    model_path = _model_path(config)
    teacher_model_path = model_path
    if adapter_path:
        teacher_model_path = str(out_dir / "teacher_snapshot")
        _merge_teacher_snapshot(config, model_path, adapter_path, teacher_model_path)

    prompt_length = max(len(sample["prompt_ids"]) for sample in samples)
    response_length = max(len(sample["generated_ids"]) for sample in samples)
    sample_count = len(samples)
    checkpoint_dir = out_dir / "verl_checkpoints"
    agent_loop_config = Path(__file__).with_name("verl_agent_loop.yaml").resolve()
    overrides = build_verl_overrides(
        config,
        dataset_path,
        model_path,
        teacher_model_path,
        adapter_path,
        sample_count,
        prompt_length,
        response_length,
        checkpoint_dir,
        agent_loop_config,
    )
    subprocess.run([sys.executable, "-m", "verl.trainer.main_ppo", *overrides], check=True)

    latest_step = (checkpoint_dir / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    actor_checkpoint = checkpoint_dir / "global_step_{}".format(latest_step) / "actor"
    adapter_export = out_dir / "adapter_export"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor_checkpoint),
            "--target_dir",
            str(adapter_export),
        ],
        check=True,
    )
    adapter_dir = adapter_export / "lora_adapter"
    summary = {
        "role": "edit",
        "objective": "opsd",
        "verl_commit": VERL_COMMIT,
        "opsd_backend": "verl_6833_native_trainer",
        "opsd_top_k": config["training"]["opsd_top_k"],
        "sample_count": sample_count,
        "dataset": str(dataset_path),
        "teacher_model": teacher_model_path,
        "trainer_checkpoint": str(actor_checkpoint),
        "adapter": str(adapter_dir.resolve()),
    }
    (out_dir / "train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
