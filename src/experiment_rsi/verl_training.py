"""Host records -> native VeRL V1 GRPO/OPSD -> reloadable PEFT adapter."""

import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

VERL_COMMIT = "9268aad322df93cc61e9c168d66d395256dbe87a"


def _model_path(config):
    path = Path(config["model"]["path"]).expanduser()
    if path.exists():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=config["model"]["path"], revision=config["model"]["revision"])


def _merge_teacher_snapshot(config, model_path, adapter_path, output_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, config["model"]["dtype"])
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    PeftModel.from_pretrained(base, adapter_path).merge_and_unload().save_pretrained(output_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_path)


def dataset_rows(samples, objective, tokenizer):
    """One GRPO row is a group; each rollout session is one research trajectory."""
    groups = OrderedDict()
    for index, sample in enumerate(samples):
        metrics = sample["task_metrics"]
        if not sample["prompt_ids"] or not sample["generated_ids"]:
            raise ValueError("replay requires nonempty prompt and response tokens")
        if len(sample["generated_ids"]) != len(sample["behavior_logprobs"]):
            raise ValueError("response tokens and behavior log-probabilities must align")
        group_id = sample["group_id"] if objective == "grpo" else str(index)
        trajectory_id = metrics.get("trajectory_id", str(index))
        groups.setdefault(group_id, OrderedDict()).setdefault(trajectory_id, []).append(sample)

    rows = []
    for group_id, trajectories in groups.items():
        if objective == "grpo" and len(trajectories) < 2:
            raise ValueError("GRPO requires at least two trajectories in group {}".format(group_id))
        replay = []
        first = next(iter(trajectories.values()))[0]
        for trajectory in trajectories.values():
            if len({sample["reward"] for sample in trajectory}) != 1:
                raise ValueError("calls from one research trajectory must share a reward")
            weights = [sample["task_metrics"].get("sample_weight", 1.0) for sample in trajectory]
            if abs(sum(weights) - 1.0) > 1e-6 or any(weight <= 0 for weight in weights):
                raise ValueError("call weights must be positive and sum to one per trajectory")
            replay.append([
                {
                    "prompt_ids": sample["prompt_ids"],
                    "response_ids": sample["generated_ids"],
                    "response_logprobs": sample["behavior_logprobs"],
                    "reward": float(sample["reward"]),
                    "sample_weight": weight,
                }
                for sample, weight in zip(trajectory, weights)
            ])
        metrics = first["task_metrics"]
        messages = metrics.get("prompt_messages")
        if messages is None:
            # Historical experiment records store rendered tokens, not chat messages.
            # This is display metadata; actor inputs always use the original replay IDs.
            messages = [{"role": "user", "content": tokenizer.decode(first["prompt_ids"], skip_special_tokens=False)}]
        feedback = metrics["post_feedback"] if objective == "opsd" else {}
        rows.append({
            "prompt": messages,
            "data_source": "remedy",
            "reward_model": {"ground_truth": json.dumps(feedback, ensure_ascii=False)},
            "extra_info": {"index": len(rows), "problem": json.dumps(messages, ensure_ascii=False)},
            "agent_name": "rsi_research_replay",
            "__rollout_n__": len(replay),
            "replay_trajectories": json.dumps(replay),
        })
    return rows


def build_verl_overrides(config, dataset_path, model_path, teacher_path, adapter_path,
                         row_count, call_count, prompt_length, response_length, checkpoint_dir,
                         objective):
    lora = config["model"]["lora"]
    tracking = config.get("tracking", {})
    dtype = config["model"]["dtype"]
    model_dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dtype]
    token_budget = prompt_length + response_length
    logger_backends = tracking.get("logger", ["console"])
    project_name = tracking.get("project_name", "remedy")
    experiment_name = tracking.get("trainer_experiment_name", objective)
    overrides = [
        "data.train_files={}".format(dataset_path),
        "data.val_files={}".format(dataset_path),
        "data.train_batch_size={}".format(row_count),
        "data.val_batch_size={}".format(row_count),
        "data.max_prompt_length={}".format(prompt_length),
        "data.max_response_length={}".format(response_length),
        "data.filter_overlong_prompts=False",
        "data.shuffle=False",
        "data.dataloader_num_workers=0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.model.path={}".format(model_path),
        "actor_rollout_ref.model.lora_adapter_path={}".format(adapter_path or "null"),
        "actor_rollout_ref.model.lora_rank={}".format(lora["r"]),
        "actor_rollout_ref.model.lora_alpha={}".format(lora["alpha"]),
        "actor_rollout_ref.model.target_modules=[{}]".format(",".join(lora["target_modules"])),
        "actor_rollout_ref.model.enable_gradient_checkpointing={}".format(
            str(config["training"]["gradient_checkpointing"]).lower()),
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.fsdp_config.model_dtype={}".format(model_dtype),
        "actor_rollout_ref.actor.fsdp_config.dtype={}".format(dtype),
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size={}".format(call_count),
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu={}".format(token_budget),
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.optim.lr={}".format(config["training"]["learning_rate"]),
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]",
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.temperature={}".format(config["model"]["temperature"]),
        "actor_rollout_ref.rollout.top_p={}".format(config["model"]["top_p"]),
        "actor_rollout_ref.rollout.dtype={}".format(dtype),
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.max_model_len={}".format(token_budget),
        "actor_rollout_ref.rollout.max_num_batched_tokens={}".format(token_budget),
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.agent.default_agent_loop=rsi_research_replay",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path={}".format(
            Path(__file__).with_name("verl_agent_loop.yaml").resolve()),
        "trainer.use_v1=True",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.save_freq=1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.resume_mode=disable",
        "trainer.logger=[{}]".format(",".join(logger_backends)),
        "trainer.project_name={}".format(project_name),
        "trainer.experiment_name={}".format(experiment_name),
        "trainer.default_local_dir={}".format(checkpoint_dir),
        "ray_kwargs.ray_init.num_cpus=8",
        "+ray_kwargs.ray_init.include_dashboard=False",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
    ]
    if objective == "opsd":
        overrides += [
            "distillation.enabled=True",
            "distillation.n_gpus_per_node=1",
            "distillation.nnodes=1",
            "distillation.teacher_key=data_source",
            "distillation.self_distillation=True",
            "distillation.privileged_solution_key=reward_model.ground_truth",
            "distillation.privileged_problem_key=extra_info.problem",
            "distillation.privileged_mode=chat_turn",
            "distillation.privileged_user_template='Research task: {problem} | Host feedback: {solution} | Use this feedback to improve the candidate.'",
            "distillation.privileged_enable_thinking=False",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk={}".format(config["training"]["opsd_top_k"]),
            "distillation.distillation_loss.use_task_rewards=False",
            "distillation.distillation_loss.use_policy_gradient=False",
            "distillation.teacher_models.teacher_model.key=remedy",
            "distillation.teacher_models.teacher_model.model_path={}".format(teacher_path),
            "distillation.teacher_models.teacher_model.num_replicas=1",
            "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
            "distillation.teacher_models.teacher_model.inference.dtype={}".format(dtype),
            "distillation.teacher_models.teacher_model.inference.max_num_seqs=8",
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5",
        ]
    return overrides


def train_with_verl(config, samples, out_dir, objective, adapter_path=None):
    if sys.version_info < (3, 11):
        raise RuntimeError("the pinned VeRL trainer requires Python 3.11+")
    if objective not in ("grpo", "opsd"):
        raise ValueError("unknown objective: {}".format(objective))
    if not samples or len({s["role"] for s in samples}) != 1:
        raise ValueError("one update must contain samples from exactly one role")
    if objective == "opsd" and samples[0]["role"] != "edit":
        raise ValueError("OPSD is defined for the edit role")
    version = str(adapter_path) if adapter_path else "base"
    if {s["adapter_version"] for s in samples} != {version}:
        raise ValueError("training adapter must match the collection adapter")
    if {s["model_revision"] for s in samples} != {str(config["model"]["revision"])}:
        raise ValueError("training model revision must match the collection revision")
    if {s["sampling_parameters"]["temperature"] for s in samples} != {config["model"]["temperature"]}:
        raise ValueError("one update must use the collection temperature")
    if objective == "grpo" and config["model"]["temperature"] <= 0:
        raise ValueError("GRPO requires stochastic collection")
    if config["model"]["lora"]["dropout"] != 0:
        raise ValueError("this VeRL adapter uses zero LoRA dropout")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = dataset_rows(samples, objective, tokenizer)
    dataset_path = out_dir / "samples.parquet"
    pq.write_table(pa.Table.from_pylist(rows), dataset_path)
    teacher_path = model_path
    if adapter_path:
        adapter_path = str(Path(adapter_path).resolve())
        if objective == "opsd":
            teacher_path = str(out_dir / "teacher_snapshot")
            _merge_teacher_snapshot(config, model_path, adapter_path, teacher_path)
    checkpoint_dir = out_dir / "verl_checkpoints"
    overrides = build_verl_overrides(
        config, dataset_path, model_path, teacher_path, adapter_path, len(rows), len(samples),
        max(len(s["prompt_ids"]) for s in samples), max(len(s["generated_ids"]) for s in samples),
        checkpoint_dir, objective,
    )
    tracking = config.get("tracking", {})
    project_name = tracking.get("project_name", "remedy")
    experiment_name = tracking.get("trainer_experiment_name", objective)
    env = os.environ.copy()
    tag = hashlib.sha1(str(out_dir).encode()).hexdigest()[:8]
    env["RAY_TMPDIR"] = "/dev/shm/rmd-{}".format(tag)
    env["TRITON_CACHE_DIR"] = str(out_dir / "triton")
    env["HYDRA_FULL_ERROR"] = "1"
    if tracking.get("swanlab_log_dir"):
        env["SWANLAB_LOG_DIR"] = tracking["swanlab_log_dir"]
    command = [sys.executable, "-m", "experiment_rsi.verl_runtime", *overrides,
               "hydra.run.dir={}".format(out_dir / "hydra")]
    with (out_dir / "trainer.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    step = int((checkpoint_dir / "latest_checkpointed_iteration.txt").read_text().strip())
    actor_checkpoint = checkpoint_dir / "global_step_{}".format(step) / "actor"
    export_dir = out_dir / "adapter_export"
    with (out_dir / "export.log").open("w", encoding="utf-8") as log:
        subprocess.run([
            sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
            "--local_dir", str(actor_checkpoint), "--target_dir", str(export_dir),
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    adapter_dir = export_dir / "lora_adapter"
    from safetensors.torch import load_file
    import torch

    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    non_finite = [name for name, value in weights.items() if not torch.isfinite(value).all()]
    if non_finite:
        raise RuntimeError(
            "VeRL exported adapter contains non-finite weights: {}".format(non_finite[0])
        )
    if adapter_path:
        before = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
        change = sum(float((v.float() - before[k].float()).abs().sum()) for k, v in weights.items())
    else:
        # PEFT initializes B to zero. A has zero gradient on the first step;
        # weight decay is disabled, so B measures this first update.
        change = sum(float(v.float().abs().sum()) for k, v in weights.items() if "lora_B" in k)
    summary = {
        "backend": "verl", "verl_commit": VERL_COMMIT, "objective": objective,
        "role": samples[0]["role"], "sample_count": len(samples),
        "group_count": len({s["group_id"] for s in samples}), "trainer_step": step,
        "trainable_parameter_count": sum(v.numel() for v in weights.values()),
        "trainable_parameter_l1_change": change, "parameter_update": change > 0,
        "run_id": tracking.get("run_id"), "period": tracking.get("period"),
        "trainer_experiment_name": experiment_name, "project_name": project_name,
        "dataset": str(dataset_path), "trainer_log": str(out_dir / "trainer.log"),
        "trainer_checkpoint": str(actor_checkpoint), "adapter": str(adapter_dir),
    }
    (out_dir / "train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
