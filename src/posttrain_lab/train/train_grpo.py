"""Minimal GRPO smoke training using boxed-answer math rewards."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import statistics
import time
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl
from posttrain_lab.eval.eval_runner import run_eval
from posttrain_lab.rewards.math_reward import (
    REWARD_VERSION,
    SUPPORTED_REWARD_VERSIONS,
    MathRewardConfig,
    math_boxed_v001,
    score_math_boxed_by_version,
)
from posttrain_lab.train.train_sft import (
    _dump_yaml,
    _generation_prompt_to_text,
    _git_commit,
    _load_yaml_subset,
    _model_load_kwargs,
    _sha256_file,
    _write_jsonl,
    _write_text,
    write_cli_dry_run_plan,
)


REQUIRED_TOP_LEVEL = {
    "model_name_or_path",
    "data_path",
    "output_dir",
    "dry_run",
    "seed",
    "selection",
    "training",
    "rollout",
    "reward_version",
}

PARSE_FAILURE_REASONS = {
    "no_boxed_answer",
    "malformed_boxed_answer",
    "empty_boxed_answer",
    "conflicting_boxed_answers",
    "multiple_boxed_answers",
    "boxed_not_final_only",
    "unclosed_think_block",
    "output_too_long",
}


def load_config(path):
    """Load and validate a shallow GRPO smoke YAML config."""

    config = _load_yaml_subset(Path(path))
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"missing required config fields: {', '.join(missing)}")
    return config


def load_rlvr_train_examples(path, split="train", limit=4):
    """Validate and load exactly ``limit`` RLVR examples from a split."""

    errors = validate_jsonl("rlvr", path)
    if errors:
        raise ValueError("\n".join(str(error) for error in errors))

    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            if record["split"] == split:
                examples.append(record)
            if len(examples) == limit:
                break

    if len(examples) != limit:
        raise ValueError(f"expected exactly {limit} {split} examples, found {len(examples)}")
    return examples


def math_boxed_reward_func(prompts, completions, answer, **kwargs):
    """TRL custom reward function for math_boxed_v001."""

    del prompts, kwargs
    return [
        math_boxed_v001(_completion_to_text(completion), target)
        for completion, target in zip(completions, answer)
    ]


def make_math_boxed_reward_func(reward_config, reward_version=REWARD_VERSION):
    """Build a TRL reward function with an explicit math reward config."""

    def _reward_func(prompts, completions, answer, **kwargs):
        del prompts, kwargs
        return [
            score_math_boxed_by_version(
                _completion_to_text(completion),
                target,
                reward_version=reward_version,
                config=reward_config,
            ).score
            for completion, target in zip(completions, answer)
        ]

    return _reward_func


def run_grpo(config, config_path):
    """Run a dry-run or real TRL GRPO smoke experiment."""

    resolved = _resolve_config(config)
    _set_local_cuda_device()
    output_dir = Path(resolved["output_dir"])
    data_path = Path(resolved["data_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolved.get("synthetic_data_if_missing") and not data_path.exists():
        _write_synthetic_rlvr_jsonl(
            data_path,
            train_count=resolved["selection"]["max_train_examples"],
            problem_style=resolved["synthetic_problem_style"],
        )

    examples = load_rlvr_train_examples(
        data_path,
        split=resolved["selection"]["train_split"],
        limit=resolved["selection"]["max_train_examples"],
    )
    data_hash = _sha256_file(data_path)
    config_hash = _sha256_file(config_path)
    git_commit = _git_commit()

    if _is_main_process():
        _write_text(output_dir / "resolved_config.yaml", _dump_yaml(resolved))

    gate_metrics = _run_rollout_format_gate(resolved, output_dir, examples)
    if gate_metrics and not _rollout_format_gate_passed(resolved, gate_metrics):
        failure_reason = _rollout_format_gate_failure_reason(resolved, gate_metrics)
        if _is_main_process():
            _write_jsonl(
                output_dir / "metrics.jsonl",
                [
                    {
                        "step": 0,
                        "final_loss": None,
                        "reward_version": resolved["reward_version"],
                        "dry_run": resolved["dry_run"],
                        "smoke_run": resolved["smoke_run"],
                        "blocked_by_rollout_format_gate": True,
                        "rollout_format_gate_failure_reason": failure_reason,
                        **_prefixed_metrics("rollout_format_gate", gate_metrics),
                    }
                ],
            )
        raise RuntimeError(f"rollout-format gate failed: {failure_reason}")

    if resolved["dry_run"]:
        trainer_log = [{"step": 0, "loss": 0.0}, {"step": int(resolved["training"]["max_steps"]), "loss": 0.0}]
        _write_jsonl(output_dir / "trainer_log.jsonl", trainer_log)
        sample_rows = _write_sample_rollouts(
            output_dir / "sample_rollouts.jsonl",
            resolved,
            examples=examples,
            dry_run=True,
        )
        final_loss = 0.0
    else:
        final_loss, trainer_log, model, tokenizer = _run_trl_grpo(resolved, examples, output_dir)
        if not _is_main_process():
            return {
                "output_dir": str(output_dir),
                "rank": _distributed_rank(),
                "world_size": _distributed_world_size(),
                "git_commit": git_commit,
            }
        _write_jsonl(output_dir / "trainer_log.jsonl", trainer_log)
        sample_rows = _write_sample_rollouts(
            output_dir / "sample_rollouts.jsonl",
            resolved,
            examples=examples,
            dry_run=False,
            model=model,
            tokenizer=tokenizer,
        )
        del model
        del tokenizer
        _empty_cuda_cache()

    eval_outputs = _run_eval_after_grpo(resolved, output_dir)
    metrics = _summarize_samples(sample_rows)
    metrics.update(_summarize_trainer_signal(trainer_log))
    metrics.update(
        {
            "step": int(resolved["training"]["max_steps"]),
            "final_loss": final_loss,
            "reward_version": resolved["reward_version"],
            "dry_run": resolved["dry_run"],
            "smoke_run": resolved["smoke_run"],
        }
    )
    if gate_metrics:
        metrics.update(_prefixed_metrics("rollout_format_gate", gate_metrics))
    _write_jsonl(output_dir / "metrics.jsonl", [metrics])
    _write_text(output_dir / "eval_report.json", json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_run_card(
        output_dir / "run_card.md",
        resolved=resolved,
        data_hash=data_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        final_loss=final_loss,
        metrics=metrics,
        eval_outputs=eval_outputs,
    )

    result = dict(metrics)
    result.update(
        {
            "output_dir": str(output_dir),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "eval_outputs": eval_outputs,
        }
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a minimal GRPO smoke experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Write a dry-run plan and do not train.")
    parser.add_argument("--output-dir", help="Override output directory for --dry-run plan artifacts.")
    args = parser.parse_args(argv)

    if args.dry_run:
        result = write_cli_dry_run_plan(args.config, args.output_dir, mode="grpo")
        print(json.dumps(result, sort_keys=True))
        return 0

    config = load_config(args.config)
    result = run_grpo(config, config_path=args.config)
    print(json.dumps(result, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = copy.deepcopy(config)
    resolved.setdefault("run_name", "grpo-smoke")
    resolved.setdefault("smoke_run", True)
    resolved.setdefault("synthetic_data_if_missing", False)
    resolved.setdefault("synthetic_problem_style", "addition")
    resolved.setdefault("torch_dtype", "auto")
    resolved.setdefault("trust_remote_code", False)
    resolved.setdefault("enable_thinking", None)
    resolved.setdefault("adapter_path", None)
    resolved["selection"].setdefault("train_split", "train")
    resolved["selection"].setdefault("max_train_examples", 4)
    resolved["training"].setdefault("max_steps", 1)
    resolved["training"].setdefault("per_device_train_batch_size", 2)
    resolved["training"].setdefault("gradient_accumulation_steps", 1)
    resolved["training"].setdefault("learning_rate", 1e-6)
    resolved["training"].setdefault("bf16", False)
    resolved["training"].setdefault("fp16", False)
    resolved["training"].setdefault("gradient_checkpointing", False)
    resolved["training"].setdefault("gradient_checkpointing_kwargs", None)
    resolved["training"].setdefault("logging_steps", 1)
    resolved["training"].setdefault("save_steps", resolved["training"]["max_steps"])
    resolved["training"].setdefault("save_total_limit", 1)
    resolved["training"].setdefault("optim", None)
    resolved["training"].setdefault("lr_scheduler_type", None)
    resolved["training"].setdefault("warmup_ratio", None)
    resolved["training"].setdefault("max_grad_norm", None)
    resolved["training"].setdefault("dataloader_num_workers", 0)
    resolved["training"].setdefault("remove_unused_columns", False)
    resolved["training"].setdefault("frac_reward_zero_std_early_stop", False)
    resolved["training"].setdefault("frac_reward_zero_std_threshold", 0.8)
    resolved["training"].setdefault("frac_reward_zero_std_patience", 20)
    resolved["rollout"].setdefault("num_generations", 2)
    resolved["rollout"].setdefault("max_completion_length", 16)
    resolved["rollout"].setdefault("max_prompt_length", None)
    resolved["rollout"].setdefault("temperature", 0.7)
    resolved["rollout"].setdefault("top_p", 1.0)
    resolved["rollout"].setdefault("top_k", 0)
    resolved["rollout"].setdefault("beta", 0.0)
    resolved["rollout"].setdefault("epsilon", 0.2)
    resolved["rollout"].setdefault("epsilon_high", None)
    resolved["rollout"].setdefault("num_iterations", 1)
    resolved["rollout"].setdefault("loss_type", "dapo")
    resolved["rollout"].setdefault("scale_rewards", "group")
    resolved["rollout"].setdefault("steps_per_generation", None)
    resolved["rollout"].setdefault("generation_batch_size", None)
    resolved["rollout"].setdefault("ds3_gather_for_generation", True)
    resolved["rollout"].setdefault("mask_truncated_completions", False)
    resolved["rollout"].setdefault("use_vllm", False)
    resolved["rollout"].setdefault("vllm_mode", "colocate")
    resolved["rollout"].setdefault("vllm_gpu_memory_utilization", 0.3)
    resolved["rollout"].setdefault("vllm_max_model_length", None)
    resolved["rollout"].setdefault("sample_count", 4)
    resolved["rollout"].setdefault("sample_rollout_generations", 1)
    resolved.setdefault("rollout_format_gate", {})
    resolved["rollout_format_gate"].setdefault("enabled", False)
    resolved["rollout_format_gate"].setdefault("sample_count", resolved["rollout"]["sample_count"])
    resolved["rollout_format_gate"].setdefault("max_parse_failure_rate", 0.0)
    resolved["rollout_format_gate"].setdefault("max_reward_mean", None)
    resolved["rollout_format_gate"].setdefault("max_perfect_reward_rate", None)
    resolved.setdefault("reward", {})
    resolved["reward"].setdefault("allow_symbolic_equivalence", False)
    resolved["reward"].setdefault("symbolic_equivalence_engine", "fraction")
    resolved["reward"].setdefault("max_symbolic_expr_chars", 120)
    resolved["reward"].setdefault("max_symbolic_ast_nodes", 64)
    resolved["reward"].setdefault("max_symbolic_collection_size", 32)
    resolved.setdefault("eval_after_train", {})
    resolved["eval_after_train"].setdefault("enabled", False)
    resolved["eval_after_train"].setdefault("prompt_path", "/tmp/posttrain_lab_eval/baseline_prompts.jsonl")
    resolved["eval_after_train"].setdefault("output_root", str(Path(resolved["output_dir"]) / "evals"))
    resolved["eval_after_train"].setdefault("dry_run", resolved["dry_run"])
    resolved["eval_after_train"].setdefault("model_name", resolved["model_name_or_path"])
    resolved["eval_after_train"].setdefault("torch_dtype", resolved["torch_dtype"])
    resolved["eval_after_train"].setdefault("trust_remote_code", resolved["trust_remote_code"])
    resolved["eval_after_train"].setdefault("apply_chat_template", True)
    resolved["eval_after_train"].setdefault("enable_thinking", resolved.get("enable_thinking"))
    resolved["eval_after_train"].setdefault("temperature", 0.0)
    resolved["eval_after_train"].setdefault("top_p", 1.0)
    resolved["eval_after_train"].setdefault("max_new_tokens", resolved["rollout"]["max_completion_length"])
    resolved["eval_after_train"].setdefault("format_regex", r"^\\boxed\{.+\}$")
    resolved["eval_after_train"].setdefault("exact_match", True)
    resolved.setdefault("peft", {})
    resolved["peft"].setdefault("method", "lora")
    resolved["peft"].setdefault("r", 4)
    resolved["peft"].setdefault("lora_alpha", 8)
    resolved["peft"].setdefault("lora_dropout", 0.0)
    resolved["peft"].setdefault("target_modules", [])

    if resolved["reward_version"] not in SUPPORTED_REWARD_VERSIONS:
        raise ValueError(f"unsupported reward_version: {resolved['reward_version']}")
    if int(resolved["rollout"]["num_generations"]) < 2:
        raise ValueError("GRPO smoke requires rollout.num_generations >= 2")
    if int(resolved["training"]["per_device_train_batch_size"]) % int(resolved["rollout"]["num_generations"]) != 0:
        raise ValueError("per_device_train_batch_size must be divisible by rollout.num_generations")
    return resolved


def _run_trl_grpo(config, examples, output_dir):
    _install_trl_fsdp_compat()
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("TRL GRPO training requires transformers, datasets, trl, and peft") from exc

    model, tokenizer, peft_config = _load_policy_model_and_tokenizer(config)
    reward_func = make_math_boxed_reward_func(_math_reward_config(config), config["reward_version"])
    train_dataset = Dataset.from_list(_to_grpo_rows(examples))
    args = GRPOConfig(**_build_grpo_config_kwargs(config, output_dir))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    _add_trainer_callbacks(trainer, _trainer_callbacks(config))
    result = trainer.train()
    trainer.save_model(str(output_dir))
    return float(result.training_loss), trainer.state.log_history, trainer.model, tokenizer


def _build_grpo_config_kwargs(config, output_dir, supported_fields=None):
    """Build TRL GRPOConfig kwargs while tolerating minor TRL version drift."""

    if supported_fields is None:
        try:
            from trl import GRPOConfig
        except ImportError as exc:
            raise RuntimeError("TRL GRPO training requires trl") from exc
        supported_fields = set(inspect.signature(GRPOConfig.__init__).parameters)

    kwargs = {
        "output_dir": str(output_dir),
        "max_steps": int(config["training"]["max_steps"]),
        "per_device_train_batch_size": int(config["training"]["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "logging_steps": int(config["training"]["logging_steps"]),
        "logging_first_step": True,
        "save_steps": int(config["training"]["save_steps"]),
        "save_total_limit": int(config["training"]["save_total_limit"]),
        "save_strategy": "steps",
        "report_to": "none",
        "seed": int(config["seed"]),
        "bf16": bool(config["training"]["bf16"]),
        "fp16": bool(config["training"]["fp16"]),
        "gradient_checkpointing": bool(config["training"]["gradient_checkpointing"]),
        "num_generations": int(config["rollout"]["num_generations"]),
        "max_completion_length": int(config["rollout"]["max_completion_length"]),
        "temperature": float(config["rollout"]["temperature"]),
        "top_p": float(config["rollout"]["top_p"]),
        "top_k": int(config["rollout"]["top_k"]),
        "beta": float(config["rollout"]["beta"]),
        "log_completions": True,
        "num_completions_to_print": int(config["rollout"]["num_generations"]),
        "model_init_kwargs": None,
        "chat_template_kwargs": _chat_template_kwargs(config),
    }
    optional = {
        "optim": config["training"].get("optim"),
        "lr_scheduler_type": config["training"].get("lr_scheduler_type"),
        "warmup_ratio": config["training"].get("warmup_ratio"),
        "max_grad_norm": config["training"].get("max_grad_norm"),
        "dataloader_num_workers": config["training"].get("dataloader_num_workers"),
        "remove_unused_columns": config["training"].get("remove_unused_columns"),
        "gradient_checkpointing_kwargs": config["training"].get("gradient_checkpointing_kwargs"),
        "max_prompt_length": config["rollout"].get("max_prompt_length"),
        "epsilon": config["rollout"].get("epsilon"),
        "epsilon_high": config["rollout"].get("epsilon_high"),
        "num_iterations": config["rollout"].get("num_iterations"),
        "loss_type": config["rollout"].get("loss_type"),
        "scale_rewards": config["rollout"].get("scale_rewards"),
        "steps_per_generation": config["rollout"].get("steps_per_generation"),
        "generation_batch_size": config["rollout"].get("generation_batch_size"),
        "ds3_gather_for_generation": config["rollout"].get("ds3_gather_for_generation"),
        "mask_truncated_completions": config["rollout"].get("mask_truncated_completions"),
        "use_vllm": config["rollout"].get("use_vllm"),
        "vllm_mode": config["rollout"].get("vllm_mode"),
        "vllm_gpu_memory_utilization": config["rollout"].get("vllm_gpu_memory_utilization"),
        "vllm_max_model_length": config["rollout"].get("vllm_max_model_length"),
    }
    for key, value in optional.items():
        if value is not None:
            kwargs[key] = value

    return {
        key: value
        for key, value in kwargs.items()
        if key in supported_fields and key != "self"
    }


def _load_policy_model_and_tokenizer(config, move_to_accelerator=False):
    _install_trl_fsdp_compat()
    try:
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("policy loading requires transformers and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **_model_load_kwargs(config))
    if config.get("adapter_path"):
        model = PeftModel.from_pretrained(model, config["adapter_path"], is_trainable=True)
        if move_to_accelerator:
            model = _move_model_to_accelerator(model)
        return model, tokenizer, None

    method = str(config["peft"].get("method", "lora")).lower().replace("-", "_")
    if method in {"none", "full", "full_finetune", "full_finetuning"}:
        if move_to_accelerator:
            model = _move_model_to_accelerator(model)
        return model, tokenizer, None
    if method != "lora":
        raise ValueError(f"unsupported peft.method: {config['peft'].get('method')}")

    target_modules = config["peft"].get("target_modules") or None
    peft_config = LoraConfig(
        r=int(config["peft"]["r"]),
        lora_alpha=int(config["peft"]["lora_alpha"]),
        lora_dropout=float(config["peft"]["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    if move_to_accelerator:
        model = _move_model_to_accelerator(model)
    return model, tokenizer, peft_config


def _trainer_callbacks(config):
    if not bool(config["training"].get("frac_reward_zero_std_early_stop")):
        return None
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("transformers is required for GRPO early-stop callbacks") from exc

    threshold = float(config["training"]["frac_reward_zero_std_threshold"])
    patience = int(config["training"]["frac_reward_zero_std_patience"])

    class FracRewardZeroStdEarlyStopCallback(TrainerCallback):
        def __init__(self):
            self.consecutive_bad_steps = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            if not logs or "frac_reward_zero_std" not in logs:
                return control
            value = float(logs["frac_reward_zero_std"])
            if value > threshold:
                self.consecutive_bad_steps += 1
            else:
                self.consecutive_bad_steps = 0
            if self.consecutive_bad_steps >= patience:
                control.should_training_stop = True
            return control

    return [FracRewardZeroStdEarlyStopCallback()]


def _add_trainer_callbacks(trainer, callbacks):
    if not callbacks:
        return
    for callback in callbacks:
        if hasattr(trainer, "add_callback"):
            trainer.add_callback(callback)
        elif hasattr(trainer, "callback_handler"):
            trainer.callback_handler.add_callback(callback)
        else:
            raise RuntimeError("trainer does not support callbacks")


def _move_model_to_accelerator(model):
    try:
        import torch
    except ImportError:
        return model
    device = _local_torch_device(torch)
    if device is not None:
        return model.to(device)
    return model


def _run_rollout_format_gate(config, output_dir, examples):
    if not config["rollout_format_gate"]["enabled"]:
        return None

    metrics_path = output_dir / "rollout_format_gate.json"
    if not _is_main_process():
        return _wait_for_json(metrics_path, timeout_seconds=3600)

    gate_config = copy.deepcopy(config)
    gate_config["rollout"]["sample_count"] = int(config["rollout_format_gate"]["sample_count"])
    if config["dry_run"]:
        rows = _write_sample_rollouts(
            output_dir / "rollout_format_gate.jsonl",
            gate_config,
            examples=examples,
            dry_run=True,
        )
    else:
        model, tokenizer, _ = _load_policy_model_and_tokenizer(config, move_to_accelerator=True)
        rows = _write_sample_rollouts(
            output_dir / "rollout_format_gate.jsonl",
            gate_config,
            examples=examples,
            dry_run=False,
            model=model,
            tokenizer=tokenizer,
        )
        del model
        del tokenizer
        _empty_cuda_cache()
    metrics = _summarize_samples(rows)
    _write_text(metrics_path, json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def _rollout_format_gate_passed(config, metrics):
    return _rollout_format_gate_failure_reason(config, metrics) is None


def _rollout_format_gate_failure_reason(config, metrics):
    gate = config["rollout_format_gate"]
    parse_failure_rate = metrics["parse_failure_rate"]
    max_parse_failure_rate = float(gate["max_parse_failure_rate"])
    if parse_failure_rate > max_parse_failure_rate:
        return f"parse_failure_rate={parse_failure_rate} > {max_parse_failure_rate}"

    max_reward_mean = gate.get("max_reward_mean")
    if max_reward_mean is not None and metrics["reward_mean"] is not None:
        max_reward_mean = float(max_reward_mean)
        if metrics["reward_mean"] > max_reward_mean:
            return f"reward_mean={metrics['reward_mean']} > {max_reward_mean}"

    max_perfect_reward_rate = gate.get("max_perfect_reward_rate")
    if max_perfect_reward_rate is not None and metrics["perfect_reward_rate"] is not None:
        max_perfect_reward_rate = float(max_perfect_reward_rate)
        if metrics["perfect_reward_rate"] > max_perfect_reward_rate:
            return f"perfect_reward_rate={metrics['perfect_reward_rate']} > {max_perfect_reward_rate}"

    return None


def _prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _distributed_rank():
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _distributed_world_size():
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _is_main_process():
    return _distributed_rank() == 0


def _local_cuda_device_index(device_count):
    if device_count <= 0:
        return None
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None:
        return None
    try:
        local_rank = int(raw_rank)
    except ValueError:
        return None
    if 0 <= local_rank < device_count:
        return local_rank
    return None


def _set_local_cuda_device():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    index = _local_cuda_device_index(torch.cuda.device_count())
    if index is None:
        return None
    torch.cuda.set_device(index)
    return index


def _local_torch_device(torch_module):
    if not torch_module.cuda.is_available():
        return None
    index = _local_cuda_device_index(torch_module.cuda.device_count())
    if index is None:
        return torch_module.device("cuda")
    return torch_module.device("cuda", index)


def _wait_for_json(path, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError as exc:
            last_error = exc
        except json.JSONDecodeError as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {path}: {last_error}")


def _empty_cuda_cache():
    try:
        import gc
        gc.collect()
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_eval_after_grpo(config, output_dir):
    if not config["eval_after_train"]["enabled"]:
        return {}

    eval_config = config["eval_after_train"]
    prompt_path = Path(eval_config["prompt_path"])
    if eval_config["dry_run"] and not prompt_path.exists():
        _write_default_eval_prompts(prompt_path)

    output_root = Path(eval_config["output_root"])
    runs = {
        "base": None,
        "sft": config.get("adapter_path"),
        "sft_rlvr": str(output_dir),
    }
    results = {}
    for run_name, adapter_path in runs.items():
        run_config = _build_eval_config(config, run_name, adapter_path)
        metrics = run_eval(run_config)
        _empty_cuda_cache()
        results[run_name] = {
            "output_dir": str(output_root / run_name),
            "adapter_path": adapter_path,
            "metrics": metrics,
        }
    _write_text(output_dir / "eval_summary.json", json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def _build_eval_config(config, run_name, adapter_path):
    eval_config = config["eval_after_train"]
    return {
        "prompt_path": eval_config["prompt_path"],
        "output_dir": str(Path(eval_config["output_root"]) / run_name),
        "dry_run": eval_config["dry_run"],
        "model_name": eval_config["model_name"],
        "adapter_path": adapter_path,
        "torch_dtype": eval_config["torch_dtype"],
        "trust_remote_code": eval_config["trust_remote_code"],
        "apply_chat_template": eval_config["apply_chat_template"],
        "enable_thinking": eval_config["enable_thinking"],
        "inference": {
            "temperature": eval_config["temperature"],
            "top_p": eval_config["top_p"],
            "max_new_tokens": eval_config["max_new_tokens"],
            "stop_tokens": [],
        },
        "metrics": {
            "exact_match": eval_config["exact_match"],
            "format_regex": eval_config["format_regex"],
        },
    }


def _write_default_eval_prompts(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "baseline-001",
            "prompt": "Return the answer to 2 + 2 in boxed format.",
            "answer": r"\boxed{4}",
            "mock_generation": r"\boxed{4}",
        }
    ]
    _write_jsonl(path, rows)


def _install_trl_fsdp_compat():
    try:
        import torch.distributed.fsdp as fsdp
    except ImportError:
        return
    if not hasattr(fsdp, "FSDPModule"):
        class FSDPModule:  # noqa: N801 - mirror torch's missing public name
            pass

        fsdp.FSDPModule = FSDPModule


def _chat_template_kwargs(config):
    if config.get("enable_thinking") is None:
        return None
    return {"enable_thinking": bool(config["enable_thinking"])}


def _to_grpo_rows(examples):
    rows = []
    for example in examples:
        rows.append(
            {
                "prompt": example["prompt"],
                "answer": example["verifier"]["answer"],
                "id": example["id"],
            }
        )
    return rows


def _write_sample_rollouts(path, config, examples, dry_run, model=None, tokenizer=None):
    rows = []
    reward_config = _math_reward_config(config)
    sample_count = min(int(config["rollout"]["sample_count"]), len(examples))
    generations_per_prompt = int(config["rollout"].get("sample_rollout_generations", 1))
    selected = examples[:sample_count]
    if not dry_run and hasattr(model, "eval"):
        model.eval()

    for index, example in enumerate(selected):
        prompt = _prompt_text(example["prompt"])
        answer = example["verifier"]["answer"]
        completion_records = []
        for generation_index in range(generations_per_prompt):
            if dry_run:
                generation = rf"\boxed{{{answer}}}"
            else:
                generation = _generate_text(
                    model,
                    tokenizer,
                    prompt,
                    config,
                    sample=generations_per_prompt > 1,
                )
            reward_result = score_math_boxed_by_version(
                generation,
                answer,
                reward_version=config["reward_version"],
                config=reward_config,
            )
            completion_records.append(
                {
                    "generation_index": generation_index,
                    "completion": generation,
                    "reward": reward_result.score,
                    "parsed_answer": reward_result.normalized_prediction,
                    "failure_reason": None if reward_result.score == 1.0 else reward_result.reason,
                    "parse_failure": reward_result.reason in PARSE_FAILURE_REASONS,
                    "completion_length": len(generation),
                }
            )
        first = completion_records[0]
        rewards = [record["reward"] for record in completion_records]
        parse_failures = [record["parse_failure"] for record in completion_records]
        completion_lengths = [record["completion_length"] for record in completion_records]
        rows.append(
            {
                "id": example["id"],
                "sample_index": index,
                "prompt": prompt,
                "answer": answer,
                "completion": first["completion"],
                "reward": first["reward"],
                "parsed_answer": first["parsed_answer"],
                "failure_reason": first["failure_reason"],
                "parse_failure": any(parse_failures),
                "completion_length": sum(completion_lengths) / len(completion_lengths),
                "reward_vector": rewards,
                "parsed_answers": [record["parsed_answer"] for record in completion_records],
                "failure_reasons": [record["failure_reason"] for record in completion_records],
                "parse_failure_vector": parse_failures,
                "completion_lengths": completion_lengths,
                "completion_records": completion_records,
            }
        )
    _write_jsonl(path, rows)
    return rows


def _generate_text(model, tokenizer, prompt, config, sample=False):
    try:
        import torch
    except ImportError:
        torch = None

    prompt_text = _generation_prompt_to_text(prompt, tokenizer, enable_thinking=config.get("enable_thinking"))
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = _model_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": int(config["rollout"]["max_completion_length"]),
        "do_sample": bool(sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generation_kwargs["temperature"] = float(config["rollout"]["temperature"])
        generation_kwargs["top_p"] = float(config["rollout"]["top_p"])
        top_k = int(config["rollout"]["top_k"])
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
    if torch is None:
        outputs = model.generate(**inputs, **generation_kwargs)
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _summarize_samples(sample_rows):
    if not sample_rows:
        return {
            "reward_mean": None,
            "reward_std": None,
            "zero_reward_rate": None,
            "perfect_reward_rate": None,
            "parse_failure_rate": None,
            "avg_completion_length": None,
            "frac_reward_zero_std": None,
            "effective_mixed_group_rate": None,
        }
    rewards = _flatten_values(sample_rows, "reward_vector", "reward")
    parse_failures = _flatten_values(sample_rows, "parse_failure_vector", "parse_failure")
    lengths = _flatten_values(sample_rows, "completion_lengths", "completion_length")
    group_reward_stds = [
        statistics.pstdev(row["reward_vector"]) if "reward_vector" in row else 0.0
        for row in sample_rows
    ]
    return {
        "reward_mean": sum(rewards) / len(rewards),
        "reward_std": statistics.pstdev(rewards),
        "zero_reward_rate": sum(1 for reward in rewards if reward == 0.0) / len(rewards),
        "perfect_reward_rate": sum(1 for reward in rewards if reward == 1.0) / len(rewards),
        "parse_failure_rate": sum(1 for value in parse_failures if value) / len(parse_failures),
        "avg_completion_length": sum(lengths) / len(lengths),
        "frac_reward_zero_std": sum(1 for value in group_reward_stds if value == 0.0) / len(group_reward_stds),
        "effective_mixed_group_rate": sum(1 for value in group_reward_stds if value > 0.0) / len(group_reward_stds),
    }


def _summarize_trainer_signal(trainer_log):
    steps = [row for row in trainer_log if isinstance(row, dict) and "frac_reward_zero_std" in row]
    if not steps:
        return {}

    frac_zero_values = [float(row["frac_reward_zero_std"]) for row in steps]
    reward_values = [float(row["reward"]) for row in steps if "reward" in row]
    reward_std_values = [float(row["reward_std"]) for row in steps if "reward_std" in row]
    grad_norm_values = [float(row["grad_norm"]) for row in steps if "grad_norm" in row]
    completion_lengths = [
        float(row["completions/mean_length"])
        for row in steps
        if "completions/mean_length" in row
    ]
    signal = {
        "frac_reward_zero_std": sum(frac_zero_values) / len(frac_zero_values),
        "effective_mixed_group_rate": sum(1 for value in frac_zero_values if value < 1.0) / len(frac_zero_values),
        "trainer_logged_steps": len(steps),
        "nonzero_grad_step_rate": (
            sum(1 for value in grad_norm_values if value > 0.0) / len(grad_norm_values)
            if grad_norm_values
            else None
        ),
    }
    if reward_values:
        signal["reward_mean"] = sum(reward_values) / len(reward_values)
    if reward_std_values:
        signal["reward_std"] = sum(reward_std_values) / len(reward_std_values)
    if completion_lengths:
        signal["avg_completion_length"] = sum(completion_lengths) / len(completion_lengths)
    return signal


def _flatten_values(rows, vector_key, scalar_key):
    values = []
    for row in rows:
        if vector_key in row:
            values.extend(row[vector_key])
        else:
            values.append(row[scalar_key])
    return values


def _write_synthetic_rlvr_jsonl(path, train_count, problem_style="addition"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(train_count):
        prompt, answer, difficulty = _synthetic_rlvr_problem(index, problem_style)
        rows.append(
            {
                "id": f"rlvr-train-{index:04d}",
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": answer},
                "metadata": {
                    "source": f"synthetic-rlvr-{problem_style}",
                    "domain": "math",
                    "difficulty": difficulty,
                    "license": "synthetic",
                },
            }
        )
    _write_jsonl(path, rows)


def _synthetic_rlvr_problem(index, problem_style):
    if problem_style == "addition":
        left = 2 + (index % 19)
        right = 3 + ((index * 5) % 23)
        return (
            f"Compute {left} + {right}. Return only the final answer in boxed format.",
            str(left + right),
            "easy",
        )
    if problem_style == "mixed_arithmetic":
        variant = index % 4
        a = 2 + (index % 13)
        b = 3 + ((index * 5) % 17)
        c = 2 + ((index * 7) % 9)
        if variant == 0:
            return (
                f"Compute {a} + {b} + {c}. Return only the final answer in boxed format.",
                str(a + b + c),
                "medium",
            )
        if variant == 1:
            minuend = a + b + c
            subtrahend = b
            return (
                f"Compute {minuend} - {subtrahend}. Return only the final answer in boxed format.",
                str(minuend - subtrahend),
                "medium",
            )
        if variant == 2:
            return (
                f"Compute {a} * {c}. Return only the final answer in boxed format.",
                str(a * c),
                "medium",
            )
        return (
            f"Compute {a} + {b} * {c}. Return only the final answer in boxed format.",
            str(a + b * c),
            "medium",
        )
    raise ValueError(f"unsupported synthetic_problem_style: {problem_style}")


def _prompt_text(prompt):
    if isinstance(prompt, list):
        return "\n".join(message["content"] for message in prompt if message.get("role") == "user")
    return str(prompt)


def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _write_run_card(path, resolved, data_hash, config_hash, git_commit, final_loss, metrics, eval_outputs):
    finality = "smoke, not final" if resolved["smoke_run"] else "small RLVR run, not final"
    lines = [
        "# GRPO Run Card",
        "",
        f"- run name: `{resolved['run_name']}`",
        f"- smoke run: `{resolved['smoke_run']}`",
        f"- finality: `{finality}`",
        f"- base model: `{resolved['model_name_or_path']}`",
        f"- adapter path: `{resolved['adapter_path']}`",
        f"- data path: `{resolved['data_path']}`",
        f"- data hash: `{data_hash}`",
        f"- config hash: `{config_hash}`",
        f"- git commit: `{git_commit}`",
        f"- output path: `{resolved['output_dir']}`",
        f"- policy checkpoint path: `{resolved['output_dir']}`",
        f"- reward version: `{resolved['reward_version']}`",
        f"- reward symbolic equivalence: `{resolved['reward']['allow_symbolic_equivalence']}`",
        f"- reward symbolic engine: `{resolved['reward']['symbolic_equivalence_engine']}`",
        f"- final loss: `{final_loss}`",
        f"- peft method: `{resolved['peft']['method']}`",
        f"- learning rate: `{resolved['training']['learning_rate']}`",
        f"- per-device train batch size: `{resolved['training']['per_device_train_batch_size']}`",
        f"- gradient accumulation steps: `{resolved['training']['gradient_accumulation_steps']}`",
        f"- max grad norm: `{resolved['training'].get('max_grad_norm')}`",
        f"- optimizer: `{resolved['training'].get('optim')}`",
        f"- lr scheduler: `{resolved['training'].get('lr_scheduler_type')}`",
        f"- warmup ratio: `{resolved['training'].get('warmup_ratio')}`",
        f"- reward mean: `{metrics['reward_mean']}`",
        f"- reward std: `{metrics['reward_std']}`",
        f"- frac reward zero std: `{metrics.get('frac_reward_zero_std')}`",
        f"- effective mixed group rate: `{metrics.get('effective_mixed_group_rate')}`",
        f"- nonzero grad step rate: `{metrics.get('nonzero_grad_step_rate')}`",
        f"- zero reward rate: `{metrics['zero_reward_rate']}`",
        f"- perfect reward rate: `{metrics['perfect_reward_rate']}`",
        f"- parse failure rate: `{metrics['parse_failure_rate']}`",
        f"- avg completion length: `{metrics['avg_completion_length']}`",
        f"- rollout format gate enabled: `{resolved['rollout_format_gate']['enabled']}`",
        f"- rollout format gate parse failure rate: `{metrics.get('rollout_format_gate_parse_failure_rate')}`",
        f"- rollout format gate reward mean: `{metrics.get('rollout_format_gate_reward_mean')}`",
        f"- rollout format gate reward std: `{metrics.get('rollout_format_gate_reward_std')}`",
        f"- rollout format gate perfect reward rate: `{metrics.get('rollout_format_gate_perfect_reward_rate')}`",
        f"- rollout format gate max reward mean: `{resolved['rollout_format_gate'].get('max_reward_mean')}`",
        f"- rollout format gate max perfect reward rate: `{resolved['rollout_format_gate'].get('max_perfect_reward_rate')}`",
        f"- dry run: `{resolved['dry_run']}`",
        f"- train examples: `{resolved['selection']['max_train_examples']}`",
        f"- max steps: `{resolved['training']['max_steps']}`",
        f"- num generations: `{resolved['rollout']['num_generations']}`",
        f"- sample rollout generations: `{resolved['rollout'].get('sample_rollout_generations')}`",
        f"- max prompt length: `{resolved['rollout'].get('max_prompt_length')}`",
        f"- max completion length: `{resolved['rollout']['max_completion_length']}`",
        f"- temperature: `{resolved['rollout']['temperature']}`",
        f"- top_p: `{resolved['rollout']['top_p']}`",
        f"- beta: `{resolved['rollout']['beta']}`",
        f"- epsilon: `{resolved['rollout'].get('epsilon')}`",
        f"- epsilon high: `{resolved['rollout'].get('epsilon_high')}`",
        f"- num iterations: `{resolved['rollout'].get('num_iterations')}`",
        f"- loss type: `{resolved['rollout'].get('loss_type')}`",
        f"- scale rewards: `{resolved['rollout'].get('scale_rewards')}`",
        f"- steps per generation: `{resolved['rollout'].get('steps_per_generation')}`",
        f"- generation batch size: `{resolved['rollout'].get('generation_batch_size')}`",
        f"- use vllm: `{resolved['rollout'].get('use_vllm')}`",
        f"- frac reward zero std early stop: `{resolved['training'].get('frac_reward_zero_std_early_stop')}`",
        f"- frac reward zero std threshold: `{resolved['training'].get('frac_reward_zero_std_threshold')}`",
        f"- frac reward zero std patience: `{resolved['training'].get('frac_reward_zero_std_patience')}`",
        f"- eval after train enabled: `{resolved['eval_after_train']['enabled']}`",
    ]
    if eval_outputs:
        lines.extend(
            [
                f"- base eval path: `{eval_outputs['base']['output_dir']}`",
                f"- SFT eval path: `{eval_outputs['sft']['output_dir']}`",
                f"- SFT+RLVR eval path: `{eval_outputs['sft_rlvr']['output_dir']}`",
            ]
        )
    _write_text(path, "\n".join(lines) + "\n")


def _math_reward_config(config):
    reward = config.get("reward", {})
    return MathRewardConfig(
        allow_symbolic_equivalence=bool(reward.get("allow_symbolic_equivalence", False)),
        symbolic_equivalence_engine=str(reward.get("symbolic_equivalence_engine", "fraction")),
        max_symbolic_expr_chars=int(reward.get("max_symbolic_expr_chars", 120)),
        max_symbolic_ast_nodes=int(reward.get("max_symbolic_ast_nodes", 64)),
        max_symbolic_collection_size=int(reward.get("max_symbolic_collection_size", 32)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
