"""Minimal GRPO smoke training using TRL and math_boxed_v001 rewards."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl
from posttrain_lab.rewards.math_reward import MathRewardConfig, math_boxed_v001, score_math_boxed_v001
from posttrain_lab.train.train_sft import (
    _dump_yaml,
    _generation_prompt_to_text,
    _git_commit,
    _load_yaml_subset,
    _model_load_kwargs,
    _sha256_file,
    _write_jsonl,
    _write_text,
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


def run_grpo(config, config_path):
    """Run a dry-run or real TRL GRPO smoke experiment."""

    resolved = _resolve_config(config)
    output_dir = Path(resolved["output_dir"])
    data_path = Path(resolved["data_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolved.get("synthetic_data_if_missing") and not data_path.exists():
        _write_synthetic_rlvr_jsonl(data_path, train_count=resolved["selection"]["max_train_examples"])

    examples = load_rlvr_train_examples(
        data_path,
        split=resolved["selection"]["train_split"],
        limit=resolved["selection"]["max_train_examples"],
    )
    data_hash = _sha256_file(data_path)
    config_hash = _sha256_file(config_path)
    git_commit = _git_commit()

    _write_text(output_dir / "resolved_config.yaml", _dump_yaml(resolved))

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
        _write_jsonl(output_dir / "trainer_log.jsonl", trainer_log)
        sample_rows = _write_sample_rollouts(
            output_dir / "sample_rollouts.jsonl",
            resolved,
            examples=examples,
            dry_run=False,
            model=model,
            tokenizer=tokenizer,
        )

    metrics = _summarize_samples(sample_rows)
    metrics.update(
        {
            "step": int(resolved["training"]["max_steps"]),
            "final_loss": final_loss,
            "reward_version": resolved["reward_version"],
            "dry_run": resolved["dry_run"],
            "smoke_run": resolved["smoke_run"],
        }
    )
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
    )

    result = dict(metrics)
    result.update(
        {
            "output_dir": str(output_dir),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "git_commit": git_commit,
        }
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a minimal GRPO smoke experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_grpo(config, config_path=args.config)
    print(json.dumps(result, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = copy.deepcopy(config)
    resolved.setdefault("run_name", "grpo-smoke")
    resolved.setdefault("smoke_run", True)
    resolved.setdefault("synthetic_data_if_missing", False)
    resolved.setdefault("torch_dtype", "auto")
    resolved.setdefault("trust_remote_code", False)
    resolved.setdefault("enable_thinking", None)
    resolved["selection"].setdefault("train_split", "train")
    resolved["selection"].setdefault("max_train_examples", 4)
    resolved["training"].setdefault("max_steps", 1)
    resolved["training"].setdefault("per_device_train_batch_size", 2)
    resolved["training"].setdefault("gradient_accumulation_steps", 1)
    resolved["training"].setdefault("learning_rate", 1e-6)
    resolved["training"].setdefault("bf16", False)
    resolved["training"].setdefault("fp16", False)
    resolved["training"].setdefault("gradient_checkpointing", False)
    resolved["training"].setdefault("logging_steps", 1)
    resolved["training"].setdefault("save_steps", resolved["training"]["max_steps"])
    resolved["training"].setdefault("save_total_limit", 1)
    resolved["rollout"].setdefault("num_generations", 2)
    resolved["rollout"].setdefault("max_completion_length", 16)
    resolved["rollout"].setdefault("temperature", 0.7)
    resolved["rollout"].setdefault("top_p", 1.0)
    resolved["rollout"].setdefault("top_k", 0)
    resolved["rollout"].setdefault("beta", 0.0)
    resolved["rollout"].setdefault("sample_count", 4)
    resolved.setdefault("peft", {})
    resolved["peft"].setdefault("method", "lora")
    resolved["peft"].setdefault("r", 4)
    resolved["peft"].setdefault("lora_alpha", 8)
    resolved["peft"].setdefault("lora_dropout", 0.0)
    resolved["peft"].setdefault("target_modules", [])

    if resolved["reward_version"] != "math_boxed_v001":
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
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("TRL GRPO training requires transformers, datasets, trl, and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = config["peft"].get("target_modules") or None
    peft_config = LoraConfig(
        r=int(config["peft"]["r"]),
        lora_alpha=int(config["peft"]["lora_alpha"]),
        lora_dropout=float(config["peft"]["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    train_dataset = Dataset.from_list(_to_grpo_rows(examples))
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=int(config["training"]["max_steps"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=int(config["training"]["logging_steps"]),
        logging_first_step=True,
        save_steps=int(config["training"]["save_steps"]),
        save_total_limit=int(config["training"]["save_total_limit"]),
        save_strategy="steps",
        report_to="none",
        seed=int(config["seed"]),
        bf16=bool(config["training"]["bf16"]),
        fp16=bool(config["training"]["fp16"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        num_generations=int(config["rollout"]["num_generations"]),
        max_completion_length=int(config["rollout"]["max_completion_length"]),
        temperature=float(config["rollout"]["temperature"]),
        top_p=float(config["rollout"]["top_p"]),
        top_k=int(config["rollout"]["top_k"]),
        beta=float(config["rollout"]["beta"]),
        log_completions=True,
        num_completions_to_print=int(config["rollout"]["num_generations"]),
        model_init_kwargs=_model_load_kwargs(config),
        chat_template_kwargs=_chat_template_kwargs(config),
    )
    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        reward_funcs=math_boxed_reward_func,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    return float(result.training_loss), trainer.state.log_history, trainer.model, tokenizer


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
    sample_count = min(int(config["rollout"]["sample_count"]), len(examples))
    selected = examples[:sample_count]
    if not dry_run and hasattr(model, "eval"):
        model.eval()

    for index, example in enumerate(selected):
        prompt = _prompt_text(example["prompt"])
        answer = example["verifier"]["answer"]
        if dry_run:
            generation = rf"\boxed{{{answer}}}"
        else:
            generation = _generate_text(model, tokenizer, prompt, config)
        reward_result = score_math_boxed_v001(generation, answer, config=MathRewardConfig())
        rows.append(
            {
                "id": example["id"],
                "sample_index": index,
                "prompt": prompt,
                "answer": answer,
                "completion": generation,
                "reward": reward_result.score,
                "parsed_answer": reward_result.normalized_prediction,
                "failure_reason": None if reward_result.score == 1.0 else reward_result.reason,
                "parse_failure": reward_result.reason in PARSE_FAILURE_REASONS,
                "completion_length": len(generation),
            }
        )
    _write_jsonl(path, rows)
    return rows


def _generate_text(model, tokenizer, prompt, config):
    try:
        import torch
    except ImportError:
        torch = None

    prompt_text = _generation_prompt_to_text(prompt, tokenizer, enable_thinking=config.get("enable_thinking"))
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": int(config["rollout"]["max_completion_length"]),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if torch is None:
        outputs = model.generate(**inputs, **generation_kwargs)
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _summarize_samples(sample_rows):
    if not sample_rows:
        return {
            "reward_mean": None,
            "reward_std": None,
            "zero_reward_rate": None,
            "perfect_reward_rate": None,
            "parse_failure_rate": None,
            "avg_completion_length": None,
        }
    rewards = [row["reward"] for row in sample_rows]
    return {
        "reward_mean": sum(rewards) / len(rewards),
        "reward_std": statistics.pstdev(rewards),
        "zero_reward_rate": sum(1 for reward in rewards if reward == 0.0) / len(rewards),
        "perfect_reward_rate": sum(1 for reward in rewards if reward == 1.0) / len(rewards),
        "parse_failure_rate": sum(1 for row in sample_rows if row["parse_failure"]) / len(sample_rows),
        "avg_completion_length": sum(row["completion_length"] for row in sample_rows) / len(sample_rows),
    }


def _write_synthetic_rlvr_jsonl(path, train_count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(train_count):
        left = 2 + (index % 19)
        right = 3 + ((index * 5) % 23)
        rows.append(
            {
                "id": f"rlvr-train-{index:04d}",
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Compute {left} + {right}. Return only the final answer in boxed format.",
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": str(left + right)},
                "metadata": {
                    "source": "synthetic-rlvr-smoke",
                    "domain": "math",
                    "difficulty": "easy",
                    "license": "synthetic",
                },
            }
        )
    _write_jsonl(path, rows)


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


def _write_run_card(path, resolved, data_hash, config_hash, git_commit, final_loss, metrics):
    lines = [
        "# GRPO Run Card",
        "",
        f"- run name: `{resolved['run_name']}`",
        f"- smoke run: `{resolved['smoke_run']}`",
        "- finality: `smoke, not final`",
        f"- base model: `{resolved['model_name_or_path']}`",
        f"- data path: `{resolved['data_path']}`",
        f"- data hash: `{data_hash}`",
        f"- config hash: `{config_hash}`",
        f"- git commit: `{git_commit}`",
        f"- output path: `{resolved['output_dir']}`",
        f"- policy checkpoint path: `{resolved['output_dir']}`",
        f"- reward version: `{resolved['reward_version']}`",
        f"- final loss: `{final_loss}`",
        f"- reward mean: `{metrics['reward_mean']}`",
        f"- reward std: `{metrics['reward_std']}`",
        f"- zero reward rate: `{metrics['zero_reward_rate']}`",
        f"- perfect reward rate: `{metrics['perfect_reward_rate']}`",
        f"- parse failure rate: `{metrics['parse_failure_rate']}`",
        f"- avg completion length: `{metrics['avg_completion_length']}`",
        f"- dry run: `{resolved['dry_run']}`",
        f"- train examples: `{resolved['selection']['max_train_examples']}`",
        f"- max steps: `{resolved['training']['max_steps']}`",
        f"- num generations: `{resolved['rollout']['num_generations']}`",
        f"- max completion length: `{resolved['rollout']['max_completion_length']}`",
        f"- temperature: `{resolved['rollout']['temperature']}`",
        f"- top_p: `{resolved['rollout']['top_p']}`",
        f"- beta: `{resolved['rollout']['beta']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
