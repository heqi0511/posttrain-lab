"""Small SFT pipelines using TRL SFTTrainer and PEFT adapters."""

import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl
from posttrain_lab.eval.eval_runner import run_eval
from posttrain_lab.rewards.math_reward import extract_boxed_answers


REQUIRED_TOP_LEVEL = {
    "model_name_or_path",
    "data_path",
    "output_dir",
    "dry_run",
    "seed",
    "selection",
    "training",
    "peft",
    "generation_check",
}


def load_config(path):
    """Load the small YAML subset used by configs/sft/overfit32.yaml."""

    config = _load_yaml_subset(Path(path))
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"missing required config fields: {', '.join(missing)}")
    return config


def load_sft_train_examples(path, split="train", limit=32):
    """Validate and load exactly ``limit`` SFT examples from the requested split."""

    return load_sft_examples(path, split=split, limit=limit)


def load_sft_examples(path, split, limit):
    """Validate and load exactly ``limit`` SFT examples from the requested split."""

    errors = validate_jsonl("sft", path)
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


def run_sft(config, config_path):
    """Run dry-run artifact generation or a real TRL SFT overfit-32 run."""

    resolved = _resolve_config(config)
    data_path = Path(resolved["data_path"])
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolved.get("dataset", {}).get("id") and not data_path.exists():
        _write_hf_sft_jsonl(data_path, resolved)

    if resolved.get("synthetic_data_if_missing") and not data_path.exists():
        _write_synthetic_sft_jsonl(
            data_path,
            train_count=resolved["selection"]["max_train_examples"],
            validation_count=resolved["selection"]["max_validation_examples"],
            answer_format=resolved["synthetic_answer_format"],
            problem_style=resolved["synthetic_problem_style"],
        )

    examples = load_sft_train_examples(
        data_path,
        split=resolved["selection"]["train_split"],
        limit=resolved["selection"]["max_train_examples"],
    )
    validation_examples = load_sft_examples(
        data_path,
        split=resolved["selection"]["validation_split"],
        limit=resolved["selection"]["max_validation_examples"],
    )

    data_hash = _sha256_file(data_path)
    config_hash = _sha256_file(config_path)
    git_commit = _git_commit()

    _write_text(output_dir / "resolved_config.yaml", _dump_yaml(resolved))

    if resolved["dry_run"]:
        train_loss = 0.0
        validation_loss = 0.0
        sample_rows = _write_generation_check(
            output_dir / "sample_generations.jsonl",
            resolved,
            dry_run=True,
            examples=examples,
        )
    else:
        train_loss, validation_loss, model, tokenizer = _run_trl_training(
            resolved,
            train_examples=examples,
            validation_examples=validation_examples,
            output_dir=output_dir,
        )
        sample_rows = _write_generation_check(
            output_dir / "sample_generations.jsonl",
            resolved,
            dry_run=False,
            model=model,
            tokenizer=tokenizer,
            examples=examples,
        )

    eval_metrics = _run_eval_after_train(resolved, output_dir, validation_examples=validation_examples)
    average_output_length = _average_generation_length(sample_rows, eval_metrics)
    format_success = eval_metrics.get("format_success")
    target_eval_score = eval_metrics.get("answer_match")
    if target_eval_score is None:
        target_eval_score = eval_metrics.get("exact_match")
    final_loss = train_loss
    _write_metrics(
        output_dir / "metrics.jsonl",
        [
            {
                "step": int(resolved["training"]["max_steps"]),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "final_loss": final_loss,
                "format_success": format_success,
                "target_eval_score": target_eval_score,
                "average_output_length": average_output_length,
                "dry_run": resolved["dry_run"],
                "smoke_run": resolved["smoke_run"],
            }
        ],
    )
    _write_run_card(
        output_dir / "run_card.md",
        resolved=resolved,
        data_hash=data_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        final_loss=final_loss,
        metrics={
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "format_success": format_success,
            "target_eval_score": target_eval_score,
            "average_output_length": average_output_length,
        },
    )
    return {
        "output_dir": str(output_dir),
        "final_loss": final_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "format_success": format_success,
        "target_eval_score": target_eval_score,
        "average_output_length": average_output_length,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a small SFT experiment with TRL SFTTrainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Write a dry-run plan and do not train.")
    parser.add_argument("--output-dir", help="Override output directory for --dry-run plan artifacts.")
    args = parser.parse_args(argv)

    if args.dry_run:
        result = write_cli_dry_run_plan(args.config, args.output_dir, mode="sft")
        print(json.dumps(result, sort_keys=True))
        return 0

    config = load_config(args.config)
    result = run_sft(config, config_path=args.config)
    print(json.dumps(result, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = copy.deepcopy(config)
    resolved.setdefault("run_name", "sft-overfit32")
    resolved.setdefault("smoke_run", False)
    if "split" in resolved["selection"] and "train_split" not in resolved["selection"]:
        resolved["selection"]["train_split"] = resolved["selection"]["split"]
    resolved["selection"].setdefault("train_split", "train")
    resolved["selection"].setdefault("validation_split", "val")
    resolved["selection"].setdefault("max_train_examples", 32)
    resolved["selection"].setdefault("max_validation_examples", 0)
    resolved["training"].setdefault("max_steps", 2)
    resolved["training"].setdefault("per_device_train_batch_size", 1)
    resolved["training"].setdefault("per_device_eval_batch_size", 1)
    resolved["training"].setdefault("gradient_accumulation_steps", 1)
    resolved["training"].setdefault("learning_rate", 2e-4)
    resolved["training"].setdefault("max_seq_length", 512)
    resolved["training"].setdefault("bf16", False)
    resolved["training"].setdefault("fp16", False)
    resolved["training"].setdefault("gradient_checkpointing", False)
    resolved["training"].setdefault("warmup_steps", 0)
    resolved["training"].setdefault("logging_steps", 1)
    resolved["training"].setdefault("eval_steps", 1)
    resolved["training"].setdefault("save_steps", resolved["training"]["max_steps"])
    resolved["training"].setdefault("save_total_limit", 2)
    resolved["peft"].setdefault("method", "lora")
    resolved["peft"].setdefault("qlora", False)
    resolved["peft"].setdefault("r", 8)
    resolved["peft"].setdefault("lora_alpha", 16)
    resolved["peft"].setdefault("lora_dropout", 0.0)
    resolved["peft"].setdefault("target_modules", [])
    resolved.setdefault("torch_dtype", "auto")
    resolved.setdefault("trust_remote_code", False)
    resolved.setdefault("attn_implementation", None)
    resolved["generation_check"].setdefault("max_new_tokens", 32)
    resolved["generation_check"].setdefault("enable_thinking", None)
    resolved["generation_check"].setdefault("prompts", [])
    resolved["generation_check"].setdefault("random_sample_count", 0)
    resolved.setdefault("eval_after_train", {})
    resolved["eval_after_train"].setdefault("enabled", False)
    resolved["eval_after_train"].setdefault("prompt_path", "/tmp/posttrain_lab_eval/baseline_prompts.jsonl")
    resolved["eval_after_train"].setdefault("output_dir", str(Path(resolved["output_dir"]) / "eval"))
    resolved["eval_after_train"].setdefault("baseline_metrics_path", "/tmp/posttrain_lab_eval/baseline_run/metrics.json")
    resolved["eval_after_train"].setdefault("dry_run", True)
    resolved["eval_after_train"].setdefault("model_name", "dummy")
    resolved["eval_after_train"].setdefault("adapter_path", str(Path(resolved["output_dir"])))
    resolved["eval_after_train"].setdefault("torch_dtype", resolved["torch_dtype"])
    resolved["eval_after_train"].setdefault("trust_remote_code", resolved["trust_remote_code"])
    resolved["eval_after_train"].setdefault("apply_chat_template", True)
    resolved["eval_after_train"].setdefault("enable_thinking", None)
    resolved["eval_after_train"].setdefault("temperature", 0.0)
    resolved["eval_after_train"].setdefault("top_p", 1.0)
    resolved["eval_after_train"].setdefault("max_new_tokens", 32)
    resolved["eval_after_train"].setdefault("format_regex", r"^\\boxed\{.+\}$")
    resolved["eval_after_train"].setdefault("exact_match", True)
    resolved["eval_after_train"].setdefault("boxed_math_match", False)
    resolved["eval_after_train"].setdefault("allow_symbolic_equivalence", False)
    resolved["eval_after_train"].setdefault("prompt_source", "file")
    resolved["eval_after_train"].setdefault("sample_size", 0)
    resolved.setdefault("synthetic_data_if_missing", False)
    resolved.setdefault("synthetic_answer_format", "plain")
    resolved.setdefault("synthetic_problem_style", "increment")
    resolved.setdefault("dataset", {})
    resolved["dataset"].setdefault("id", "")
    resolved["dataset"].setdefault("config", None)
    resolved["dataset"].setdefault("split", "train")
    resolved["dataset"].setdefault("messages_field", "messages")
    resolved["dataset"].setdefault("source_field", "source")
    resolved["dataset"].setdefault("domain", resolved["dataset"].get("config") or "unknown")
    resolved["dataset"].setdefault("difficulty", "unknown")
    resolved["dataset"].setdefault("license", "source-dataset-card")
    resolved["dataset"].setdefault("shuffle", True)
    resolved["dataset"].setdefault("streaming", False)
    resolved["dataset"].setdefault("shuffle_buffer_size", 10000)
    if resolved["run_name"] == "sft-overfit32" and resolved["selection"]["max_train_examples"] != 32:
        raise ValueError("overfit-32 requires selection.max_train_examples == 32")
    return resolved


def write_cli_dry_run_plan(config_path, output_dir=None, mode="training"):
    """Write a small dry-run plan without loading a model or starting training."""

    config = _load_yaml_subset(Path(config_path))
    resolved_output_dir = Path(output_dir or config.get("output_dir", "runs/dry_run"))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "mode": "dry_run",
        "task": mode,
        "config_path": str(config_path),
        "output_dir": str(resolved_output_dir),
        "will_train": False,
        "will_load_model": False,
    }
    _write_text(resolved_output_dir / "dry_run_plan.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _run_trl_training(config, train_examples, validation_examples, output_dir):
    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("TRL SFT training requires transformers, datasets, trl, and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = _model_load_kwargs(config)
    if config["peft"]["qlora"]:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA requires BitsAndBytesConfig from transformers") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_bnb_compute_dtype(config["torch_dtype"]),
        )

    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
    if bool(config["training"]["gradient_checkpointing"]) and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if config["peft"]["qlora"]:
        model = prepare_model_for_kbit_training(model)

    target_modules = config["peft"].get("target_modules") or None
    peft_config = LoraConfig(
        r=int(config["peft"]["r"]),
        lora_alpha=int(config["peft"]["lora_alpha"]),
        lora_dropout=float(config["peft"]["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    train_dataset = Dataset.from_list(
        [{"text": _messages_to_text(example["messages"], tokenizer)} for example in train_examples]
    )
    eval_dataset = Dataset.from_list(
        [{"text": _messages_to_text(example["messages"], tokenizer)} for example in validation_examples]
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=int(config["training"]["max_steps"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        logging_steps=int(config["training"]["logging_steps"]),
        logging_first_step=True,
        eval_strategy="steps" if validation_examples else "no",
        eval_steps=int(config["training"]["eval_steps"]) if validation_examples else None,
        save_steps=int(config["training"]["save_steps"]),
        save_total_limit=int(config["training"]["save_total_limit"]),
        report_to="none",
        seed=int(config["seed"]),
        bf16=bool(config["training"]["bf16"]),
        fp16=bool(config["training"]["fp16"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        dataset_text_field="text",
        max_length=int(config["training"]["max_seq_length"]),
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if validation_examples else None,
        args=args,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))

    final_loss = float(result.training_loss)
    validation_loss = _last_metric(trainer.state.log_history, "eval_loss")
    _write_metrics(output_dir / "trainer_log.jsonl", trainer.state.log_history)
    _write_loss_curve(output_dir / "loss_curve.csv", trainer.state.log_history)
    _write_checkpoint_manifest(output_dir / "checkpoint_manifest.json", output_dir)
    return final_loss, validation_loss, trainer.model, tokenizer


def _messages_to_text(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _write_generation_check(path, config, dry_run, model=None, tokenizer=None, examples=None):
    rows = []
    prompts = list(config["generation_check"].get("prompts", []))
    sample_count = int(config["generation_check"].get("random_sample_count", 0))
    if sample_count and examples:
        rng = random.Random(int(config["seed"]))
        selected = rng.sample(examples, k=min(sample_count, len(examples)))
        prompts.extend(_example_prompt(example) for example in selected)

    if not dry_run and hasattr(model, "eval"):
        model.eval()

    cache_state = _enable_generation_cache(model) if not dry_run else None
    try:
        for index, prompt in enumerate(prompts):
            if dry_run:
                generation = _dry_generation_for_prompt(prompt)
            else:
                generation = _generate_text(
                    model,
                    tokenizer,
                    prompt,
                    config["generation_check"]["max_new_tokens"],
                    enable_thinking=config["generation_check"]["enable_thinking"],
                )
            rows.append(
                {
                    "id": f"generation-check-{index}",
                    "prompt": prompt,
                    "generation": generation,
                    "dry_run": dry_run,
                }
            )
            _write_jsonl(path, rows)
    finally:
        _restore_generation_cache(model, cache_state)

    if not prompts:
        _write_jsonl(path, rows)
    return rows


def _enable_generation_cache(model):
    """Temporarily undo gradient-checkpointing cache disablement for generation."""

    state = {}
    for attr_name in ("config", "generation_config"):
        config_obj = getattr(model, attr_name, None)
        if config_obj is not None and hasattr(config_obj, "use_cache"):
            state[attr_name] = getattr(config_obj, "use_cache")
            setattr(config_obj, "use_cache", True)
    return state


def _restore_generation_cache(model, state):
    if not state:
        return
    for attr_name, value in state.items():
        config_obj = getattr(model, attr_name, None)
        if config_obj is not None and hasattr(config_obj, "use_cache"):
            setattr(config_obj, "use_cache", value)


def _write_synthetic_sft_jsonl(path, train_count, validation_count, answer_format="plain", problem_style="increment"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(train_count):
        prompt, answer = _synthetic_math_example(index, answer_format, problem_style)
        rows.append(
            {
                "id": f"sft-train-{index:04d}",
                "split": "train",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": "synthetic-dry-run",
                    "domain": "math",
                    "difficulty": "easy",
                    "license": "synthetic",
                },
            }
        )
    for index in range(validation_count):
        prompt, answer = _synthetic_math_example(train_count + index, answer_format, problem_style)
        rows.append(
            {
                "id": f"sft-smoke-val-{index:03d}",
                "split": "val",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": "synthetic-dry-run",
                    "domain": "math",
                    "difficulty": "easy",
                    "license": "synthetic",
                },
            }
        )
    _write_jsonl(path, rows)


def _write_hf_sft_jsonl(path, config):
    dataset_config = config["dataset"]
    dataset_id = dataset_config["id"]
    if not dataset_id:
        raise ValueError("dataset.id is required when dataset is configured")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to materialize Hugging Face SFT data") from exc

    total_needed = int(config["selection"]["max_train_examples"]) + int(config["selection"]["max_validation_examples"])
    if total_needed <= 0:
        raise ValueError("at least one train or validation example is required")

    load_kwargs = {}
    if dataset_config["config"]:
        load_kwargs["name"] = dataset_config["config"]
    dataset = load_dataset(
        dataset_id,
        split=dataset_config["split"],
        streaming=bool(dataset_config["streaming"]),
        **load_kwargs,
    )
    if dataset_config["shuffle"]:
        if dataset_config["streaming"]:
            dataset = dataset.shuffle(
                seed=int(config["seed"]),
                buffer_size=int(dataset_config["shuffle_buffer_size"]),
            )
        else:
            dataset = dataset.shuffle(seed=int(config["seed"]))

    rows = []
    skipped = 0
    for source_index, record in enumerate(dataset):
        split = "train" if len(rows) < int(config["selection"]["max_train_examples"]) else "val"
        converted = _convert_hf_sft_record(record, dataset_config, split, len(rows), source_index)
        if converted is None:
            skipped += 1
            continue
        rows.append(converted)
        if len(rows) == total_needed:
            break

    if len(rows) != total_needed:
        raise ValueError(f"expected {total_needed} usable SFT records from {dataset_id}, found {len(rows)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, rows)
    _write_text(
        path.with_suffix(".manifest.json"),
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_config": dataset_config["config"],
                "source_split": dataset_config["split"],
                "seed": int(config["seed"]),
                "shuffle": bool(dataset_config["shuffle"]),
                "streaming": bool(dataset_config["streaming"]),
                "shuffle_buffer_size": int(dataset_config["shuffle_buffer_size"]),
                "train_examples": int(config["selection"]["max_train_examples"]),
                "validation_examples": int(config["selection"]["max_validation_examples"]),
                "skipped_records": skipped,
                "output_path": str(path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _convert_hf_sft_record(record, dataset_config, split, row_index, source_index):
    messages = _normalize_hf_messages(record.get(dataset_config["messages_field"]))
    if not messages or not any(message["role"] == "assistant" for message in messages):
        return None

    source = record.get(dataset_config["source_field"]) or dataset_config["id"]
    return {
        "id": f"openr1-{dataset_config['config'] or 'default'}-{split}-{row_index:06d}",
        "split": split,
        "messages": messages,
        "metadata": {
            "source": str(source),
            "domain": str(dataset_config["domain"]),
            "difficulty": str(dataset_config["difficulty"]),
            "license": str(dataset_config["license"]),
        },
    }


def _normalize_hf_messages(messages):
    if not isinstance(messages, list) or not messages:
        return []

    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return []
        role = message.get("role") or message.get("from")
        if role == "human":
            role = "user"
        elif role in {"gpt", "model"}:
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            if len(messages) == 2 and index == 0:
                role = "user"
            elif len(messages) == 2 and index == 1:
                role = "assistant"
            else:
                return []
        content = message.get("content") or message.get("value")
        if not isinstance(content, str) or not content.strip():
            return []
        normalized.append({"role": role, "content": content.strip()})
    return normalized


def _synthetic_math_example(index, answer_format, problem_style):
    if problem_style == "increment":
        left = index
        right = 1
        prompt = f"Compute {left} + {right}."
    elif problem_style == "addition":
        left = 3 + (index % 97)
        right = 5 + ((index * 7) % 89)
        prompt = f"Compute {left} + {right}. Return only the final answer in boxed format."
    else:
        raise ValueError(f"unsupported synthetic_problem_style: {problem_style}")

    answer = str(left + right)
    if answer_format == "plain":
        return prompt, answer
    if answer_format == "boxed":
        return prompt, rf"\boxed{{{answer}}}"
    raise ValueError(f"unsupported synthetic_answer_format: {answer_format}")


def _run_eval_after_train(config, output_dir, validation_examples=None):
    if not config["eval_after_train"]["enabled"]:
        _write_eval_diff(output_dir / "eval_diff.md", candidate=None, baseline=None)
        return {}

    eval_config = _build_eval_config(config, output_dir)
    if config["eval_after_train"].get("prompt_source") == "validation":
        eval_config["prompt_path"] = _write_validation_eval_prompts(
            output_dir / "eval_validation_prompts.jsonl",
            validation_examples or [],
            sample_size=int(config["eval_after_train"].get("sample_size") or 0),
        )
    prompt_path = Path(eval_config["prompt_path"])
    if eval_config["dry_run"] and not prompt_path.exists():
        _write_default_eval_prompts(prompt_path)

    candidate_metrics = run_eval(eval_config)
    baseline_metrics = _load_json_if_exists(config["eval_after_train"]["baseline_metrics_path"])
    _write_eval_diff(output_dir / "eval_diff.md", candidate=candidate_metrics, baseline=baseline_metrics)
    return candidate_metrics


def _build_eval_config(config, output_dir):
    eval_config = config["eval_after_train"]
    return {
        "prompt_path": eval_config["prompt_path"],
        "output_dir": eval_config.get("output_dir") or str(output_dir / "eval"),
        "dry_run": eval_config["dry_run"],
        "model_name": eval_config["model_name"],
        "adapter_path": eval_config["adapter_path"],
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
            "boxed_math_match": eval_config["boxed_math_match"],
            "allow_symbolic_equivalence": eval_config["allow_symbolic_equivalence"],
        },
    }


def _write_validation_eval_prompts(path, validation_examples, sample_size):
    rows = []
    limit = sample_size if sample_size > 0 else len(validation_examples)
    for example in validation_examples:
        prompt = _example_prompt(example)
        answer = _final_boxed_answer_from_example(example)
        if not prompt or not answer:
            continue
        rows.append({"id": example["id"], "prompt": prompt, "answer": answer})
        if len(rows) == limit:
            break

    if not rows:
        raise ValueError("validation eval requested but no boxed answers were found in validation examples")

    _write_jsonl(path, rows)
    return str(path)


def _final_boxed_answer_from_example(example):
    for message in reversed(example["messages"]):
        if message["role"] != "assistant":
            continue
        answers = extract_boxed_answers(message["content"])
        if answers:
            return answers[-1]
    return None


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


def _write_eval_diff(path, candidate, baseline):
    lines = ["# Eval Diff", "", "This run is marked as smoke, not final.", ""]
    if not candidate:
        lines.extend(["No eval-after-train metrics were produced.", ""])
    else:
        lines.extend(["## Candidate", ""])
        for key in sorted(candidate):
            lines.append(f"- {key}: {candidate[key]}")
        lines.append("")

    if baseline:
        lines.extend(["## Baseline Comparison", ""])
        for key in sorted(candidate or {}):
            baseline_value = baseline.get(key)
            candidate_value = candidate.get(key)
            if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float)):
                lines.append(f"- {key}: candidate={candidate_value}, baseline={baseline_value}, delta={candidate_value - baseline_value}")
        lines.append("")
    else:
        lines.extend(["Baseline metrics not found; comparison skipped.", ""])

    _write_text(path, "\n".join(lines))


def _generate_text(model, tokenizer, prompt, max_new_tokens, enable_thinking=None):
    try:
        import torch
    except ImportError:
        torch = None

    prompt_text = _generation_prompt_to_text(prompt, tokenizer, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    if torch is None:
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _model_load_kwargs(config):
    kwargs = {"trust_remote_code": bool(config["trust_remote_code"])}
    dtype = _torch_dtype(config["torch_dtype"])
    if dtype is not None:
        kwargs["dtype"] = dtype
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = config["attn_implementation"]
    return kwargs


def _torch_dtype(name):
    if name in {None, "auto"}:
        return "auto" if name == "auto" else None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for non-auto torch_dtype settings") from exc
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"unsupported torch_dtype: {name}")
    return aliases[name]


def _bnb_compute_dtype(name):
    dtype = _torch_dtype(name)
    if dtype != "auto":
        return dtype
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for QLoRA compute dtype settings") from exc
    return torch.bfloat16


def _generation_prompt_to_text(prompt, tokenizer, enable_thinking=None):
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        except ValueError:
            return prompt
    return prompt


def _write_run_card(path, resolved, data_hash, config_hash, git_commit, final_loss, metrics):
    finality = "smoke, not final" if resolved["smoke_run"] else "pipeline check, not final"
    lines = [
        "# SFT Run Card",
        "",
        f"- run name: `{resolved['run_name']}`",
        f"- smoke run: `{resolved['smoke_run']}`",
        f"- finality: `{finality}`",
        f"- base model: `{resolved['model_name_or_path']}`",
        f"- data path: `{resolved['data_path']}`",
        f"- data hash: `{data_hash}`",
        f"- config hash: `{config_hash}`",
        f"- git commit: `{git_commit}`",
        f"- output path: `{resolved['output_dir']}`",
        f"- adapter/checkpoint path: `{resolved['output_dir']}`",
        f"- final loss: `{final_loss}`",
        f"- train loss: `{metrics['train_loss']}`",
        f"- validation loss: `{metrics['validation_loss']}`",
        f"- format success: `{metrics['format_success']}`",
        f"- target eval score: `{metrics['target_eval_score']}`",
        f"- average output length: `{metrics['average_output_length']}`",
        f"- dry run: `{resolved['dry_run']}`",
        f"- train examples: `{resolved['selection']['max_train_examples']}`",
        f"- validation examples: `{resolved['selection']['max_validation_examples']}`",
        f"- max steps: `{resolved['training']['max_steps']}`",
        f"- max sequence length: `{resolved['training']['max_seq_length']}`",
        f"- save steps: `{resolved['training']['save_steps']}`",
        f"- save total limit: `{resolved['training']['save_total_limit']}`",
        f"- generation check max new tokens: `{resolved['generation_check']['max_new_tokens']}`",
        f"- generation check enable thinking: `{resolved['generation_check']['enable_thinking']}`",
        f"- eval max new tokens: `{resolved['eval_after_train']['max_new_tokens']}`",
        f"- eval enable thinking: `{resolved['eval_after_train']['enable_thinking']}`",
        f"- eval prompt source: `{resolved['eval_after_train']['prompt_source']}`",
        f"- loss curve path: `{resolved['output_dir']}/loss_curve.csv`",
        f"- checkpoint manifest path: `{resolved['output_dir']}/checkpoint_manifest.json`",
        f"- eval output path: `{resolved['eval_after_train']['output_dir']}`",
        f"- eval diff path: `{resolved['output_dir']}/eval_diff.md`",
    ]
    if resolved.get("dataset", {}).get("id"):
        lines.extend(
            [
                f"- source dataset: `{resolved['dataset']['id']}`",
                f"- source dataset config: `{resolved['dataset']['config']}`",
                f"- source dataset split: `{resolved['dataset']['split']}`",
                f"- dataset shuffle seed: `{resolved['seed']}`",
            ]
        )
    _write_text(path, "\n".join(lines) + "\n")


def _write_metrics(path, rows):
    _write_jsonl(path, rows)


def _write_loss_curve(path, rows):
    fields = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "mean_token_accuracy",
        "eval_mean_token_accuracy",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if "loss" not in row and "eval_loss" not in row:
                continue
            writer.writerow({field: row.get(field) for field in fields})


def _write_checkpoint_manifest(path, output_dir):
    checkpoints = []
    for checkpoint_dir in sorted(Path(output_dir).glob("checkpoint-*"), key=_checkpoint_step):
        checkpoints.append({"step": _checkpoint_step(checkpoint_dir), "path": str(checkpoint_dir)})
    _write_text(Path(path), json.dumps({"checkpoints": checkpoints}, indent=2, sort_keys=True) + "\n")


def _checkpoint_step(path):
    try:
        return int(Path(path).name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _last_metric(rows, key):
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def _average_generation_length(sample_rows, eval_metrics):
    if eval_metrics.get("completion_length_mean") is not None:
        return eval_metrics["completion_length_mean"]
    if not sample_rows:
        return None
    return sum(len(row["generation"]) for row in sample_rows) / len(sample_rows)


def _example_prompt(example):
    for message in example["messages"]:
        if message["role"] == "user":
            return message["content"]
    return example["messages"][0]["content"]


def _dry_generation_for_prompt(prompt):
    if "boxed" in prompt.lower():
        return r"\boxed{4}"
    return "<dry-run generation>"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _load_yaml_subset(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    config = {}
    section = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key, value = _split_key_value(line)
            if value == "":
                config[key] = {}
                section = key
            else:
                config[key] = _parse_scalar(value)
                section = None
            continue
        if section is None:
            raise ValueError(f"nested config entry without section: {line}")
        key, value = _split_key_value(line.strip())
        config[section][key] = _parse_scalar(value)
    return config


def _dump_yaml(payload):
    lines = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key in sorted(value):
                lines.append(f"  {child_key}: {_format_scalar(value[child_key])}")
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _split_key_value(line):
    if ":" not in line:
        raise ValueError(f"invalid config line: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value):
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"null", "None"}:
        return None
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _format_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
