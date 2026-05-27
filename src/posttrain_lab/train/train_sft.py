"""SFT overfit-32 pipeline using TRL SFTTrainer and PEFT adapters."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl


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

    if resolved.get("dry_run") and resolved.get("synthetic_data_if_missing") and not data_path.exists():
        _write_synthetic_sft_jsonl(data_path, count=resolved["selection"]["max_train_examples"])

    examples = load_sft_train_examples(
        data_path,
        split=resolved["selection"]["split"],
        limit=resolved["selection"]["max_train_examples"],
    )

    data_hash = _sha256_file(data_path)
    config_hash = _sha256_file(config_path)
    git_commit = _git_commit()

    _write_text(output_dir / "resolved_config.yaml", _dump_yaml(resolved))

    if resolved["dry_run"]:
        final_loss = 0.0
        _write_metrics(output_dir / "metrics.jsonl", [{"step": 0, "final_loss": final_loss, "dry_run": True}])
        _write_generation_check(output_dir / "sample_generations.jsonl", resolved, dry_run=True)
    else:
        final_loss, model, tokenizer = _run_trl_training(resolved, examples, output_dir)
        _write_generation_check(
            output_dir / "sample_generations.jsonl",
            resolved,
            dry_run=False,
            model=model,
            tokenizer=tokenizer,
        )

    _write_run_card(
        output_dir / "run_card.md",
        resolved=resolved,
        data_hash=data_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        final_loss=final_loss,
    )
    return {
        "output_dir": str(output_dir),
        "final_loss": final_loss,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SFT overfit-32 with TRL SFTTrainer.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_sft(config, config_path=args.config)
    print(json.dumps(result, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = copy.deepcopy(config)
    resolved["selection"].setdefault("split", "train")
    resolved["selection"].setdefault("max_train_examples", 32)
    resolved["training"].setdefault("max_steps", 2)
    resolved["training"].setdefault("per_device_train_batch_size", 1)
    resolved["training"].setdefault("gradient_accumulation_steps", 1)
    resolved["training"].setdefault("learning_rate", 2e-4)
    resolved["training"].setdefault("max_seq_length", 512)
    resolved["peft"].setdefault("method", "lora")
    resolved["peft"].setdefault("qlora", False)
    resolved["peft"].setdefault("r", 8)
    resolved["peft"].setdefault("lora_alpha", 16)
    resolved["peft"].setdefault("lora_dropout", 0.0)
    resolved["generation_check"].setdefault("max_new_tokens", 32)
    resolved["generation_check"].setdefault("prompts", [])
    resolved.setdefault("synthetic_data_if_missing", False)
    if resolved["selection"]["max_train_examples"] != 32:
        raise ValueError("overfit-32 requires selection.max_train_examples == 32")
    return resolved


def _run_trl_training(config, examples, output_dir):
    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("TRL SFT training requires transformers, datasets, trl, and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if config["peft"]["qlora"]:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA requires BitsAndBytesConfig from transformers") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
    if config["peft"]["qlora"]:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(config["peft"]["r"]),
        lora_alpha=int(config["peft"]["lora_alpha"]),
        lora_dropout=float(config["peft"]["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list([{"text": _messages_to_text(example["messages"], tokenizer)} for example in examples])
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(config["training"]["max_steps"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=1,
        save_steps=int(config["training"]["max_steps"]),
        report_to=[],
        seed=int(config["seed"]),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=int(config["training"]["max_seq_length"]),
        args=args,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))

    final_loss = float(result.training_loss)
    _write_metrics(output_dir / "metrics.jsonl", trainer.state.log_history + [{"final_loss": final_loss}])
    return final_loss, trainer.model, tokenizer


def _messages_to_text(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _write_generation_check(path, config, dry_run, model=None, tokenizer=None):
    rows = []
    for index, prompt in enumerate(config["generation_check"].get("prompts", [])):
        if dry_run:
            generation = "<dry-run generation>"
        else:
            generation = _generate_text(model, tokenizer, prompt, config["generation_check"]["max_new_tokens"])
        rows.append(
            {
                "id": f"generation-check-{index}",
                "prompt": prompt,
                "generation": generation,
                "dry_run": dry_run,
            }
        )
    _write_jsonl(path, rows)


def _write_synthetic_sft_jsonl(path, count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"sft-overfit32-{index:03d}",
                "split": "train",
                "messages": [
                    {"role": "user", "content": f"Compute {index} + 1."},
                    {"role": "assistant", "content": str(index + 1)},
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


def _generate_text(model, tokenizer, prompt, max_new_tokens):
    try:
        import torch
    except ImportError:
        torch = None

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    if torch is None:
        outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)

    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _write_run_card(path, resolved, data_hash, config_hash, git_commit, final_loss):
    lines = [
        "# SFT Overfit-32 Run Card",
        "",
        f"- base model: `{resolved['model_name_or_path']}`",
        f"- data path: `{resolved['data_path']}`",
        f"- data hash: `{data_hash}`",
        f"- config hash: `{config_hash}`",
        f"- git commit: `{git_commit}`",
        f"- output path: `{resolved['output_dir']}`",
        f"- adapter/checkpoint path: `{resolved['output_dir']}`",
        f"- final loss: `{final_loss}`",
        f"- dry run: `{resolved['dry_run']}`",
        f"- train examples: `{resolved['selection']['max_train_examples']}`",
        f"- max steps: `{resolved['training']['max_steps']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_metrics(path, rows):
    _write_jsonl(path, rows)


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


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
