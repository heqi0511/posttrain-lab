"""End-to-end toy math post-training smoke pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from posttrain_lab.data.validate import validate_file
from posttrain_lab.eval.eval_runner import run_eval
from posttrain_lab.rewards.math_reward import math_boxed_v001, score_math_boxed_v001
from posttrain_lab.train.train_grpo import run_grpo
from posttrain_lab.train.train_sft import run_sft


def run_e2e_smoke(config_path, output_dir=None, real_run=False):
    """Run the full tiny math post-training smoke pipeline."""

    config_path = Path(config_path)
    config = _load_yaml_subset(config_path)
    limits = config["limits"]
    run_dir = Path(output_dir or config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = _run_mode(config, real_run)
    model_name = config["real_model_name_or_path"] if real_run else config["model_name_or_path"]
    dry_run = not real_run

    sft_seed_path = Path(config["sft_seed_path"])
    rlvr_seed_path = Path(config["rlvr_seed_path"])
    eval_prompt_path = Path(config["eval_prompt_path"])
    _validate_or_fail(sft_seed_path, "sft")
    _validate_or_fail(rlvr_seed_path, "rlvr")

    staged_dir = run_dir / "staged_data"
    sft_data_path = staged_dir / "sft_smoke.jsonl"
    rlvr_data_path = staged_dir / "rlvr_toy.jsonl"
    _stage_sft_data(
        source_path=sft_seed_path,
        output_path=sft_data_path,
        train_count=int(limits["sft_train_examples"]),
        val_count=int(limits["sft_val_examples"]),
    )
    _stage_rlvr_data(
        source_path=rlvr_seed_path,
        output_path=rlvr_data_path,
        train_count=int(limits["rlvr_train_examples"]),
    )
    _validate_or_fail(sft_data_path, "sft")
    _validate_or_fail(rlvr_data_path, "rlvr")
    _assert_no_eval_leakage(sft_data_path, rlvr_data_path, eval_prompt_path)

    baseline_metrics = _run_eval_stage(
        run_dir=run_dir,
        stage="base",
        report_name="baseline_eval_report.json",
        prompt_path=eval_prompt_path,
        model_name=model_name,
        adapter_path=None,
        dry_run=dry_run,
        config=config,
    )
    _enforce_eval_gates(
        baseline_metrics,
        run_dir / "evals" / "base",
        limits,
        parse_threshold_key="max_baseline_parse_failure_rate",
    )

    sft_result = run_sft(
        _sft_config(config, model_name, sft_data_path, run_dir / "sft", dry_run),
        config_path=config_path,
    )
    shutil.copyfile(run_dir / "sft" / "run_card.md", run_dir / "sft_run_card.md")
    shutil.copyfile(run_dir / "sft" / "sample_generations.jsonl", run_dir / "sample_generations.jsonl")
    _enforce_sample_generation_length(run_dir / "sample_generations.jsonl", limits)

    sft_metrics = _run_eval_stage(
        run_dir=run_dir,
        stage="sft",
        report_name="sft_eval_report.json",
        prompt_path=eval_prompt_path,
        model_name=model_name,
        adapter_path=str(run_dir / "sft") if real_run else None,
        dry_run=dry_run,
        config=config,
    )
    _enforce_eval_gates(sft_metrics, run_dir / "evals" / "sft", limits)

    reward_metrics = _run_reward_checks()

    rlvr_result = run_grpo(
        _rlvr_config(config, model_name, rlvr_data_path, run_dir / "rlvr", run_dir / "sft", dry_run),
        config_path=config_path,
    )
    shutil.copyfile(run_dir / "rlvr" / "run_card.md", run_dir / "rlvr_run_card.md")
    shutil.copyfile(run_dir / "rlvr" / "sample_rollouts.jsonl", run_dir / "sample_rollouts.jsonl")
    _enforce_rollout_gates(rlvr_result, run_dir / "sample_rollouts.jsonl", limits)

    rlvr_metrics = _run_eval_stage(
        run_dir=run_dir,
        stage="sft_rlvr",
        report_name="rlvr_eval_report.json",
        prompt_path=eval_prompt_path,
        model_name=model_name,
        adapter_path=str(run_dir / "rlvr") if real_run else None,
        dry_run=dry_run,
        config=config,
    )
    _enforce_eval_gates(rlvr_metrics, run_dir / "evals" / "sft_rlvr", limits)

    _write_comparison_report(
        run_dir / "comparison_report.md",
        mode=mode,
        baseline=baseline_metrics,
        sft=sft_metrics,
        rlvr=rlvr_metrics,
        reward_metrics=reward_metrics,
        rlvr_result=rlvr_result,
    )

    result = {
        "mode": mode,
        "output_dir": str(run_dir),
        "baseline_metrics": baseline_metrics,
        "sft_metrics": sft_metrics,
        "rlvr_metrics": rlvr_metrics,
        "reward_metrics": reward_metrics,
    }
    (run_dir / "pipeline_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the toy math post-training e2e smoke pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--real-run", action="store_true", help="Allow real model loading/training.")
    args = parser.parse_args(argv)

    result = run_e2e_smoke(args.config, output_dir=args.output_dir, real_run=args.real_run)
    print(json.dumps(result, sort_keys=True))
    return 0


def _sft_config(config, model_name, data_path, output_dir, dry_run):
    max_steps = int(config["limits"]["sft_max_steps"] if dry_run else config["limits"]["real_sft_max_steps"])
    learning_rate = float(0.0001 if dry_run else config["limits"]["real_sft_learning_rate"])
    target_modules = [] if dry_run else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return {
        "run_name": "e2e-sft-diverse-smoke",
        "model_name_or_path": model_name,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "synthetic_data_if_missing": False,
        "smoke_run": True,
        "seed": int(config["seed"]),
        "torch_dtype": config["torch_dtype"],
        "trust_remote_code": bool(config["trust_remote_code"]),
        "selection": {
            "train_split": "train",
            "validation_split": "val",
            "max_train_examples": int(config["limits"]["sft_train_examples"]),
            "max_validation_examples": int(config["limits"]["sft_val_examples"]),
        },
        "training": {
            "max_steps": max_steps,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": learning_rate,
            "max_seq_length": 128,
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "eval_steps": 1,
            "save_steps": max_steps,
            "save_total_limit": 1,
        },
        "peft": {
            "method": "lora",
            "qlora": False,
            "r": 4 if dry_run else 16,
            "lora_alpha": 8 if dry_run else 32,
            "lora_dropout": 0.0,
            "target_modules": target_modules,
        },
        "generation_check": {
            "max_new_tokens": int(config["limits"]["eval_max_new_tokens"]),
            "enable_thinking": bool(config["enable_thinking"]),
            "random_sample_count": 4,
            "prompts": ["Compute 2 + 2. Return only the final answer in boxed format."],
        },
        "eval_after_train": {"enabled": False},
    }


def _rlvr_config(config, model_name, data_path, output_dir, sft_adapter_path, dry_run):
    return {
        "run_name": "e2e-rlvr-grpo-toy",
        "model_name_or_path": model_name,
        "adapter_path": str(sft_adapter_path) if not dry_run else None,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "synthetic_data_if_missing": False,
        "smoke_run": True,
        "seed": int(config["seed"]),
        "torch_dtype": config["torch_dtype"],
        "trust_remote_code": bool(config["trust_remote_code"]),
        "enable_thinking": bool(config["enable_thinking"]),
        "reward_version": "math_boxed_v001",
        "selection": {
            "train_split": "train",
            "max_train_examples": int(config["limits"]["rlvr_train_examples"]),
        },
        "training": {
            "max_steps": int(config["limits"]["rlvr_max_steps"]),
            "per_device_train_batch_size": int(config["limits"]["rlvr_per_device_train_batch_size"]),
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.000001,
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "save_steps": int(config["limits"]["rlvr_max_steps"]),
            "save_total_limit": 1,
        },
        "rollout": {
            "num_generations": int(config["limits"]["rlvr_num_generations"]),
            "max_completion_length": int(config["limits"]["eval_max_new_tokens"]),
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "beta": 0.0,
            "sample_count": int(config["limits"]["rlvr_train_examples"]),
        },
        "rollout_format_gate": {
            "enabled": True,
            "sample_count": min(int(config["limits"]["rlvr_gate_examples"]), int(config["limits"]["rlvr_train_examples"])),
            "max_parse_failure_rate": float(config["limits"]["max_parse_failure_rate"]),
            "max_reward_mean": None if dry_run else float(config["limits"]["max_rollout_gate_reward_mean"]),
            "max_perfect_reward_rate": None if dry_run else float(config["limits"]["max_rollout_gate_perfect_reward_rate"]),
        },
        "peft": {
            "method": "lora",
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": [],
        },
        "eval_after_train": {"enabled": False},
    }


def _run_eval_stage(run_dir, stage, report_name, prompt_path, model_name, adapter_path, dry_run, config):
    eval_config = {
        "prompt_path": str(prompt_path),
        "output_dir": str(run_dir / "evals" / stage),
        "dry_run": dry_run,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "torch_dtype": config["torch_dtype"],
        "trust_remote_code": bool(config["trust_remote_code"]),
        "apply_chat_template": not dry_run,
        "enable_thinking": bool(config["enable_thinking"]),
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": int(config["limits"]["eval_max_new_tokens"]),
            "stop_tokens": [],
        },
        "metrics": {
            "exact_match": True,
            "format_regex": r"^\\boxed\{.+\}$",
        },
    }
    metrics = run_eval(eval_config)
    payload = {"stage": stage, "metrics": metrics, "eval_output_dir": eval_config["output_dir"]}
    (run_dir / report_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _run_reward_checks():
    cases = [
        (r"\boxed{4}", "4", 1.0),
        (r"\boxed{5}", "4", 0.0),
        ("no boxed answer", "4", 0.0),
    ]
    rewards = []
    for completion, answer, expected in cases:
        reward = math_boxed_v001(completion, answer)
        if reward != expected:
            raise RuntimeError(f"reward check failed for {completion!r}: expected {expected}, got {reward}")
        rewards.append(reward)
    adversarial = score_math_boxed_v001(r"\boxed{4} and also \boxed{4}", "4")
    if adversarial.score != 0.0:
        raise RuntimeError("reward check failed: repeated boxed answers must score 0 under math_boxed_v001")
    rewards.append(adversarial.score)
    return {
        "reward_mean": sum(rewards) / len(rewards),
        "reward_std": _pstdev(rewards),
        "num_reward_checks": len(rewards),
    }


def _stage_sft_data(source_path, output_path, train_count, val_count):
    rows = _read_jsonl(source_path)
    staged = _select_split(rows, "train", train_count, source_path) + _select_split(
        rows, "val", val_count, source_path
    )
    _write_jsonl(output_path, staged)


def _stage_rlvr_data(source_path, output_path, train_count):
    rows = _read_jsonl(source_path)
    staged = _select_split(rows, "train", train_count, source_path)
    _write_jsonl(output_path, staged)


def _select_split(rows, split, count, source_path):
    selected = [row for row in rows if row.get("split") == split]
    if len(selected) < count:
        raise RuntimeError(f"{source_path} has {len(selected)} {split} rows, expected at least {count}")
    return selected[:count]


def _assert_no_eval_leakage(sft_data_path, rlvr_data_path, eval_prompt_path):
    sft_rows = _read_jsonl(sft_data_path)
    rlvr_rows = _read_jsonl(rlvr_data_path)
    eval_rows = _read_jsonl(eval_prompt_path)

    sft_train = _normalized_prompts(row for row in sft_rows if row.get("split") == "train")
    sft_val = _normalized_prompts(row for row in sft_rows if row.get("split") == "val")
    rlvr_train = _normalized_prompts(row for row in rlvr_rows if row.get("split") == "train")
    eval_prompts = _normalized_prompts(eval_rows)

    _fail_on_prompt_overlap("SFT train", sft_train, "SFT val", sft_val)
    _fail_on_prompt_overlap("SFT train", sft_train, "eval", eval_prompts)
    _fail_on_prompt_overlap("SFT val", sft_val, "eval", eval_prompts)
    _fail_on_prompt_overlap("RLVR train", rlvr_train, "eval", eval_prompts)


def _normalized_prompts(rows):
    return {_normalize_prompt(_prompt_text(row)) for row in rows}


def _prompt_text(row):
    if "messages" in row:
        for message in row["messages"]:
            if message.get("role") == "user":
                return message.get("content", "")
        return ""
    if "prompt" in row:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            return "\n".join(str(message.get("content", "")) for message in prompt)
        return str(prompt)
    return str(row.get("prompt", ""))


def _normalize_prompt(prompt):
    return " ".join(str(prompt).lower().split())


def _fail_on_prompt_overlap(left_name, left, right_name, right):
    overlap = sorted(left & right)
    if overlap:
        sample = overlap[0]
        raise RuntimeError(f"prompt leakage between {left_name} and {right_name}: {sample}")


def _validate_or_fail(path, dataset_type):
    report = validate_file(path, dataset_type)
    if not report.ok:
        messages = "; ".join(str(error) for error in report.errors)
        raise RuntimeError(f"{dataset_type} data validation failed for {path}: {messages}")


def _enforce_eval_gates(metrics, eval_output_dir, limits, parse_threshold_key="max_parse_failure_rate"):
    parse_failure_rate = metrics.get("parse_failure_rate")
    threshold = float(limits[parse_threshold_key])
    if parse_failure_rate is not None and parse_failure_rate > threshold:
        raise RuntimeError(f"parse failure rate {parse_failure_rate} exceeds configured threshold")
    _enforce_raw_generation_length(Path(eval_output_dir) / "raw_generations.jsonl", limits)


def _enforce_raw_generation_length(path, limits):
    max_length = int(limits["max_output_length"])
    for row in _read_jsonl(path):
        if int(row.get("completion_length", 0)) > max_length:
            raise RuntimeError(f"output length exceeds configured limit in {path}: {row.get('id')}")


def _enforce_sample_generation_length(path, limits):
    max_length = int(limits["max_output_length"])
    for row in _read_jsonl(path):
        if len(str(row.get("generation", ""))) > max_length:
            raise RuntimeError(f"sample generation exceeds configured limit in {path}: {row.get('id')}")


def _enforce_rollout_gates(result, rollout_path, limits):
    if result["parse_failure_rate"] > float(limits["max_parse_failure_rate"]):
        raise RuntimeError(f"RLVR parse failure rate {result['parse_failure_rate']} exceeds threshold")
    max_length = int(limits["max_output_length"])
    for row in _read_jsonl(rollout_path):
        if int(row.get("completion_length", 0)) > max_length:
            raise RuntimeError(f"rollout completion length exceeds configured limit: {row.get('id')}")


def _write_comparison_report(path, mode, baseline, sft, rlvr, reward_metrics, rlvr_result):
    rows = [
        ("Base", baseline, None),
        ("SFT", sft, None),
        ("SFT+RLVR", rlvr, rlvr_result),
    ]
    lines = [
        "# Toy Math Post-Training E2E Comparison",
        "",
        f"Run mode: `{mode}`.",
        "",
        "| stage | target accuracy | format success rate | parse failure rate | average output length | reward mean | reward std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics, rewards in rows:
        reward_mean = rewards.get("reward_mean") if rewards else None
        reward_std = rewards.get("reward_std") if rewards else None
        lines.append(
            "| {name} | {target} | {fmt} | {parse} | {length} | {reward_mean} | {reward_std} |".format(
                name=name,
                target=_fmt(metrics.get("exact_match")),
                fmt=_fmt(metrics.get("format_success")),
                parse=_fmt(metrics.get("parse_failure_rate")),
                length=_fmt(metrics.get("completion_length_mean")),
                reward_mean=_fmt(reward_mean),
                reward_std=_fmt(reward_std),
            )
        )

    lines.extend(
        [
            "",
            "## Reward Checks",
            "",
            f"- reward mean: `{_fmt(reward_metrics['reward_mean'])}`",
            f"- reward std: `{_fmt(reward_metrics['reward_std'])}`",
            f"- checks: `{reward_metrics['num_reward_checks']}`",
            "",
            "## Heldout Eval Conclusion",
            "",
            _heldout_conclusion(sft, rlvr),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _heldout_conclusion(sft, rlvr):
    sft_score = sft.get("exact_match")
    rlvr_score = rlvr.get("exact_match")
    if not isinstance(sft_score, (int, float)) or not isinstance(rlvr_score, (int, float)):
        return "Heldout improvement is unavailable because target accuracy is missing."
    delta = rlvr_score - sft_score
    if delta > 0:
        return f"SFT+RLVR improved heldout target accuracy over SFT by {delta:.4f}."
    if delta < 0:
        return f"SFT+RLVR regressed heldout target accuracy versus SFT by {delta:.4f}."
    return "SFT+RLVR did not change heldout target accuracy versus SFT."


def _run_mode(config, real_run):
    if not real_run:
        return "dry-run"
    if "tiny" in str(config["real_model_name_or_path"]).lower():
        return "tiny-model run"
    return "real-model run"


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pstdev(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_yaml_subset(path):
    config = {}
    section = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
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
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


if __name__ == "__main__":
    raise SystemExit(main())
