"""Offline rollout audit for selecting frontier RLVR prompts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl
from posttrain_lab.rewards.math_reward import MathRewardConfig, score_math_boxed_v001
from posttrain_lab.train.train_grpo import (
    PARSE_FAILURE_REASONS,
    _empty_cuda_cache,
    _load_policy_model_and_tokenizer,
    _model_device,
    _prompt_text,
)
from posttrain_lab.train.train_sft import _dump_yaml, _generation_prompt_to_text, _load_yaml_subset


REQUIRED_TOP_LEVEL = {
    "input_path",
    "output_dir",
    "filtered_output_path",
    "excluded_output_path",
    "dry_run",
    "model_name_or_path",
    "seed",
    "selection",
    "rollout",
    "frontier",
}


def load_config(path):
    """Load and validate an offline rollout-audit config."""

    config = _load_yaml_subset(Path(path))
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"missing required config fields: {', '.join(missing)}")
    return config


def run_rollout_audit(config, config_path=None):
    """Sample completions without training and write frontier-selection artifacts."""

    resolved = _resolve_config(config)
    random.seed(int(resolved["seed"]))

    input_path = Path(resolved["input_path"])
    output_dir = Path(resolved["output_dir"])
    filtered_output_path = Path(resolved["filtered_output_path"])
    excluded_output_path = Path(resolved["excluded_output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = _load_rlvr_examples(
        input_path,
        split=resolved["selection"]["split"],
        limit=int(resolved["selection"]["max_prompts"]),
    )
    _write_text(output_dir / "resolved_config.yaml", _dump_yaml(resolved))

    model = None
    tokenizer = None
    if not resolved["dry_run"]:
        model, tokenizer, _ = _load_policy_model_and_tokenizer(resolved, move_to_accelerator=True)
        if hasattr(model, "eval"):
            model.eval()

    try:
        prompt_reports = []
        review_rows = []
        kept_records = []
        excluded_rows = []

        for prompt_index, example in enumerate(examples):
            rollouts = _audit_prompt(
                example=example,
                prompt_index=prompt_index,
                config=resolved,
                model=model,
                tokenizer=tokenizer,
            )
            report = _summarize_prompt(example, prompt_index, rollouts)
            keep, reason, reason_detail = _frontier_decision(report, resolved["frontier"])
            report["keep"] = keep
            report["exclude_reason"] = "" if keep else reason
            report["exclude_reason_detail"] = "" if keep else reason_detail
            prompt_reports.append(report)

            if keep:
                kept_records.append(example)
            else:
                excluded_rows.append(
                    {
                        "id": example["id"],
                        "reason": reason,
                        "reason_detail": reason_detail,
                        "record": example,
                    }
                )

            if prompt_index < int(resolved["review"]["max_prompts"]):
                review_rows.extend(rollouts)

        _write_prompt_csv(output_dir / "rollout_audit_by_prompt.csv", prompt_reports)
        _write_jsonl(output_dir / "sample_rollouts_for_review.jsonl", review_rows)
        _write_jsonl(filtered_output_path, kept_records)
        _write_jsonl(excluded_output_path, excluded_rows)

        summary = _summary(
            prompt_reports=prompt_reports,
            config=resolved,
            filtered_output_path=filtered_output_path,
            excluded_output_path=excluded_output_path,
            config_path=config_path,
        )
        _write_text(output_dir / "rollout_audit_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        _empty_cuda_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit RLVR prompts with offline rollouts and filter frontier data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    summary = run_rollout_audit(config, config_path=args.config)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("adapter_path", None)
    resolved.setdefault("torch_dtype", "auto")
    resolved.setdefault("trust_remote_code", False)
    resolved.setdefault("enable_thinking", False)
    resolved.setdefault("reward_version", "math_boxed_v001")
    resolved.setdefault("peft", {})
    resolved["peft"].setdefault("method", "lora")
    resolved["peft"].setdefault("r", 4)
    resolved["peft"].setdefault("lora_alpha", 8)
    resolved["peft"].setdefault("lora_dropout", 0.0)
    resolved["peft"].setdefault("target_modules", [])
    resolved["selection"].setdefault("split", "train")
    resolved["selection"].setdefault("max_prompts", 20)
    resolved["rollout"].setdefault("completions_per_prompt", 16)
    resolved["rollout"].setdefault("max_new_tokens", 32)
    resolved["rollout"].setdefault("temperature", 0.7)
    resolved["rollout"].setdefault("top_p", 0.95)
    resolved["rollout"].setdefault("top_k", 0)
    resolved["frontier"].setdefault("min_reward_mean", 0.2)
    resolved["frontier"].setdefault("max_reward_mean", 0.8)
    resolved["frontier"].setdefault("max_parse_failure_rate", 0.5)
    resolved["frontier"].setdefault("min_unique_answer_count", 3)
    resolved.setdefault("review", {})
    resolved["review"].setdefault("max_prompts", min(20, int(resolved["selection"]["max_prompts"])))

    if resolved["reward_version"] != "math_boxed_v001":
        raise ValueError(f"unsupported reward_version: {resolved['reward_version']}")
    if int(resolved["rollout"]["completions_per_prompt"]) < 2:
        raise ValueError("rollout.completions_per_prompt must be >= 2")
    return resolved


def _load_rlvr_examples(path, split, limit):
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


def _audit_prompt(example, prompt_index, config, model=None, tokenizer=None):
    prompt = _prompt_text(example["prompt"])
    answer = example["verifier"]["answer"]
    rows = []
    count = int(config["rollout"]["completions_per_prompt"])
    for sample_index in range(count):
        if config["dry_run"]:
            completion = _mock_completion(answer, prompt_index, sample_index)
        else:
            completion = _sample_completion(model, tokenizer, prompt, config)
        reward_result = score_math_boxed_v001(completion, answer, config=MathRewardConfig())
        rows.append(
            {
                "id": example["id"],
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "prompt": prompt,
                "answer": answer,
                "completion": completion,
                "reward": reward_result.score,
                "parsed_answer": reward_result.normalized_prediction,
                "failure_reason": None if reward_result.score == 1.0 else reward_result.reason,
                "parse_failure": reward_result.reason in PARSE_FAILURE_REASONS,
                "completion_length": len(completion),
            }
        )
    return rows


def _mock_completion(answer, prompt_index, sample_index):
    pattern = prompt_index % 5
    if pattern == 0:
        if sample_index % 2 == 0:
            return rf"\boxed{{{answer}}}"
        return rf"\boxed{{{_wrong_answer(answer, sample_index)}}}"
    if pattern == 1:
        return rf"\boxed{{{_wrong_answer(answer, sample_index)}}}"
    if pattern == 2:
        return rf"\boxed{{{answer}}}"
    if pattern == 3:
        return f"Final answer: {answer}"
    if sample_index % 2 == 0:
        return rf"\boxed{{{answer}}}"
    return r"\boxed{0}"


def _wrong_answer(answer, sample_index):
    if answer.lstrip("-").isdigit():
        return str(int(answer) + sample_index + 1)
    return f"{answer}_wrong_{sample_index}"


def _sample_completion(model, tokenizer, prompt, config):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for real rollout audit generation") from exc

    prompt_text = _generation_prompt_to_text(prompt, tokenizer, enable_thinking=config.get("enable_thinking"))
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = _model_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    temperature = float(config["rollout"]["temperature"])
    generation_kwargs = {
        "max_new_tokens": int(config["rollout"]["max_new_tokens"]),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(config["rollout"]["top_p"])
        top_k = int(config["rollout"]["top_k"])
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _summarize_prompt(example, prompt_index, rollouts):
    rewards = [float(row["reward"]) for row in rollouts]
    correct_count = sum(1 for reward in rewards if reward == 1.0)
    parsed_answers = [row["parsed_answer"] for row in rollouts if row["parsed_answer"] is not None]
    bucket = "mixed"
    if correct_count == 0:
        bucket = "all_zero"
    elif correct_count == len(rollouts):
        bucket = "all_one"

    metadata = example.get("metadata", {})
    return {
        "id": example["id"],
        "split": example["split"],
        "prompt_index": prompt_index,
        "prompt": _prompt_text(example["prompt"]),
        "answer": example["verifier"]["answer"],
        "reward_mean": sum(rewards) / len(rewards),
        "reward_std": statistics.pstdev(rewards),
        "correct_count": correct_count,
        "bucket": bucket,
        "parse_failure_rate": sum(1 for row in rollouts if row["parse_failure"]) / len(rollouts),
        "unique_answer_count": len(set(parsed_answers)),
        "avg_completion_length": sum(row["completion_length"] for row in rollouts) / len(rollouts),
        "template": metadata.get("template", ""),
        "task_type": metadata.get("task_type", metadata.get("domain", "")),
        "domain": metadata.get("domain", ""),
        "difficulty": metadata.get("difficulty", ""),
        "source": metadata.get("source", ""),
        "license": metadata.get("license", ""),
    }


def _frontier_decision(report, frontier):
    if report["bucket"] == "all_zero" and report["parse_failure_rate"] <= float(frontier["max_parse_failure_rate"]):
        return False, "all_zero", "all completions received zero reward"
    if report["bucket"] == "all_one":
        return False, "all_one", "all completions received perfect reward"
    if report["parse_failure_rate"] > float(frontier["max_parse_failure_rate"]):
        return False, "parse_fail", "parse_failure_rate above threshold"
    if report["unique_answer_count"] < int(frontier["min_unique_answer_count"]):
        return False, "low_diversity", "unique_answer_count below threshold"
    if report["reward_mean"] < float(frontier["min_reward_mean"]):
        return False, "all_zero", "reward_mean below frontier band"
    if report["reward_mean"] > float(frontier["max_reward_mean"]):
        return False, "all_one", "reward_mean above frontier band"
    return True, "", ""


def _summary(prompt_reports, config, filtered_output_path, excluded_output_path, config_path):
    total = len(prompt_reports)
    bucket_counts = {
        "all_zero": sum(1 for report in prompt_reports if report["bucket"] == "all_zero"),
        "all_one": sum(1 for report in prompt_reports if report["bucket"] == "all_one"),
        "mixed": sum(1 for report in prompt_reports if report["bucket"] == "mixed"),
    }
    exclude_reason_counts = {}
    for report in prompt_reports:
        reason = report["exclude_reason"]
        if reason:
            exclude_reason_counts[reason] = exclude_reason_counts.get(reason, 0) + 1
    kept = sum(1 for report in prompt_reports if report["keep"])
    return {
        "config_path": str(config_path) if config_path is not None else None,
        "input_path": config["input_path"],
        "filtered_output_path": str(filtered_output_path),
        "excluded_output_path": str(excluded_output_path),
        "dry_run": bool(config["dry_run"]),
        "model_name_or_path": config["model_name_or_path"],
        "adapter_path": config.get("adapter_path"),
        "reward_version": config["reward_version"],
        "audited_prompt_count": total,
        "completions_per_prompt": int(config["rollout"]["completions_per_prompt"]),
        "all_zero_count": bucket_counts["all_zero"],
        "all_one_count": bucket_counts["all_one"],
        "mixed_count": bucket_counts["mixed"],
        "bucket_counts": bucket_counts,
        "all_zero_rate": bucket_counts["all_zero"] / total if total else 0.0,
        "all_one_rate": bucket_counts["all_one"] / total if total else 0.0,
        "mixed_rate": bucket_counts["mixed"] / total if total else 0.0,
        "parse_failure_rate": _mean_prompt_metric(prompt_reports, "parse_failure_rate"),
        "unique_answer_count": _mean_prompt_metric(prompt_reports, "unique_answer_count"),
        "avg_completion_length": _mean_prompt_metric(prompt_reports, "avg_completion_length"),
        "kept_prompt_count": kept,
        "excluded_prompt_count": total - kept,
        "exclude_reason_counts": exclude_reason_counts,
        "effective_mixed_group_rate": bucket_counts["mixed"] / total if total else 0.0,
        "selected_prompt_rate": kept / total if total else 0.0,
        "frontier_filter": config["frontier"],
        "no_training_executed": True,
    }


def _mean_prompt_metric(prompt_reports, key):
    if not prompt_reports:
        return 0.0
    return sum(float(report[key]) for report in prompt_reports) / len(prompt_reports)


def _write_prompt_csv(path, prompt_reports):
    fieldnames = [
        "id",
        "split",
        "prompt_index",
        "reward_mean",
        "reward_std",
        "correct_count",
        "bucket",
        "parse_failure_rate",
        "unique_answer_count",
        "avg_completion_length",
        "template",
        "task_type",
        "domain",
        "difficulty",
        "keep",
        "exclude_reason",
        "exclude_reason_detail",
        "prompt",
        "answer",
        "source",
        "license",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in prompt_reports:
            writer.writerow({field: report.get(field, "") for field in fieldnames})


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
