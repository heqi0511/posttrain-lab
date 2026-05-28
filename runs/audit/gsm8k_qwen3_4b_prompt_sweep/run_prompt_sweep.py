"""No-training GSM8K prompt/generation sweep for Qwen3-4B."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

from posttrain_lab.rewards.math_reward import MathRewardConfig, score_math_boxed_v001
from posttrain_lab.train.train_grpo import PARSE_FAILURE_REASONS, _empty_cuda_cache, _model_device
from posttrain_lab.train.train_sft import _generation_prompt_to_text


VARIANTS = {
    "current_128": {
        "label": "current prompt, max_new_tokens=128",
        "prompt_mode": "current",
        "max_new_tokens": 128,
        "enable_thinking": False,
        "generation_batch_size": 8,
    },
    "strong_128": {
        "label": "stronger boxed prompt, max_new_tokens=128",
        "prompt_mode": "strong",
        "max_new_tokens": 128,
        "enable_thinking": None,
        "generation_batch_size": 8,
    },
    "strong_512": {
        "label": "stronger boxed prompt, max_new_tokens=512",
        "prompt_mode": "strong",
        "max_new_tokens": 512,
        "enable_thinking": None,
        "generation_batch_size": 4,
    },
    "strong_1024": {
        "label": "stronger boxed prompt, max_new_tokens=1024",
        "prompt_mode": "strong",
        "max_new_tokens": 1024,
        "enable_thinking": None,
        "generation_batch_size": 2,
    },
    "strong_nonthinking_512": {
        "label": "non-thinking mode, stronger boxed prompt, max_new_tokens=512",
        "prompt_mode": "strong",
        "max_new_tokens": 512,
        "enable_thinking": False,
        "generation_batch_size": 4,
    },
    "strong_thinking_1024": {
        "label": "thinking mode, stronger boxed prompt, max_new_tokens=1024",
        "prompt_mode": "strong",
        "max_new_tokens": 1024,
        "enable_thinking": True,
        "generation_batch_size": 2,
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or merge a no-training GSM8K prompt sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)

    config = load_simple_yaml(Path(args.config))
    if args.merge:
        merge_reports(config)
        return 0
    if not args.variant:
        raise ValueError("--variant is required unless --merge is set")
    run_variant(config, args.variant)
    return 0


def run_variant(config, variant_name):
    if variant_name not in VARIANTS:
        raise ValueError(f"unknown variant: {variant_name}")
    variant = dict(VARIANTS[variant_name])
    output_dir = Path(config["output_dir"]) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_fixed_sample(
        path=Path(config["input_path"]),
        split=config["split"],
        sample_size=int(config["sample_size"]),
        seed=int(config["seed"]),
    )
    prompts = [build_prompt(record, config, variant) for record in records]
    write_jsonl(output_dir / "sampled_prompts.jsonl", prompts)
    write_json(output_dir / "resolved_variant.json", {"name": variant_name, **variant, "config": config})

    model, tokenizer = load_model(config)
    started_at = time.time()
    rows = []
    try:
        for prompt_index, prompt_record in enumerate(prompts):
            completions = sample_completions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_record["prompt"],
                max_new_tokens=int(variant["max_new_tokens"]),
                num_generations=int(config["num_generations"]),
                generation_batch_size=int(variant["generation_batch_size"]),
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                top_k=int(config["top_k"]),
                enable_thinking=variant["enable_thinking"],
            )
            for sample_index, completion in enumerate(completions):
                reward = score_math_boxed_v001(
                    completion["text"],
                    prompt_record["answer"],
                    config=MathRewardConfig(),
                )
                rows.append(
                    {
                        "id": prompt_record["id"],
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "variant": variant_name,
                        "prompt": prompt_record["prompt"],
                        "answer": prompt_record["answer"],
                        "completion": completion["text"],
                        "reward": reward.score,
                        "format_success": reward.normalized_prediction is not None,
                        "parse_failure": reward.reason in PARSE_FAILURE_REASONS,
                        "failure_reason": None if reward.score == 1.0 else reward.reason,
                        "parsed_answer": reward.normalized_prediction,
                        "completion_length": len(completion["text"]),
                        "generated_token_count": completion["generated_token_count"],
                        "reached_max_new_tokens": completion["generated_token_count"] >= int(variant["max_new_tokens"]),
                    }
                )
            if (prompt_index + 1) % 25 == 0:
                flush_variant_outputs(output_dir, rows, prompts, config, variant_name, variant, started_at, completed=False)
        flush_variant_outputs(output_dir, rows, prompts, config, variant_name, variant, started_at, completed=True)
    finally:
        del model
        del tokenizer
        _empty_cuda_cache()


def flush_variant_outputs(output_dir, rows, prompts, config, variant_name, variant, started_at, completed):
    write_jsonl(output_dir / "completions.jsonl", rows)
    prompt_rows = summarize_by_prompt(rows, int(config["num_generations"]))
    write_csv(output_dir / "by_prompt.csv", prompt_rows)
    summary = summarize_variant(
        rows=rows,
        prompt_rows=prompt_rows,
        prompts=prompts,
        config=config,
        variant_name=variant_name,
        variant=variant,
        elapsed_seconds=time.time() - started_at,
        completed=completed,
    )
    write_json(output_dir / "summary.json", summary)
    write_examples(output_dir / "examples.md", rows, max_examples=20)


def merge_reports(config):
    output_dir = Path(config["output_dir"])
    variant_names = parse_list(config["variant_order"])
    summaries = []
    for name in variant_names:
        summary_path = output_dir / name / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError(f"no variant summaries found under {output_dir}")

    write_csv(output_dir / "sweep_summary.csv", summaries)
    write_json(output_dir / "sweep_summary.json", {"variants": summaries, "no_training_executed": True})
    write_report(output_dir / "sweep_report.md", summaries)


def summarize_variant(rows, prompt_rows, prompts, config, variant_name, variant, elapsed_seconds, completed):
    total = len(rows)
    reward_sum = sum(float(row["reward"]) for row in rows)
    parseable = [row for row in rows if row["format_success"]]
    correct = [row for row in rows if float(row["reward"]) == 1.0]
    return {
        "variant": variant_name,
        "label": variant["label"],
        "prompt_mode": variant["prompt_mode"],
        "enable_thinking": variant["enable_thinking"],
        "max_new_tokens": variant["max_new_tokens"],
        "num_generations": int(config["num_generations"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "sample_size": len(prompts),
        "completed": completed,
        "elapsed_seconds": elapsed_seconds,
        "total_completions": total,
        "reward_mean": safe_ratio(reward_sum, total),
        "format_success_rate": safe_ratio(len(parseable), total),
        "parse_failure_rate": safe_ratio(sum(1 for row in rows if row["parse_failure"]), total),
        "correctness_given_parse": safe_ratio(len(correct), len(parseable)),
        "any_correct_prompt_rate": safe_ratio(sum(1 for row in prompt_rows if int(row["correct_count"]) > 0), len(prompt_rows)),
        "all_correct_prompt_rate": safe_ratio(
            sum(1 for row in prompt_rows if int(row["correct_count"]) == int(config["num_generations"])),
            len(prompt_rows),
        ),
        "mixed_prompt_rate": safe_ratio(sum(1 for row in prompt_rows if row["bucket"] == "mixed"), len(prompt_rows)),
        "all_zero_rate": safe_ratio(sum(1 for row in prompt_rows if row["bucket"] == "all_zero"), len(prompt_rows)),
        "avg_completion_length": mean([row["completion_length"] for row in rows]),
        "avg_generated_tokens": mean([row["generated_token_count"] for row in rows]),
        "truncation_rate": safe_ratio(sum(1 for row in rows if row["reached_max_new_tokens"]), total),
        "failure_reason_counts": dict(count_values(row["failure_reason"] or "none" for row in rows)),
        "git_commit": git_commit(),
    }


def summarize_by_prompt(rows, num_generations):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    prompt_rows = []
    for prompt_id, group in grouped.items():
        rewards = [float(row["reward"]) for row in group]
        correct_count = sum(1 for reward in rewards if reward == 1.0)
        if correct_count == 0:
            bucket = "all_zero"
        elif correct_count == num_generations:
            bucket = "all_one"
        else:
            bucket = "mixed"
        prompt_rows.append(
            {
                "id": prompt_id,
                "correct_count": correct_count,
                "reward_mean": safe_ratio(sum(rewards), len(rewards)),
                "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
                "bucket": bucket,
                "parse_failure_rate": safe_ratio(sum(1 for row in group if row["parse_failure"]), len(group)),
                "format_success_rate": safe_ratio(sum(1 for row in group if row["format_success"]), len(group)),
                "avg_completion_length": mean([row["completion_length"] for row in group]),
                "truncation_rate": safe_ratio(sum(1 for row in group if row["reached_max_new_tokens"]), len(group)),
            }
        )
    return prompt_rows


def write_report(path, summaries):
    completed = [summary for summary in summaries if summary.get("completed")]
    candidates = completed or summaries
    reasonable = [s for s in candidates if s["avg_completion_length"] <= 3000 and s["truncation_rate"] <= 0.20]
    ranked = sorted(reasonable or candidates, key=lambda s: (s["parse_failure_rate"], s["avg_completion_length"]))
    best = ranked[0]

    lines = [
        "# GSM8K Qwen3-4B Prompt Sweep",
        "",
        "This is a no-training rollout audit. It does not modify reward functions, training code, GSM8K splits, or official eval prompts.",
        "",
        f"Best parse-failure config with reasonable length: `{best['variant']}`.",
        "",
        "## Summary Table",
        "",
        "| variant | completed | max_new_tokens | thinking | reward_mean | format_success | parse_failure | correctness_given_parse | any_correct | all_correct | mixed | all_zero | avg_len | truncation |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {variant} | {completed} | {max_new_tokens} | {thinking} | {reward:.4f} | {format:.4f} | {parse:.4f} | {cgparse:.4f} | {anyc:.4f} | {allc:.4f} | {mixed:.4f} | {allzero:.4f} | {avglen:.1f} | {trunc:.4f} |".format(
                variant=summary["variant"],
                completed=summary["completed"],
                max_new_tokens=summary["max_new_tokens"],
                thinking=summary["enable_thinking"],
                reward=summary["reward_mean"],
                format=summary["format_success_rate"],
                parse=summary["parse_failure_rate"],
                cgparse=summary["correctness_given_parse"],
                anyc=summary["any_correct_prompt_rate"],
                allc=summary["all_correct_prompt_rate"],
                mixed=summary["mixed_prompt_rate"],
                allzero=summary["all_zero_rate"],
                avglen=summary["avg_completion_length"],
                trunc=summary["truncation_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Format improvement is measured by lower parse failure and higher format success.",
            "- Math accuracy improvement is measured by correctness_given_parse and reward_mean after accounting for parseability.",
            f"- The selected config is `{best['variant']}` because it has the lowest parse_failure_rate among outputs with avg length <= 3000 chars and truncation_rate <= 0.20.",
            "- If a longer-token config improves parse failure but leaves correctness_given_parse flat, the improvement is primarily formatting/completion-length, not math reasoning.",
            "- No GRPO or SFT training was started by this sweep.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_fixed_sample(path, split, sample_size, seed):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    if len(rows) < sample_size:
        raise ValueError(f"requested {sample_size} examples, found {len(rows)}")
    return random.Random(seed).sample(rows, sample_size)


def build_prompt(record, config, variant):
    current = record["prompt"][0]["content"]
    if variant["prompt_mode"] == "current":
        prompt = current
    else:
        prompt = config["strong_prompt_template"].replace("{question}", extract_question(current))
    return {
        "id": record["id"],
        "prompt": prompt,
        "answer": record["verifier"]["answer"],
        "source_prompt": current,
    }


def extract_question(prompt):
    marker = "Problem:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=bool(config["trust_remote_code"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config["torch_dtype"] == "bfloat16" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=dtype,
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def sample_completions(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    num_generations,
    generation_batch_size,
    temperature,
    top_p,
    top_k,
    enable_thinking,
):
    import torch

    prompt_text = _generation_prompt_to_text(prompt, tokenizer, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = _model_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    completions = []
    while len(completions) < num_generations:
        current_batch = min(generation_batch_size, num_generations - len(completions))
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_return_sequences": current_batch,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            if top_k > 0:
                kwargs["top_k"] = top_k
        with torch.no_grad():
            outputs = model.generate(**inputs, **kwargs)
        for output in outputs:
            generated = output[prompt_length:]
            completions.append(
                {
                    "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    "generated_token_count": int(generated.shape[-1]),
                }
            )
    return completions


def load_simple_yaml(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = parse_scalar(value.strip())
    return values


def parse_scalar(value):
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("'").strip('"') for item in value[1:-1].split(",") if item.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def parse_list(value):
    return value if isinstance(value, list) else [item.strip() for item in str(value).split(",") if item.strip()]


def count_values(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def mean(values):
    return sum(values) / len(values) if values else 0.0


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_examples(path, rows, max_examples):
    lines = ["# Sample Completions", ""]
    for row in rows[:max_examples]:
        lines.extend(
            [
                f"## {row['id']} sample {row['sample_index']}",
                "",
                f"- reward: `{row['reward']}`",
                f"- format_success: `{row['format_success']}`",
                f"- failure_reason: `{row['failure_reason']}`",
                f"- generated_token_count: `{row['generated_token_count']}`",
                f"- reached_max_new_tokens: `{row['reached_max_new_tokens']}`",
                "",
                "```text",
                row["completion"][:1200],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
