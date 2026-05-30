"""Sampled heldout eval for boxed-answer math models."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from posttrain_lab.eval.metrics import format_success, mean_boolean
from posttrain_lab.rewards.math_reward import MathRewardConfig, score_math_boxed_v001
from posttrain_lab.train.train_sft import _load_yaml_subset, _model_load_kwargs


PARSE_FAILURE_REASONS = {
    "malformed_boxed_answer",
    "no_boxed_answer",
    "conflicting_boxed_answers",
    "multiple_boxed_answers",
    "empty_boxed_answer",
    "boxed_not_final_only",
    "unclosed_think_block",
    "output_too_long",
}


def load_config(path):
    config = _load_yaml_subset(Path(path))
    if "prompt_path" not in config:
        raise ValueError("missing required config field: prompt_path")
    if "output_dir" not in config:
        raise ValueError("missing required config field: output_dir")
    return config


def run_sampled_eval(config):
    resolved = _resolve_config(config)
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(resolved["seed"]))
    examples = _load_examples(Path(resolved["prompt_path"]), int(resolved["selection"]["max_prompts"]))

    generator = _DryRunGenerator() if resolved["dry_run"] else _HFSampledGenerator(resolved)
    raw_rows = []
    prompt_rows = []

    try:
        for prompt_index, example in enumerate(examples):
            completions = generator.generate_many(example, resolved["inference"])
            prompt_raw_rows = []
            for sample_index, generation in enumerate(completions):
                generation = _apply_stop_tokens(generation, resolved["inference"]["stop_tokens"])
                row = _score_generation(example, generation, prompt_index, sample_index, resolved["metrics"])
                raw_rows.append(row)
                prompt_raw_rows.append(row)
            prompt_rows.append(_summarize_prompt(example, prompt_index, prompt_raw_rows))
    finally:
        if hasattr(generator, "close"):
            generator.close()

    metrics = _summarize_eval(raw_rows, prompt_rows, resolved)
    _write_jsonl(output_dir / "sampled_generations.jsonl", raw_rows)
    _write_jsonl(output_dir / "sampled_by_prompt.jsonl", prompt_rows)
    _write_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir / "eval_report.md", resolved, metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run sampled heldout eval for boxed-answer math.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    metrics = run_sampled_eval(load_config(args.config))
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _resolve_config(config):
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("dry_run", False)
    resolved.setdefault("model_name", "dummy")
    resolved.setdefault("adapter_path", None)
    resolved.setdefault("torch_dtype", "auto")
    resolved.setdefault("trust_remote_code", False)
    resolved.setdefault("attn_implementation", None)
    resolved.setdefault("apply_chat_template", True)
    resolved.setdefault("enable_thinking", None)
    resolved.setdefault("seed", 17)
    resolved.setdefault("selection", {})
    resolved["selection"].setdefault("max_prompts", 0)
    resolved.setdefault("inference", {})
    resolved["inference"].setdefault("completions_per_prompt", 4)
    resolved["inference"].setdefault("batch_size", min(4, int(resolved["inference"]["completions_per_prompt"])))
    resolved["inference"].setdefault("temperature", 0.7)
    resolved["inference"].setdefault("top_p", 0.95)
    resolved["inference"].setdefault("top_k", 0)
    resolved["inference"].setdefault("max_new_tokens", 64)
    resolved["inference"].setdefault("stop_tokens", [])
    resolved.setdefault("metrics", {})
    resolved["metrics"].setdefault("boxed_math_match", True)
    resolved["metrics"].setdefault("format_regex", r"^\\boxed\{.+\}$")
    resolved["metrics"].setdefault("allow_symbolic_equivalence", False)
    resolved["metrics"].setdefault("symbolic_equivalence_engine", "fraction")
    resolved["metrics"].setdefault("max_symbolic_expr_chars", 120)
    resolved["metrics"].setdefault("max_symbolic_ast_nodes", 64)
    resolved["metrics"].setdefault("max_symbolic_collection_size", 32)

    if int(resolved["inference"]["completions_per_prompt"]) < 1:
        raise ValueError("inference.completions_per_prompt must be >= 1")
    if int(resolved["inference"]["batch_size"]) < 1:
        raise ValueError("inference.batch_size must be >= 1")
    return resolved


class _DryRunGenerator:
    def generate_many(self, example, inference_config):
        count = int(inference_config["completions_per_prompt"])
        generations = example.get("mock_generations")
        if generations is None:
            generations = [example.get("mock_generation", "")]
        generations = [str(value) for value in generations]
        if not generations:
            generations = [""]
        return [generations[index % len(generations)] for index in range(count)]


class _HFSampledGenerator:
    def __init__(self, config):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for sampled eval") from exc

        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            trust_remote_code=bool(config["trust_remote_code"]),
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = {
            "trust_remote_code": config["trust_remote_code"],
            "torch_dtype": config["torch_dtype"],
            "attn_implementation": config.get("attn_implementation"),
        }
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"], **_model_load_kwargs(model_config))
        if config.get("adapter_path"):
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load adapter_path for sampled eval") from exc
            self.model = PeftModel.from_pretrained(self.model, config["adapter_path"])
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def generate_many(self, example, inference_config):
        prompt_text = _prompt_to_text(
            example["prompt"],
            tokenizer=self.tokenizer,
            apply_chat_template=bool(self.config["apply_chat_template"]),
            enable_thinking=self.config.get("enable_thinking"),
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = _model_device(self.model)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        count = int(inference_config["completions_per_prompt"])
        batch_size = int(inference_config["batch_size"])
        completions = []
        while len(completions) < count:
            current_batch = min(batch_size, count - len(completions))
            generation_kwargs = {
                "do_sample": float(inference_config["temperature"]) > 0.0,
                "num_return_sequences": current_batch,
                "max_new_tokens": int(inference_config["max_new_tokens"]),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = float(inference_config["temperature"])
                generation_kwargs["top_p"] = float(inference_config["top_p"])
                top_k = int(inference_config["top_k"])
                if top_k > 0:
                    generation_kwargs["top_k"] = top_k

            with self.torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            prompt_length = inputs["input_ids"].shape[-1]
            for output in outputs:
                generated = output[prompt_length:]
                completions.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip())
        return completions

    def close(self):
        del self.model
        del self.tokenizer
        try:
            import gc

            gc.collect()
        except ImportError:
            pass
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _load_examples(path, limit):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if "id" not in record:
                raise ValueError(f"{path}:{line_number}: missing required field: id")
            if "prompt" not in record:
                raise ValueError(f"{path}:{line_number}: missing required field: prompt")
            if _answer(record) is None:
                raise ValueError(f"{path}:{line_number}: missing required answer or verifier.answer")
            rows.append(record)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _score_generation(example, generation, prompt_index, sample_index, metrics_config):
    answer = _answer(example)
    reward = None
    parsed_answer = None
    failure_reason = None
    parse_failure = None
    if metrics_config["boxed_math_match"]:
        result = score_math_boxed_v001(
            generation,
            answer,
            config=MathRewardConfig(
                allow_symbolic_equivalence=bool(metrics_config["allow_symbolic_equivalence"]),
                symbolic_equivalence_engine=str(metrics_config["symbolic_equivalence_engine"]),
                max_symbolic_expr_chars=int(metrics_config["max_symbolic_expr_chars"]),
                max_symbolic_ast_nodes=int(metrics_config["max_symbolic_ast_nodes"]),
                max_symbolic_collection_size=int(metrics_config["max_symbolic_collection_size"]),
            ),
        )
        reward = result.score
        parsed_answer = result.normalized_prediction
        failure_reason = None if result.score == 1.0 else result.reason
        parse_failure = result.reason in PARSE_FAILURE_REASONS

    format_value = None
    if metrics_config.get("format_regex"):
        format_value = format_success(metrics_config["format_regex"], generation)

    return {
        "id": example["id"],
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": example["prompt"],
        "answer": answer,
        "generation": generation,
        "reward": reward,
        "answer_match": None if reward is None else reward == 1.0,
        "parsed_answer": parsed_answer,
        "failure_reason": failure_reason,
        "parse_failure": parse_failure,
        "format_success": format_value,
        "completion_length": len(generation),
    }


def _summarize_prompt(example, prompt_index, rows):
    rewards = [float(row["reward"]) for row in rows if isinstance(row["reward"], (int, float))]
    parse_failures = [row["parse_failure"] for row in rows if row["parse_failure"] is not None]
    format_values = [row["format_success"] for row in rows if row["format_success"] is not None]
    parsed_answers = [row["parsed_answer"] for row in rows if row["parsed_answer"] is not None]
    correct_count = sum(1 for reward in rewards if reward == 1.0)
    if correct_count == 0:
        bucket = "all_zero"
    elif correct_count == len(rows):
        bucket = "all_one"
    else:
        bucket = "mixed"
    return {
        "id": example["id"],
        "prompt_index": prompt_index,
        "answer": _answer(example),
        "prompt": example["prompt"],
        "sample_count": len(rows),
        "correct_count": correct_count,
        "pass": correct_count > 0,
        "all_correct": correct_count == len(rows),
        "bucket": bucket,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "reward_std": statistics.pstdev(rewards) if rewards else None,
        "parse_failure_rate": mean_boolean(parse_failures),
        "format_success_rate": mean_boolean(format_values),
        "unique_answer_count": len(set(parsed_answers)),
        "avg_completion_length": sum(row["completion_length"] for row in rows) / len(rows) if rows else None,
    }


def _summarize_eval(raw_rows, prompt_rows, config):
    rewards = [float(row["reward"]) for row in raw_rows if isinstance(row["reward"], (int, float))]
    parse_failures = [row["parse_failure"] for row in raw_rows if row["parse_failure"] is not None]
    format_values = [row["format_success"] for row in raw_rows if row["format_success"] is not None]
    parseable = [row for row in raw_rows if row["parse_failure"] is False]
    correct_parseable = [row for row in parseable if row["reward"] == 1.0]
    completions_per_prompt = int(config["inference"]["completions_per_prompt"])
    return {
        "prompt_count": len(prompt_rows),
        "completions_per_prompt": completions_per_prompt,
        "total_completions": len(raw_rows),
        "sampled_accuracy": sum(rewards) / len(rewards) if rewards else None,
        f"pass_at_{completions_per_prompt}": mean_boolean(row["pass"] for row in prompt_rows),
        "all_correct_rate": mean_boolean(row["all_correct"] for row in prompt_rows),
        "all_zero_rate": mean_boolean(row["bucket"] == "all_zero" for row in prompt_rows),
        "all_one_rate": mean_boolean(row["bucket"] == "all_one" for row in prompt_rows),
        "mixed_prompt_rate": mean_boolean(row["bucket"] == "mixed" for row in prompt_rows),
        "format_success_rate": mean_boolean(format_values),
        "parse_failure_rate": mean_boolean(parse_failures),
        "correctness_given_parse": len(correct_parseable) / len(parseable) if parseable else None,
        "avg_completion_length": (
            sum(row["completion_length"] for row in raw_rows) / len(raw_rows) if raw_rows else None
        ),
        "mean_unique_answer_count": (
            sum(row["unique_answer_count"] for row in prompt_rows) / len(prompt_rows) if prompt_rows else None
        ),
        "model_name": config["model_name"],
        "adapter_path": config.get("adapter_path"),
        "prompt_path": config["prompt_path"],
        "dry_run": bool(config["dry_run"]),
        "temperature": float(config["inference"]["temperature"]),
        "top_p": float(config["inference"]["top_p"]),
        "max_new_tokens": int(config["inference"]["max_new_tokens"]),
    }


def _answer(example):
    if "answer" in example:
        return str(example["answer"])
    verifier = example.get("verifier") or {}
    if "answer" in verifier:
        return str(verifier["answer"])
    return None


def _prompt_to_text(prompt, tokenizer=None, apply_chat_template=False, enable_thinking=None):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    if apply_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except (TypeError, ValueError):
            pass
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(message.get("content", "")) for message in prompt)
    return str(prompt)


def _model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _apply_stop_tokens(text, stop_tokens):
    result = text
    for token in stop_tokens or []:
        if token and token in result:
            result = result.split(token, 1)[0]
    return result


def _set_seed(seed):
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_report(path, config, metrics):
    lines = [
        "# Sampled Eval Report",
        "",
        f"- model: `{config.get('model_name')}`",
        f"- adapter_path: `{config.get('adapter_path')}`",
        f"- prompt_path: `{config.get('prompt_path')}`",
        f"- dry_run: `{config.get('dry_run')}`",
        f"- completions_per_prompt: `{config['inference']['completions_per_prompt']}`",
        f"- temperature: `{config['inference']['temperature']}`",
        f"- top_p: `{config['inference']['top_p']}`",
        f"- max_new_tokens: `{config['inference']['max_new_tokens']}`",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics):
        lines.append(f"- {key}: {metrics[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
