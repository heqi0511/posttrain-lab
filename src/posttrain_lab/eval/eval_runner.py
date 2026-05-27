"""Minimal eval runner for dry-run and Hugging Face causal LM generation."""

import argparse
import json
from pathlib import Path

from posttrain_lab.eval.metrics import exact_match, format_success, mean_boolean


def run_eval(config):
    prompt_path = Path(config["prompt_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_jsonl(prompt_path)
    inference_config = config.get("inference", {})
    metrics_config = config.get("metrics", {})
    stop_tokens = inference_config.get("stop_tokens") or []

    generator = _DryRunGenerator() if config.get("dry_run") else _HFCausalLMGenerator(config)
    rows = []
    exact_values = []
    format_values = []

    for example in examples:
        generation = generator.generate(example, inference_config)
        generation = _apply_stop_tokens(generation, stop_tokens)
        answer = str(example.get("answer", ""))

        exact_value = None
        if metrics_config.get("exact_match") and "answer" in example:
            exact_value = exact_match(generation, answer)
            exact_values.append(exact_value)

        format_value = None
        format_regex = metrics_config.get("format_regex")
        if format_regex:
            format_value = format_success(format_regex, generation)
            format_values.append(format_value)

        rows.append(
            {
                "id": example.get("id"),
                "prompt": example.get("prompt"),
                "answer": example.get("answer"),
                "generation": generation,
                "exact_match": exact_value,
                "format_success": format_value,
                "completion_length": len(generation),
            }
        )

    metrics = {
        "count": len(rows),
        "exact_match": mean_boolean(exact_values),
        "format_success": mean_boolean(format_values),
        "parse_failure_rate": _parse_failure_rate(rows, bool(metrics_config.get("format_regex"))),
        "completion_length_mean": _completion_length_mean(rows),
    }

    _write_jsonl(output_dir / "raw_generations.jsonl", rows)
    _write_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir / "eval_report.md", config, metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a local eval from a config file.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metrics = run_eval(config)
    print(json.dumps(metrics, sort_keys=True))
    return 0


def load_config(path):
    """Load a tiny YAML subset used by configs/eval/baseline.yaml."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
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


class _DryRunGenerator:
    def generate(self, example, inference_config):
        if "mock_generation" in example:
            return str(example["mock_generation"])
        return str(example.get("prompt", ""))


class _HFCausalLMGenerator:
    def __init__(self, config):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for non-dry-run evals") from exc

        model_name = config["model_name"]
        self.torch = torch
        self.apply_chat_template = bool(config.get("apply_chat_template", False))
        self.enable_thinking = config.get("enable_thinking")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
        dtype = _torch_dtype(config.get("torch_dtype"))
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        adapter_path = config.get("adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load adapter_path for evals") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def generate(self, example, inference_config):
        prompt = _prompt_to_text(
            example.get("prompt", ""),
            tokenizer=self.tokenizer,
            apply_chat_template=self.apply_chat_template,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = inference_config.get("temperature", 0.0) > 0
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": int(inference_config.get("max_new_tokens", 128)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = inference_config.get("temperature", 1.0)
            generation_kwargs["top_p"] = inference_config.get("top_p", 1.0)
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _load_jsonl(path):
    records = []
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
            records.append(record)
    return records


def _prompt_to_text(prompt, tokenizer=None, apply_chat_template=False, enable_thinking=None):
    if apply_chat_template and tokenizer is not None:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
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


def _apply_stop_tokens(text, stop_tokens):
    result = text
    for token in stop_tokens:
        if token and token in result:
            result = result.split(token, 1)[0]
    return result


def _parse_failure_rate(rows, has_format_metric):
    if not rows or not has_format_metric:
        return None
    failures = sum(1 for row in rows if row["format_success"] is False)
    return failures / len(rows)


def _completion_length_mean(rows):
    if not rows:
        return None
    return sum(row["completion_length"] for row in rows) / len(rows)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path, config, metrics):
    lines = [
        "# Eval Report",
        "",
        f"- model: `{config.get('model_name')}`",
        f"- dry_run: `{config.get('dry_run')}`",
        f"- prompt_path: `{config.get('prompt_path')}`",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics):
        lines.append(f"- {key}: {metrics[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
