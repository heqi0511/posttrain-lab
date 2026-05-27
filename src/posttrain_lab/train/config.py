"""Lightweight training config loaders with smoke-run guardrails."""

from __future__ import annotations

from pathlib import Path


def load_sft_config(path):
    """Load an SFT config and reject unsafe smoke settings."""

    config = _load_yaml_subset(path)
    training = config.setdefault("training", {})
    _require(config, "output_dir", path)
    _require(config, "model_name_or_path", path)
    _require(training, "max_train_examples", path, "training")
    _require(training, "max_steps", path, "training")
    _require(training, "dry_run", path, "training")
    if int(training["max_train_examples"]) > 32:
        raise ValueError("SFT smoke config max_train_examples must be <= 32")
    if int(training["max_steps"]) > 10:
        raise ValueError("SFT smoke config max_steps must be <= 10")
    if training["dry_run"] is not True:
        raise ValueError("SFT smoke config must set training.dry_run: true")
    if not str(config["output_dir"]).startswith("runs/sft/"):
        raise ValueError("SFT output_dir must be under runs/sft/")
    return config


def load_rlvr_config(path):
    """Load an RLVR config and reject unsafe toy settings."""

    config = _load_yaml_subset(path)
    training = config.setdefault("training", {})
    rollout = config.setdefault("rollout", {})
    reward = config.setdefault("reward", {})
    _require(config, "output_dir", path)
    _require(config, "model_name_or_path", path)
    _require(training, "algorithm", path, "training")
    _require(training, "max_steps", path, "training")
    _require(training, "dry_run", path, "training")
    _require(rollout, "num_generations", path, "rollout")
    _require(rollout, "max_completion_length", path, "rollout")
    _require(reward, "name", path, "reward")
    if str(training["algorithm"]).lower() != "grpo":
        raise ValueError("RLVR toy config must use GRPO")
    if int(training["max_steps"]) > 10:
        raise ValueError("RLVR toy config max_steps must be <= 10")
    if training["dry_run"] is not True:
        raise ValueError("RLVR toy config must set training.dry_run: true")
    if int(rollout["num_generations"]) > 4:
        raise ValueError("RLVR toy config num_generations must be <= 4")
    if int(rollout["max_completion_length"]) > 128:
        raise ValueError("RLVR toy config max_completion_length must be <= 128")
    if reward["name"] != "math_boxed_v001":
        raise ValueError("RLVR toy config reward.name must be math_boxed_v001")
    return config


def _require(container, key, path, prefix=None):
    if key not in container:
        name = f"{prefix}.{key}" if prefix else key
        raise ValueError(f"{path}: missing required config field: {name}")


def _load_yaml_subset(path):
    path = Path(path)
    config = {}
    section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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
            raise ValueError(f"{path}: nested config entry without section: {line}")
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
