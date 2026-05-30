import json
from pathlib import Path

from posttrain_lab.train.train_sft import (
    _convert_hf_sft_record,
    _enable_generation_cache,
    _final_boxed_answer_from_example,
    _normalize_hf_messages,
    _restore_generation_cache,
    _write_validation_eval_prompts,
    load_config,
    load_sft_train_examples,
    run_sft,
)


def write_sft_jsonl(path, count, split="train", start=0):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            value = start + index
            record = {
                "id": f"{split}-{index:03d}",
                "split": split,
                "messages": [
                    {"role": "user", "content": f"Compute {value} + 1."},
                    {"role": "assistant", "content": str(value + 1)},
                ],
                "metadata": {
                    "source": "unit-test",
                    "domain": "math",
                    "difficulty": "easy",
                    "license": "synthetic",
                },
            }
            handle.write(json.dumps(record) + "\n")


def write_config(path, data_path, output_dir):
    path.write_text(
        "\n".join(
            [
                "model_name_or_path: hf-internal-testing/tiny-random-gpt2",
                f"data_path: {data_path}",
                f"output_dir: {output_dir}",
                "dry_run: true",
                "synthetic_data_if_missing: false",
                "seed: 7",
                "selection:",
                "  split: train",
                "  max_train_examples: 32",
                "training:",
                "  max_steps: 2",
                "  per_device_train_batch_size: 1",
                "  gradient_accumulation_steps: 1",
                "  learning_rate: 0.0002",
                "  max_seq_length: 128",
                "peft:",
                "  method: lora",
                "  qlora: false",
                "  r: 4",
                "  lora_alpha: 8",
                "  lora_dropout: 0.0",
                "generation_check:",
                "  max_new_tokens: 8",
                "  prompts: [Compute 1 + 1., Compute 2 + 2.]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_config_parses_overfit32_fields(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    output_dir = tmp_path / "runs" / "sft" / "overfit32"
    config_path = tmp_path / "overfit32.yaml"
    write_sft_jsonl(data_path, 32)
    write_config(config_path, data_path, output_dir)

    config = load_config(config_path)

    assert config["model_name_or_path"] == "hf-internal-testing/tiny-random-gpt2"
    assert config["data_path"] == str(data_path)
    assert config["output_dir"] == str(output_dir)
    assert config["dry_run"] is True
    assert config["selection"]["max_train_examples"] == 32
    assert config["training"]["max_steps"] == 2
    assert config["peft"]["method"] == "lora"
    assert config["peft"]["qlora"] is False
    assert config["generation_check"]["prompts"] == ["Compute 1 + 1.", "Compute 2 + 2."]


def test_load_sft_train_examples_selects_exactly_32_train_records(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    write_sft_jsonl(data_path, 40)

    examples = load_sft_train_examples(data_path, split="train", limit=32)

    assert len(examples) == 32
    assert examples[0]["id"] == "train-000"
    assert examples[-1]["id"] == "train-031"


def test_dry_run_writes_required_run_artifacts(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    output_dir = tmp_path / "runs" / "sft" / "overfit32"
    config_path = tmp_path / "overfit32.yaml"
    write_sft_jsonl(data_path, 32)
    write_config(config_path, data_path, output_dir)

    result = run_sft(load_config(config_path), config_path=config_path)

    assert result["final_loss"] == 0.0
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "run_card.md").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "sample_generations.jsonl").exists()

    run_card = (output_dir / "run_card.md").read_text(encoding="utf-8")
    assert "base model: `hf-internal-testing/tiny-random-gpt2`" in run_card
    assert f"data path: `{data_path}`" in run_card
    assert "data hash:" in run_card
    assert "config hash:" in run_card
    assert "git commit:" in run_card
    assert f"output path: `{output_dir}`" in run_card
    assert "final loss: `0.0`" in run_card

    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[-1]["final_loss"] == 0.0


def test_dry_run_supports_separate_validation_data_path(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "runs" / "sft" / "separate-val"
    config_path = tmp_path / "separate_val.yaml"
    write_sft_jsonl(train_path, 32, split="train")
    write_sft_jsonl(val_path, 4, split="val", start=100)
    write_config(config_path, train_path, output_dir)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(f"data_path: {train_path}", f"data_path: {train_path}\nvalidation_data_path: {val_path}")
    text = text.replace("  max_train_examples: 32", "  max_train_examples: 32\n  max_validation_examples: 4")
    config_path.write_text(text, encoding="utf-8")

    result = run_sft(load_config(config_path), config_path=config_path)

    assert result["final_loss"] == 0.0
    assert result["data_hash"] != result["validation_data_hash"]
    run_card = (output_dir / "run_card.md").read_text(encoding="utf-8")
    assert f"validation data path: `{val_path}`" in run_card
    assert "validation data hash:" in run_card


def test_dry_run_can_create_temporary_synthetic_data_when_enabled(tmp_path):
    data_path = tmp_path / "missing" / "sft.jsonl"
    output_dir = tmp_path / "runs" / "sft" / "overfit32"
    config_path = tmp_path / "overfit32.yaml"
    write_config(config_path, data_path, output_dir)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("synthetic_data_if_missing: false", "synthetic_data_if_missing: true"), encoding="utf-8")

    result = run_sft(load_config(config_path), config_path=config_path)

    assert data_path.exists()
    assert result["final_loss"] == 0.0
    assert len(data_path.read_text(encoding="utf-8").splitlines()) == 32


def test_synthetic_data_can_use_boxed_addition_format(tmp_path):
    data_path = tmp_path / "missing" / "boxed_sft.jsonl"
    output_dir = tmp_path / "runs" / "sft" / "boxed"
    config_path = tmp_path / "boxed.yaml"
    write_config(config_path, data_path, output_dir)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("synthetic_data_if_missing: false", "synthetic_data_if_missing: true")
    text += "synthetic_answer_format: boxed\n"
    text += "synthetic_problem_style: addition\n"
    config_path.write_text(text, encoding="utf-8")

    run_sft(load_config(config_path), config_path=config_path)

    first_record = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])
    assert "boxed format" in first_record["messages"][0]["content"]
    assert first_record["messages"][1]["content"].startswith(r"\boxed{")


def test_openr1_config_parses_dataset_and_checkpoint_fields():
    config = load_config("configs/sft/openr1_math_1k.yaml")

    assert config["dataset"]["id"] == "open-r1/Mixture-of-Thoughts"
    assert config["dataset"]["config"] == "math"
    assert config["selection"]["max_train_examples"] == 1000
    assert config["selection"]["max_validation_examples"] == 128
    assert config["training"]["save_steps"] == 100
    assert config["training"]["save_total_limit"] == 10
    assert config["dataset"]["streaming"] is True
    assert config["dataset"]["shuffle_buffer_size"] == 10000


def test_openr1_long_config_uses_reasoning_lengths_and_validation_eval():
    config = load_config("configs/sft/openr1_math_1k_len8192.yaml")

    assert config["run_name"] == "sft-openr1-math-1k-len8192"
    assert config["output_dir"] == "runs/sft/openr1_math_1k_len8192"
    assert config["training"]["max_seq_length"] == 8192
    assert config["training"]["per_device_eval_batch_size"] == 1
    assert config["generation_check"]["max_new_tokens"] == 2048
    assert config["generation_check"]["enable_thinking"] is True
    assert config["eval_after_train"]["prompt_source"] == "validation"
    assert config["eval_after_train"]["sample_size"] == 8
    assert config["eval_after_train"]["max_new_tokens"] == 2048
    assert config["eval_after_train"]["enable_thinking"] is True
    assert config["eval_after_train"]["boxed_math_match"] is True
    assert config["eval_after_train"]["exact_match"] is False
    assert config["eval_after_train"]["format_regex"] is None


def test_openr1_format_repair_config_uses_short_boxed_targets_and_parent_adapter():
    config = load_config("configs/sft/openr1_math_format_repair_1k.yaml")

    assert config["run_name"] == "sft-openr1-math-format-repair-1k"
    assert config["parent_adapter_path"].endswith("/runs/sft/openr1_math_1k_len8192/checkpoint-1000")
    assert config["dataset"]["target_format"] == "boxed_final_only"
    assert config["training"]["max_seq_length"] == 2048
    assert config["training"]["max_steps"] == 500
    assert config["generation_check"]["max_new_tokens"] == 256
    assert config["generation_check"]["enable_thinking"] is False
    assert config["eval_after_train"]["prompt_source"] == "validation"
    assert config["eval_after_train"]["sample_size"] == 16
    assert config["eval_after_train"]["max_new_tokens"] == 256
    assert config["eval_after_train"]["enable_thinking"] is False
    assert config["eval_after_train"]["boxed_math_match"] is True
    assert config["eval_after_train"]["exact_match"] is False
    assert config["eval_after_train"]["format_regex"] is None


def test_qwen3_4b_openr1_format_repair_config_is_small_and_boxed_only():
    config = load_config("configs/sft/qwen3_4b_openr1_format_repair_tiny.yaml")

    assert config["run_name"] == "sft-qwen3-4b-openr1-format-repair-tiny"
    assert config["model_name_or_path"] == "Qwen/Qwen3-4B"
    assert config["dataset"]["id"] == "open-r1/Mixture-of-Thoughts"
    assert config["dataset"]["target_format"] == "boxed_final_only"
    assert config["selection"]["max_train_examples"] == 512
    assert config["selection"]["max_validation_examples"] == 128
    assert config["training"]["max_steps"] == 300
    assert config["training"]["max_seq_length"] == 2048
    assert config["training"]["eval_steps"] == 50
    assert config["training"]["save_steps"] == 50
    assert config["generation_check"]["max_new_tokens"] == 128
    assert config["generation_check"]["enable_thinking"] is False
    assert config["eval_after_train"]["sample_size"] == 16
    assert config["eval_after_train"]["max_new_tokens"] == 128
    assert config["eval_after_train"]["enable_thinking"] is False
    assert config["eval_after_train"]["boxed_math_match"] is True
    assert config["eval_after_train"]["exact_match"] is False
    assert config["eval_after_train"]["format_regex"] is None


def test_qwen3_4b_sympy_boxed_configs_use_staged_data_and_sympy_eval():
    smoke = load_config("configs/sft/qwen3_4b_sympy_boxed_smoke.yaml")
    full = load_config("configs/sft/qwen3_4b_sympy_boxed_full.yaml")

    assert smoke["model_name_or_path"] == "Qwen/Qwen3-4B"
    assert smoke["data_path"] == "data/staged/openr1_deepmath_sympy_boxed_v1/train.jsonl"
    assert smoke["validation_data_path"] == "data/staged/openr1_deepmath_sympy_boxed_v1/val.jsonl"
    assert smoke["selection"]["max_train_examples"] == 256
    assert smoke["selection"]["max_validation_examples"] == 64
    assert smoke["smoke_run"] is True
    assert smoke["eval_after_train"]["allow_symbolic_equivalence"] is True
    assert smoke["eval_after_train"]["symbolic_equivalence_engine"] == "sympy"

    assert full["selection"]["max_train_examples"] == 5000
    assert full["selection"]["max_validation_examples"] == 500
    assert full["training"]["max_steps"] == 1800
    assert full["training"]["gradient_accumulation_steps"] == 4
    assert full["smoke_run"] is False
    assert full["eval_after_train"]["allow_symbolic_equivalence"] is True
    assert full["eval_after_train"]["symbolic_equivalence_engine"] == "sympy"


def test_qwen3_4b_openr1_cn_math_config_uses_filtered_staged_data_and_early_stop():
    config = load_config("configs/sft/qwen3_4b_openr1_cn_math_sft.yaml")

    assert config["model_name_or_path"] == "Qwen/Qwen3-4B"
    assert config["data_path"] == "data/staged/openr1_cn_math_alg_nt_sft_v1/train.jsonl"
    assert config["validation_data_path"] == "data/staged/openr1_cn_math_alg_nt_sft_v1/val.jsonl"
    assert config["selection"]["max_train_examples"] == 18500
    assert config["selection"]["max_validation_examples"] == 2300
    assert config["training"]["max_steps"] == 6000
    assert config["training"]["eval_steps"] == 250
    assert config["training"]["save_steps"] == 250
    assert config["training"]["early_stopping"] is True
    assert config["training"]["early_stopping_patience"] == 5
    assert config["training"]["load_best_model_at_end"] is True
    assert config["eval_after_train"]["allow_symbolic_equivalence"] is True
    assert config["eval_after_train"]["symbolic_equivalence_engine"] == "sympy"


def test_validation_eval_prompt_writer_extracts_final_boxed_answers(tmp_path):
    examples = [
        {
            "id": "val-1",
            "messages": [
                {"role": "user", "content": "Compute 2 + 2."},
                {
                    "role": "assistant",
                    "content": "<think>Try 5. Wrong: \\boxed{5}</think>\nFinal: \\boxed{4}",
                },
            ],
        },
        {
            "id": "val-2",
            "messages": [
                {"role": "user", "content": "Compute 3 + 3."},
                {"role": "assistant", "content": "Final: \\boxed{6}"},
            ],
        },
    ]

    assert _final_boxed_answer_from_example(examples[0]) == "4"

    path = tmp_path / "eval_prompts.jsonl"
    written = _write_validation_eval_prompts(path, examples, sample_size=1)

    assert written == str(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "val-1", "prompt": "Compute 2 + 2.", "answer": "4"}]


def test_generation_cache_is_temporarily_enabled_and_restored():
    class Config:
        def __init__(self, use_cache):
            self.use_cache = use_cache

    class Model:
        def __init__(self):
            self.config = Config(False)
            self.generation_config = Config(False)

    model = Model()

    state = _enable_generation_cache(model)

    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True

    _restore_generation_cache(model, state)

    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False


def test_hf_message_normalization_and_openr1_conversion():
    record = {
        "messages": [
            {"role": "user", "content": "Solve 2 + 2."},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        "source": "open-r1/OpenR1-Math-220k",
    }
    dataset_config = {
        "id": "open-r1/Mixture-of-Thoughts",
        "config": "math",
        "messages_field": "messages",
        "source_field": "source",
        "domain": "math",
        "difficulty": "mixed",
        "license": "source-dataset-card",
    }

    assert _normalize_hf_messages(record["messages"]) == record["messages"]
    converted = _convert_hf_sft_record(record, dataset_config, "train", 0, 7)

    assert converted["id"] == "openr1-math-train-000000"
    assert converted["split"] == "train"
    assert converted["messages"][1]["role"] == "assistant"
    assert converted["metadata"]["source"] == "open-r1/OpenR1-Math-220k"


def test_openr1_conversion_can_rewrite_reasoning_target_to_final_boxed_only():
    record = {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve 19 + 23."},
            {
                "role": "assistant",
                "content": "<think>Do not train on this boxed scratch: \\boxed{41}</think>\nFinal answer: \\boxed{42}",
            },
        ],
        "source": "open-r1/OpenR1-Math-220k",
    }
    dataset_config = {
        "id": "open-r1/Mixture-of-Thoughts",
        "config": "math",
        "messages_field": "messages",
        "source_field": "source",
        "domain": "math",
        "difficulty": "mixed",
        "license": "source-dataset-card",
        "target_format": "boxed_final_only",
    }

    converted = _convert_hf_sft_record(record, dataset_config, "train", 0, 7)

    assert converted["messages"] == [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "Solve 19 + 23."},
        {"role": "assistant", "content": r"\boxed{42}"},
    ]
