#!/usr/bin/env python3
"""Check greedy recall on SFT JSONL examples for an adapter checkpoint."""

import argparse
import json
import re
from pathlib import Path


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT train-set recall for a PEFT adapter.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype="auto")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    records = _load_sft_records(Path(args.data_path), args.split)
    rows = []
    exact = 0
    prefix = 0
    answer = 0
    for record in records:
        prompt = _first_message(record["messages"], "user")
        target = _first_message(record["messages"], "assistant").strip()
        prompt_text = _generation_prompt(prompt, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        normalized_generation = _normalize_answer(generated)
        normalized_target = _normalize_answer(target)
        exact_match = generated == target
        prefix_match = generated.startswith(target)
        answer_match = normalized_generation == normalized_target
        exact += int(exact_match)
        prefix += int(prefix_match)
        answer += int(answer_match)
        rows.append(
            {
                "id": record["id"],
                "prompt": prompt,
                "target": target,
                "generation": generated,
                "normalized_generation": normalized_generation,
                "normalized_target": normalized_target,
                "exact_match": exact_match,
                "prefix_match": prefix_match,
                "answer_match": answer_match,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "count": len(rows),
        "answer": answer,
        "answer_rate": answer / len(rows) if rows else 0.0,
        "exact": exact,
        "exact_rate": exact / len(rows) if rows else 0.0,
        "prefix": prefix,
        "prefix_rate": prefix / len(rows) if rows else 0.0,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True))


def _load_sft_records(path, split):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == split:
                records.append(record)
    return records


def _first_message(messages, role):
    for message in messages:
        if message["role"] == role:
            return message["content"]
    raise ValueError(f"missing {role} message")


def _generation_prompt(prompt, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            pass
    return prompt


def _normalize_answer(text):
    without_think = THINK_BLOCK_RE.sub("", text)
    return " ".join(without_think.strip().split())


if __name__ == "__main__":
    main()
