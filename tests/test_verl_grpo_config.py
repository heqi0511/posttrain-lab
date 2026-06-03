from pathlib import Path


def test_verl_qwen25_dapo_smoke_config_records_paperish_settings():
    text = Path("configs/verl/qwen25_math_1_5b_dapo_grpo_smoke.sh").read_text(encoding="utf-8")

    assert "Qwen2.5-Math-1.5B" in text
    assert "data.max_prompt_length=1024" in text
    assert "data.max_response_length=2048" in text
    assert "actor_rollout_ref.actor.optim.lr=0.000002" in text
    assert "actor_rollout_ref.actor.clip_ratio=0.22" in text
    assert "actor_rollout_ref.actor.ppo_epochs=2" in text
    assert "actor_rollout_ref.rollout.name=vllm" in text
    assert "actor_rollout_ref.rollout.n=4" in text
    assert "actor_rollout_ref.rollout.temperature=0.8" in text
    assert "reward.reward_manager.name=dapo" in text
    assert "reward.custom_reward_function.name=compute_score_verl_style" in text
