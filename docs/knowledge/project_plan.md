# Posttrain Lab Project Plan

Last updated: 2026-05-27

This is the milestone tracker for the post-training repo. Update this file whenever a milestone item changes status, and include the related commit or run path when available.

## Update Rules

- Mark an item complete only after the artifact exists in the repo or the run output exists under `runs/`.
- Do not mark training items complete from plans alone; require smoke/run evidence.
- If an item is blocked, add a short note under the milestone instead of silently leaving it unchecked.
- Keep eval, reward, data, SFT, and RLVR progress separated.

## Milestone 0: Repo Foundation

- [x] repo skeleton
- [x] `AGENTS.md`
- [x] skills
- [ ] `Makefile`
- [ ] CI smoke test

Notes:
- Repo skeleton, agent rules, and five skills are present.
- `Makefile` and CI are not present yet.

## Milestone 1: Data and Eval

- [ ] SFT schema
- [ ] RLVR schema
- [ ] validators
- [ ] leakage check
- [ ] baseline eval runner

## Milestone 2: SFT

- [ ] SFT overfit-32
- [ ] SFT smoke-1k
- [ ] eval diff
- [ ] run cards

## Milestone 3: Reward

- [ ] `math_boxed_v001`
- [ ] reward fixtures
- [ ] adversarial tests
- [ ] `reward_card`

## Milestone 4: RLVR

- [ ] GRPO toy smoke
- [ ] RLVR small run
- [ ] base/SFT/RLVR comparison
- [ ] failure taxonomy

## Milestone 5: Scale Decision

- [ ] determine whether TRL is bottleneck
- [ ] verl migration plan
- [ ] cost estimate
- [ ] larger run approval
