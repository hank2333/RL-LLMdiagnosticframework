# RL-LLM Diagnostic Framework

Minimal closed-loop PPO diagnosis framework for testing whether history-aware LLM diagnosis outperforms single-shot diagnosis in a lightweight discrete-action environment.

## Project Goal

Build a small research-oriented codebase that can:

- train PPO on a custom discrete-action environment
- summarize policy behavior compactly
- generate structured diagnoses in `single_shot` and `history_aware` modes
- apply guarded parameter updates over multiple rounds
- compare against a `fixed` baseline

The implementation is intentionally minimal. Architecture and testability take priority over completeness.

## Project Tree

```text
project/
  README.md
  configs/
    fixed.yaml
    single_shot.yaml
    history_aware.yaml
  artifacts/
    history.json
    runs/
  docs/
    design_notes.md
  src/
    envs/
      __init__.py
      mini_defense_env.py
    core/
      __init__.py
      action_mapper.py
      diagnosis_types.py
      guardrail.py
      history_manager.py
      llm_client.py
      loop.py
      prompt_builder.py
      summarizer.py
      train_runner.py
    experiments/
      __init__.py
      run_fixed.py
      run_history_aware.py
      run_single_shot.py
```

## Core Module Responsibilities

- `src/envs/mini_defense_env.py`
  Defines `MiniDefenseEnv`, a fast custom Gymnasium environment with 6 discrete actions, short episodes, low-dimensional observations, and reward dynamics that can induce action collapse.
- `src/core/train_runner.py`
  Holds PPO training configuration, parameter bounds, and the training entrypoint abstraction.
- `src/core/summarizer.py`
  Builds the compact training summary used by the diagnosis step.
- `src/core/diagnosis_types.py`
  Defines structured diagnosis schemas, allowed fault types, and outcome labels.
- `src/core/history_manager.py`
  Stores and retrieves the last 3 round records in JSON format.
- `src/core/prompt_builder.py`
  Assembles inputs for `single_shot` and `history_aware` diagnosis modes.
- `src/core/llm_client.py`
  Provides a mock LLM interface that returns structured diagnosis payloads.
- `src/core/action_mapper.py`
  Converts symbolic adjustment actions like `increase_small` into bounded numeric PPO updates.
- `src/core/guardrail.py`
  Enforces allowed keys, parameter ranges, and per-round change limits.
- `src/core/loop.py`
  Coordinates train -> summarize -> diagnose -> guard -> record history.
- `src/experiments/run_fixed.py`
  Runs the baseline with no adaptive updates.
- `src/experiments/run_single_shot.py`
  Runs closed-loop diagnosis using only the current summary.
- `src/experiments/run_history_aware.py`
  Runs closed-loop diagnosis using the current summary plus recent history.

## Initial Implementation Notes

- Environment:
  `MiniDefenseEnv` exposes 6 actions:
  `monitor`, `scan`, `isolate_host`, `patch`, `deploy_decoy`, `do_nothing`
- PPO tunable parameters:
  `learning_rate`, `ent_coef`, `clip_range`
- Diagnosis modes:
  `single_shot`, `history_aware`
- Experiment groups:
  `fixed`, `single_shot`, `history_aware`
- History length:
  3 rounds

## Short Implementation Plan

1. Finalize the custom environment and verify that PPO can train on it quickly.
2. Connect `train_runner.py` to Stable-Baselines3 PPO and collect rollout metrics.
3. Implement deterministic summary generation from evaluation traces and policy stats.
4. Improve the mock diagnosis policy, then swap in a real LLM client behind the same interface.
5. Run side-by-side experiments for `fixed`, `single_shot`, and `history_aware`.

## Status

Current state: project skeleton and core interfaces are in place. PPO integration and end-to-end experiment execution are the next implementation steps.

