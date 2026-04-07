# Design Notes

## Research Question

Does a history-aware diagnosis loop produce better PPO parameter adjustments than single-shot diagnosis in a small but non-trivial discrete-action environment?

## Minimal System Loop

1. Train PPO with the current config.
2. Build a compact training summary.
3. Generate a structured diagnosis.
4. Map symbolic adjustments to numeric updates.
5. Enforce guardrails.
6. Store the round result in history.

## Current Scope

- custom Gymnasium environment
- lightweight JSON history
- mock LLM diagnosis interface
- three experiment groups
- only three tunable PPO parameters

## Out of Scope For v1

- arbitrary hyperparameter search
- complex retrieval or memory ranking
- distributed training
- production serving concerns
