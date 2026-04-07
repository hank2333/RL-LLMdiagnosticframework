from __future__ import annotations  # Keep annotations unevaluated until runtime.

from src.core.train_runner import PARAM_RANGES, PPOConfig  # Import the allowed parameter ranges and config type.


def apply_symbolic_adjustments(config: PPOConfig, proposed_adjustments: dict[str, str]) -> dict[str, float]:  # Convert symbolic LLM actions into concrete PPO parameter values.
    updated = config.to_dict().copy()  # Start from a mutable copy of the current config.
    for key, action in proposed_adjustments.items():  # Iterate through the symbolic adjustments.
        if key not in updated:  # Ignore any unsupported parameter names.
            continue  # Skip unknown keys.
        current = updated[key]  # Read the current parameter value.
        if action == "increase_small":  # Handle the increase action.
            updated[key] = _increase_value(key=key, current=current)  # Apply the approved increase mapping.
        elif action == "decrease_small":  # Handle the decrease action.
            updated[key] = _decrease_value(key=key, current=current)  # Apply the approved decrease mapping.
        else:  # Treat any other action as keep for robustness.
            updated[key] = current  # Preserve the current value.
        lower, upper = PARAM_RANGES[key]  # Read the allowed bounds for the parameter.
        updated[key] = min(upper, max(lower, updated[key]))  # Clip the value into the allowed range.
    return updated  # Return the proposed numeric config.


def _increase_value(key: str, current: float) -> float:  # Apply the approved small-increase mapping.
    if key in {"learning_rate", "ent_coef"}:  # Use multiplicative updates for these two parameters.
        return current * 1.5  # Multiply by 1.5 for a small increase.
    return current + 0.02  # Increase clip range additively by 0.02.


def _decrease_value(key: str, current: float) -> float:  # Apply the approved small-decrease mapping.
    if key in {"learning_rate", "ent_coef"}:  # Use multiplicative updates for these two parameters.
        return current / 1.5  # Divide by 1.5 for a small decrease.
    return current - 0.02  # Decrease clip range additively by 0.02.
