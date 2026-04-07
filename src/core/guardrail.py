from __future__ import annotations  # Keep annotations unevaluated until runtime.

from src.core.train_runner import PARAM_RANGES  # Import the project-approved parameter ranges.


ALLOWED_KEYS = set(PARAM_RANGES.keys())  # Restrict updates to the approved PPO parameter names.


def validate_adjustments(current_config: dict[str, float], proposed_config: dict[str, float], risk_level: str) -> tuple[dict[str, float], list[str], bool]:  # Validate, clip, and optionally reject a proposed parameter update.
    accepted = current_config.copy()  # Start from the current config so rejected changes preserve the status quo.
    alerts: list[str] = []  # Start with an empty list of guardrail alerts.
    extra_keys = [key for key in proposed_config if key not in ALLOWED_KEYS]  # Collect unsupported parameter names.
    if extra_keys:  # Reject the whole proposal when unsupported parameters are present.
        alerts.append(f"Rejected unsupported parameter keys: {sorted(extra_keys)}.")  # Record a clear rejection reason.
        return accepted, alerts, False  # Reject the proposal immediately.
    changed_keys = [key for key in ALLOWED_KEYS if key in proposed_config and proposed_config[key] != current_config.get(key)]  # Collect the approved parameters that actually change value this round.
    max_changes = 1 if risk_level == "high" else 2  # Tighten the change count when diagnosis risk is high.
    if len(changed_keys) > max_changes:  # Reject proposals that change too many parameters in one round.
        alerts.append(f"Too many parameter changes proposed: {len(changed_keys)} > {max_changes}.")  # Record the rejection reason.
        return accepted, alerts, False  # Reject the proposal immediately.
    for key, value in proposed_config.items():  # Validate each proposed parameter value.
        lower, upper = PARAM_RANGES[key]  # Read the allowed lower and upper bounds.
        clipped = min(upper, max(lower, value))  # Clip the proposed value into the allowed range.
        if clipped != value:  # Detect when clipping was needed.
            alerts.append(f"Clipped {key} into the allowed range [{lower}, {upper}].")  # Record the clipping event.
        accepted[key] = clipped  # Store the validated or clipped value.
    return accepted, alerts, True  # Accept the validated proposal.
