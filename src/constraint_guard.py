"""Validate proposed updates before they are applied."""


def validate_config_update(update: dict) -> dict:
    """Check whether a config update satisfies guardrail rules."""
    raise NotImplementedError("Implement constraint validation.")
