from __future__ import annotations  # Keep annotations unevaluated until runtime.

from statistics import mean, pstdev  # Use small standard-library helpers for compact aggregation.

from src.core.diagnosis_types import TrainingSummary  # Import the structured summary type.


SAFE_ACTIONS = {"monitor", "scan"}  # Treat monitor and scan as the collapse-prone safe actions.


def summarize_training(summary_inputs: dict) -> TrainingSummary:  # Convert raw evaluation metrics into the compact summary schema.
    returns = summary_inputs["episode_returns"]  # Read per-episode returns.
    lengths = summary_inputs["episode_lengths"]  # Read per-episode lengths.
    entropies = summary_inputs["policy_entropy"]  # Read phase-level entropy values.
    policy_shift = summary_inputs["policy_shift"]  # Read phase-level policy-shift values.
    action_counts = summary_inputs["action_counts"]  # Read aggregate action counts.
    total_actions = max(1, sum(action_counts.values()))  # Prevent division by zero when normalizing action counts.
    action_distribution = {name: round(count / total_actions, 4) for name, count in action_counts.items()}  # Convert action counts into normalized frequencies.
    dominant_action = max(action_distribution, key=action_distribution.get)  # Identify the most common action.
    dominant_ratio = action_distribution[dominant_action]  # Read the dominant action's frequency.
    safe_action_ratio = round(sum(action_distribution[name] for name in SAFE_ACTIONS), 4)  # Measure monitor/scan concentration.
    mean_policy_shift = round(mean(policy_shift), 4) if policy_shift else 0.0  # Collapse phase shifts into one stability signal.
    alerts: list[str] = []  # Start with an empty alert list.
    if dominant_ratio >= 0.5:  # Flag highly concentrated action usage.
        alerts.append("Dominant action exceeds 50% of selections.")  # Record an action-concentration alert.
    if safe_action_ratio >= 0.7:  # Flag excessive reliance on the two safe actions.
        alerts.append("Monitor/scan usage exceeds 70% of actions.")  # Record a safe-action collapse alert.
    if entropies[-1] < entropies[0] * 0.75:  # Flag material entropy decline over training.
        alerts.append("Entropy declined materially during training.")  # Record an entropy-collapse alert.
    if mean_policy_shift >= 0.12:  # Flag elevated policy movement.
        alerts.append("Policy probabilities changed sharply across the episode phases.")  # Record an instability alert.
    action_collapse = bool(dominant_ratio >= 0.5 or safe_action_ratio >= 0.72)  # Decide whether the policy shows collapse-like behavior.
    policy_stability = "stable" if mean_policy_shift <= 0.12 else "unstable"  # Convert numeric shift into a compact stability label.
    return TrainingSummary(avg_return=round(mean(returns), 4), std_return=round(pstdev(returns), 4), success_rate=round(summary_inputs["successes"] / len(returns), 4), avg_episode_length=round(mean(lengths), 4), reward_trend=_trend_label(returns, higher_is_better=True), entropy_trend=_trend_label(entropies, higher_is_better=False), policy_stability=policy_stability, action_distribution=action_distribution, dominant_action=dominant_action, action_collapse=action_collapse, alerts=alerts)  # Build and return the compact project summary.


def _trend_label(values: list[float], higher_is_better: bool) -> str:  # Convert a numeric sequence into a compact trend label.
    if values[-1] > values[0] * 1.05:  # Detect a material increase from start to finish.
        return "improving" if higher_is_better else "rising"  # Use domain-specific positive wording.
    if values[-1] < values[0] * 0.95:  # Detect a material decrease from start to finish.
        return "declining" if higher_is_better else "falling"  # Use domain-specific negative wording.
    return "flat"  # Use a neutral label when change is small.
