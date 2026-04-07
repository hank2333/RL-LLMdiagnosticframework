from __future__ import annotations  # Keep annotations unevaluated until runtime.

from dataclasses import asdict, dataclass, field  # Use dataclasses for structured records.
from typing import Literal  # Use Literal to constrain enumerated string values.


FaultType = Literal["exploration_collapse", "training_instability", "overly_conservative_update", "mixed_or_uncertain"]  # Restrict diagnosis fault types to the approved set.
AdjustmentAction = Literal["increase_small", "decrease_small", "keep"]  # Restrict symbolic adjustments to the approved set.
RiskLevel = Literal["low", "medium", "high"]  # Restrict diagnosis risk levels.
OutcomeLabel = Literal["accepted_improved", "accepted_neutral", "accepted_worsened", "rejected_by_guard"]  # Restrict round outcomes to the approved set.
DiagnosisMode = Literal["single_shot", "history_aware"]  # Restrict diagnosis modes.
ExperimentGroup = Literal["fixed", "single_shot", "history_aware"]  # Restrict experiment groups.

ALLOWED_FAULT_TYPES = {"exploration_collapse", "training_instability", "overly_conservative_update", "mixed_or_uncertain"}  # Provide a runtime set for fault-type validation.
ALLOWED_ADJUSTMENTS = {"increase_small", "decrease_small", "keep"}  # Provide a runtime set for adjustment validation.
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}  # Provide a runtime set for risk-level validation.


@dataclass  # Store the compact training summary as a serializable dataclass.
class TrainingSummary:  # Define the summary structure required by the project.
    avg_return: float  # Store the mean evaluation return.
    std_return: float  # Store the standard deviation of evaluation returns.
    success_rate: float  # Store the fraction of successful episodes.
    avg_episode_length: float  # Store the mean episode length.
    reward_trend: str  # Store a compact label describing return trend.
    entropy_trend: str  # Store a compact label describing entropy trend.
    policy_stability: str  # Store a compact label describing policy stability.
    action_distribution: dict[str, float]  # Store normalized action frequencies.
    dominant_action: str  # Store the most common action.
    action_collapse: bool  # Store whether the policy appears collapsed.
    alerts: list[str] = field(default_factory=list)  # Store summary alerts for downstream review.

    def to_dict(self) -> dict:  # Convert the dataclass into a plain dictionary.
        return asdict(self)  # Return the dataclass as a nested dictionary.


@dataclass  # Store the structured diagnosis as a serializable dataclass.
class DiagnosisOutput:  # Define the diagnosis structure required by the project.
    fault_type: FaultType  # Store the chosen fault type.
    observed_symptoms: list[str]  # Store the symptoms the diagnosis is based on.
    reasoning: str  # Store the explanatory reasoning text.
    proposed_adjustments: dict[str, AdjustmentAction]  # Store symbolic parameter adjustments only.
    confidence: float  # Store a bounded confidence estimate.
    risk_level: RiskLevel  # Store the risk label for the proposal.
    rollback_trigger: str  # Store a plain-language rollback condition.

    def to_dict(self) -> dict:  # Convert the dataclass into a plain dictionary.
        return asdict(self)  # Return the dataclass as a nested dictionary.


@dataclass  # Store the persisted history item as a serializable dataclass.
class HistoryRecord:  # Define the JSON history record schema.
    round_id: int  # Store the loop round number.
    config: dict[str, float]  # Store the PPO config used for the round.
    summary: dict  # Store the summary generated for the round.
    diagnosis: dict  # Store the diagnosis generated for the round.
    outcome: OutcomeLabel  # Store the coarse outcome label for the round.

    def to_dict(self) -> dict:  # Convert the history record into a dictionary.
        return asdict(self)  # Return the dataclass as a nested dictionary.
