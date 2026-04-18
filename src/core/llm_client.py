from __future__ import annotations  # Keep annotations unevaluated until runtime.

import json  # Use JSON to serialize prompts and parse model responses.
from typing import Any  # Use Any for loose JSON-like payloads.

import requests  # Use requests for the local Ollama HTTP call.

from src.core.diagnosis_types import ALLOWED_ADJUSTMENTS, ALLOWED_FAULT_TYPES, ALLOWED_RISK_LEVELS, DiagnosisOutput  # Import the structured diagnosis type and runtime validation sets.


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Default to the standard local Ollama HTTP endpoint.
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"  # Default to the discovered local model name.
ALLOWED_PARAM_KEYS = {"learning_rate", "ent_coef", "clip_range", "n_steps", "gamma"}  # Restrict adjustable parameter keys to the approved set.


def generate_diagnosis(payload: dict, model_name: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout_sec: float = 60.0, use_fallback_on_error: bool = True) -> DiagnosisOutput:  # Generate a structured diagnosis using Ollama with a heuristic fallback.
    prompt = _build_prompt(payload=payload)  # Build the prompt string for the model.
    try:  # Attempt the live local Ollama call first.
        response = requests.post(f"{base_url}/api/generate", json={"model": model_name, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0}}, timeout=timeout_sec)  # Send a non-streaming generation request to Ollama.
        response.raise_for_status()  # Raise an exception for HTTP failures.
        content = response.json()["response"]  # Read the model's text response.
        parsed = json.loads(content)  # Parse the model response as JSON.
        diagnosis = _normalize_diagnosis(parsed)  # Validate and normalize the diagnosis payload.
    except Exception:  # Fall back when the local LLM call or parsing fails.
        if use_fallback_on_error:  # Respect the caller's fallback preference.
            diagnosis = _heuristic_diagnosis(payload=payload)  # Return a deterministic heuristic diagnosis.
        else:  # Re-raise the error when fallback is disabled.
            raise  # Propagate the LLM failure to the caller.
    diagnosis = _apply_summary_policy(payload=payload, diagnosis=diagnosis)  # Correct overly conservative diagnoses when the current summary already shows a clear failure mode.
    return _apply_history_aware_policy(payload=payload, diagnosis=diagnosis)  # Apply a lightweight history-aware correction when repeated ineffective advice is detected.


def _build_prompt(payload: dict) -> str:  # Build a strict JSON-only instruction prompt for the local model.
    return json.dumps({"task": "Diagnose PPO training behavior and propose symbolic parameter adjustments only.", "output_rules": {"must_be_json_object": True, "fault_type_allowed": sorted(ALLOWED_FAULT_TYPES), "adjustment_allowed": sorted(ALLOWED_ADJUSTMENTS), "parameter_keys_allowed": sorted(ALLOWED_PARAM_KEYS), "risk_levels_allowed": sorted(ALLOWED_RISK_LEVELS), "required_fields": ["fault_type", "observed_symptoms", "reasoning", "proposed_adjustments", "confidence", "risk_level", "rollback_trigger"]}, "guidance": ["Prefer exploration_collapse when action_distribution is concentrated or entropy is falling.", "Prefer training_instability when policy_stability is unstable or variance is high.", "Prefer overly_conservative_update when performance is flat with no strong failure signs.", "Use mixed_or_uncertain when signals are mixed.", "Do not output numeric parameter values.", "If mode is history_aware and recent accepted changes did not reduce collapse or low-success behavior, do not repeat the same ineffective recommendation.", "Use gamma or n_steps when persistent collapse suggests the issue is not being fixed by entropy alone."], "input": payload}, ensure_ascii=True)  # Serialize a compact meta-instruction package as plain text.


def _normalize_diagnosis(raw: dict[str, Any]) -> DiagnosisOutput:  # Validate and normalize the model output into the required schema.
    fault_type = raw.get("fault_type", "mixed_or_uncertain")  # Read the proposed fault type with a safe default.
    if fault_type not in ALLOWED_FAULT_TYPES:  # Reject unknown fault types.
        fault_type = "mixed_or_uncertain"  # Fall back to the safe mixed/uncertain label.
    risk_level = raw.get("risk_level", "medium")  # Read the proposed risk level with a safe default.
    if risk_level not in ALLOWED_RISK_LEVELS:  # Reject unknown risk levels.
        risk_level = "medium"  # Fall back to a moderate risk level.
    proposed_adjustments = {}  # Start a normalized adjustment mapping.
    for key, action in raw.get("proposed_adjustments", {}).items():  # Iterate through the model-proposed adjustments.
        if key in ALLOWED_PARAM_KEYS and action in ALLOWED_ADJUSTMENTS:  # Keep only approved parameter keys and symbolic actions.
            proposed_adjustments[key] = action  # Store the normalized symbolic adjustment.
    for required_key in ALLOWED_PARAM_KEYS:  # Ensure every tunable parameter appears in the final adjustment mapping.
        proposed_adjustments.setdefault(required_key, "keep")  # Default missing parameters to keep.
    confidence = raw.get("confidence", 0.5)  # Read the proposed confidence value.
    try:  # Normalize confidence as a bounded float.
        confidence_value = float(confidence)  # Convert the confidence to float.
    except (TypeError, ValueError):  # Handle non-numeric confidence values.
        confidence_value = 0.5  # Fall back to a neutral confidence.
    confidence_value = max(0.0, min(1.0, confidence_value))  # Clip confidence into [0, 1].
    observed_symptoms = [str(item) for item in raw.get("observed_symptoms", [])]  # Normalize the symptom list as strings.
    reasoning = str(raw.get("reasoning", "No reasoning provided."))  # Normalize the reasoning text.
    rollback_trigger = str(raw.get("rollback_trigger", "Rollback if accepted changes worsen return."))  # Normalize the rollback rule.
    return DiagnosisOutput(fault_type=fault_type, observed_symptoms=observed_symptoms, reasoning=reasoning, proposed_adjustments=proposed_adjustments, confidence=round(confidence_value, 4), risk_level=risk_level, rollback_trigger=rollback_trigger)  # Return the validated structured diagnosis object.


def _heuristic_diagnosis(payload: dict) -> DiagnosisOutput:  # Provide a deterministic fallback diagnosis when the live LLM is unavailable.
    summary = payload["current_summary"]  # Read the current summary from the diagnosis payload.
    if summary["action_collapse"]:  # Diagnose exploration collapse when the summary explicitly marks collapse.
        return DiagnosisOutput(fault_type="exploration_collapse", observed_symptoms=["Action distribution is concentrated.", "Entropy trend is falling or flat."], reasoning="The policy appears to rely on a narrow action set, especially the dominant action.", proposed_adjustments={"learning_rate": "keep", "ent_coef": "increase_small", "clip_range": "keep", "n_steps": "keep", "gamma": "keep"}, confidence=0.78, risk_level="medium", rollback_trigger="Rollback if average return decreases in the next accepted round.")  # Return the exploration-collapse diagnosis.
    if summary["policy_stability"] == "unstable" or summary["std_return"] > 3.0:  # Diagnose instability when shift or variance is high.
        return DiagnosisOutput(fault_type="training_instability", observed_symptoms=["Policy stability is unstable or returns are noisy."], reasoning="The policy appears to change too sharply relative to the small task.", proposed_adjustments={"learning_rate": "decrease_small", "ent_coef": "keep", "clip_range": "keep", "n_steps": "increase_small", "gamma": "keep"}, confidence=0.72, risk_level="medium", rollback_trigger="Rollback if success rate drops after the change.")  # Return the training-instability diagnosis using only two actual parameter changes.
    if summary["reward_trend"] == "flat" and summary["success_rate"] < 0.5:  # Diagnose overly conservative updates when learning is stagnant.
        return DiagnosisOutput(fault_type="overly_conservative_update", observed_symptoms=["Return trend is flat.", "Success rate remains modest."], reasoning="Learning appears too cautious, so a small step-up may help without overreacting.", proposed_adjustments={"learning_rate": "increase_small", "ent_coef": "keep", "clip_range": "keep", "n_steps": "keep", "gamma": "increase_small"}, confidence=0.64, risk_level="medium", rollback_trigger="Rollback if accepted changes reduce both return and success rate.")  # Return the overly-conservative-update diagnosis using only two actual parameter changes.
    return DiagnosisOutput(fault_type="mixed_or_uncertain", observed_symptoms=["No dominant failure pattern was detected."], reasoning="Signals are mixed, so the safest recommendation is to keep the current settings.", proposed_adjustments={"learning_rate": "keep", "ent_coef": "keep", "clip_range": "keep", "n_steps": "keep", "gamma": "keep"}, confidence=0.5, risk_level="low", rollback_trigger="Rollback is not needed because no parameter changes were proposed.")  # Return a safe default when no strong fault signature is found.


def _apply_summary_policy(payload: dict, diagnosis: DiagnosisOutput) -> DiagnosisOutput:  # Override passive diagnoses when the current summary already indicates a clear failure mode.
    summary = payload["current_summary"]  # Read the current summary from the diagnosis payload.
    low_success = summary.get("success_rate", 0.0) <= 0.25  # Detect low-success behavior that still needs action even without hard collapse.
    low_return = summary.get("avg_return", 0.0) <= 3.0  # Detect very weak average returns.
    falling_entropy = summary.get("entropy_trend") in {"falling", "flat"}  # Detect insufficiently exploratory behavior.
    if low_success and low_return and falling_entropy and diagnosis.fault_type == "mixed_or_uncertain":  # Replace passive advice when the model under-reacts to a clearly poor result.
        return DiagnosisOutput(fault_type="exploration_collapse", observed_symptoms=diagnosis.observed_symptoms + ["Success rate is very low while entropy is not recovering."], reasoning="Current results are too weak to justify a no-op, so the diagnosis shifts toward insufficient exploration.", proposed_adjustments={"learning_rate": "keep", "ent_coef": "increase_small", "clip_range": "keep", "n_steps": "keep", "gamma": "keep"}, confidence=max(diagnosis.confidence, 0.74), risk_level="medium", rollback_trigger="Rollback if the next accepted round still has no success improvement.")  # Push a minimal exploratory adjustment when the summary clearly indicates under-exploration.
    return diagnosis  # Return the original diagnosis when the current summary does not justify an override.


def _apply_history_aware_policy(payload: dict, diagnosis: DiagnosisOutput) -> DiagnosisOutput:  # Apply a lightweight policy that uses recent history to avoid repeating ineffective actions.
    if payload.get("mode") != "history_aware":  # Leave single-shot diagnoses unchanged.
        return diagnosis  # Return the diagnosis as-is for non-history-aware mode.
    history = payload.get("recent_history", [])  # Read the recent history records from the payload.
    if len(history) < 1:  # Require at least one prior history record before applying history-based corrections.
        return diagnosis  # Return the diagnosis unchanged when there is not enough history.
    last_record = history[-1]  # Inspect the most recent history record.
    current_summary = payload["current_summary"]  # Read the current summary for repeated-pattern detection.
    repeated_failure = last_record.get("outcome") != "accepted_improved" and current_summary.get("success_rate", 0.0) <= 0.25  # Detect a failed previous update followed by another poor round.
    repeated_entropy_push = last_record.get("diagnosis", {}).get("proposed_adjustments", {}).get("ent_coef") == "increase_small" and diagnosis.proposed_adjustments.get("ent_coef") == "increase_small"  # Detect repeated attempts to fix the issue through entropy alone.
    persistent_low_exploration = last_record.get("summary", {}).get("entropy_trend") in {"falling", "flat"} and current_summary.get("entropy_trend") in {"falling", "flat"}  # Detect repeated low-exploration behavior.
    if repeated_failure and repeated_entropy_push and persistent_low_exploration:  # Trigger a different recommendation when repeating entropy increases is not working.
        reasoning = f"{diagnosis.reasoning} Recent history shows that a prior entropy increase did not rescue performance, so the strategy now changes rollout structure instead of repeating the same step."  # Explain why the history-aware branch is changing strategy.
        return DiagnosisOutput(fault_type="exploration_collapse", observed_symptoms=diagnosis.observed_symptoms + ["Recent history shows ineffective entropy-only adjustment."], reasoning=reasoning, proposed_adjustments={"learning_rate": "keep", "ent_coef": "keep", "clip_range": "keep", "n_steps": "increase_small", "gamma": "decrease_small"}, confidence=max(diagnosis.confidence, 0.82), risk_level="medium", rollback_trigger="Rollback if success rate and average return both fail to improve in the next round.")  # Return a history-aware alternative to repeating the same ineffective adjustment.
    if len(history) >= 2:  # Allow a second history-aware branch that reacts to repeated small improvements without success.
        prev_record = history[-2]  # Inspect the second most recent record as well.
        sustained_zero_success = prev_record.get("summary", {}).get("success_rate", 0.0) == 0.0 and last_record.get("summary", {}).get("success_rate", 0.0) == 0.0 and current_summary.get("success_rate", 0.0) == 0.0  # Detect a run of zero-success rounds.
        improving_but_stalled = prev_record.get("outcome") == "accepted_improved" and last_record.get("outcome") == "accepted_improved" and current_summary.get("avg_return", 0.0) < 1.0  # Detect that returns are improving only slightly while the task remains unsolved.
        repeated_instability_tuning = last_record.get("diagnosis", {}).get("fault_type") == "training_instability" and diagnosis.fault_type == "training_instability"  # Detect repeated stabilizing advice across rounds.
        if sustained_zero_success and improving_but_stalled and repeated_instability_tuning:  # Switch strategy when stabilization improved return a little but still did not reach actual success.
            reasoning = f"{diagnosis.reasoning} History shows that stabilization helped only marginally and success remains zero, so the strategy now shifts from pure stabilization toward a more assertive update."  # Explain why the history-aware branch is changing direction.
            return DiagnosisOutput(fault_type="overly_conservative_update", observed_symptoms=diagnosis.observed_symptoms + ["Recent history shows only marginal improvement without any successful episodes."], reasoning=reasoning, proposed_adjustments={"learning_rate": "increase_small", "ent_coef": "keep", "clip_range": "keep", "n_steps": "keep", "gamma": "increase_small"}, confidence=max(diagnosis.confidence, 0.79), risk_level="medium", rollback_trigger="Rollback if the more assertive update reduces return again without improving success.")  # Return a history-aware branch that exploits the information that prior conservative tuning has stalled.
    return diagnosis  # Return the original diagnosis when the history pattern does not require correction.
