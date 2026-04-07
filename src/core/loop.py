from __future__ import annotations  # Keep annotations unevaluated until runtime.

import json  # Use JSON for round-level artifact output.
from pathlib import Path  # Use Path for artifact and history locations.

from src.core.action_mapper import apply_symbolic_adjustments  # Convert symbolic LLM outputs into numeric configs.
from src.core.diagnosis_types import DiagnosisMode, DiagnosisOutput, HistoryRecord, OutcomeLabel  # Import shared structured types.
from src.core.guardrail import validate_adjustments  # Enforce update guardrails.
from src.core.history_manager import HistoryManager  # Persist recent round history.
from src.core.llm_client import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, generate_diagnosis  # Call the local Ollama diagnosis client.
from src.core.prompt_builder import build_diagnosis_payload  # Build the LLM input payload.
from src.core.summarizer import summarize_training  # Build the compact training summary.
from src.core.train_runner import PPOConfig, run_training  # Run PPO training and evaluation.


def run_closed_loop(mode: DiagnosisMode, rounds: int, history_path: str | Path, initial_config: PPOConfig | None = None, train_steps: int = 5_000, eval_episodes: int = 10, seed: int = 0, artifact_dir: str | Path = "artifacts/runs", ollama_model: str = DEFAULT_OLLAMA_MODEL, ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL, ollama_timeout_sec: float = 60.0, reset_history: bool = True) -> list[HistoryRecord]:  # Run a diagnosis-driven adaptive experiment loop.
    history_manager = HistoryManager(history_path=history_path)  # Create the rolling history manager.
    if reset_history:  # Start from a clean history file when requested.
        history_manager.clear()  # Reset the history store.
    config = initial_config or PPOConfig()  # Use the provided config or fall back to defaults.
    produced_records: list[HistoryRecord] = []  # Store the produced records for the caller.
    previous_avg_return: float | None = None  # Track the previous round's average return for outcome labeling.
    run_artifact_root = Path(artifact_dir)  # Normalize the artifact root into a Path object.
    run_artifact_root.mkdir(parents=True, exist_ok=True)  # Ensure the artifact root exists.
    for round_id in range(1, rounds + 1):  # Iterate through each training-diagnosis round.
        round_seed = seed + round_id - 1  # Derive a deterministic per-round seed from the base seed.
        round_dir = run_artifact_root / f"round_{round_id:02d}"  # Build the round-specific artifact directory.
        round_dir.mkdir(parents=True, exist_ok=True)  # Ensure the round artifact directory exists.
        artifacts = run_training(config=config, train_steps=train_steps, eval_episodes=eval_episodes, seed=round_seed, artifact_dir=round_dir)  # Train PPO and evaluate the current config.
        summary = summarize_training(artifacts.summary_inputs)  # Build the compact summary from evaluation metrics.
        prior_records = history_manager.load()  # Load the recent history before generating the next diagnosis.
        payload = build_diagnosis_payload(mode=mode, summary=summary, history_records=prior_records)  # Build the diagnosis payload.
        diagnosis = generate_diagnosis(payload=payload, model_name=ollama_model, base_url=ollama_base_url, timeout_sec=ollama_timeout_sec, use_fallback_on_error=True)  # Call the local LLM client to produce a structured diagnosis.
        proposed_config = apply_symbolic_adjustments(config=config, proposed_adjustments=diagnosis.proposed_adjustments)  # Convert symbolic adjustments into numeric updates.
        accepted_config, guard_alerts, is_valid = validate_adjustments(current_config=config.to_dict(), proposed_config=proposed_config, risk_level=diagnosis.risk_level)  # Validate the proposed numeric update.
        if guard_alerts:  # Attach guardrail alerts to the training summary for review.
            summary.alerts.extend(guard_alerts)  # Append the guard alerts to the summary alerts.
        if is_valid:  # Apply the accepted config only when guardrails pass.
            config = PPOConfig(**accepted_config)  # Promote the accepted config to the next round.
        outcome = _classify_outcome(current_avg_return=summary.avg_return, previous_avg_return=previous_avg_return, accepted=is_valid)  # Label the round outcome using acceptance and return change.
        previous_avg_return = summary.avg_return  # Update the previous average return for the next round.
        record = HistoryRecord(round_id=round_id, config=config.to_dict(), summary=summary.to_dict(), diagnosis=diagnosis.to_dict(), outcome=outcome)  # Build the persisted round record.
        history_manager.append(record)  # Persist the latest history record.
        _write_round_artifact(output_path=round_dir / "round_summary.json", record=record, raw_metrics=artifacts.raw_metrics, diagnosis=diagnosis)  # Persist a round-level JSON artifact for inspection.
        produced_records.append(record)  # Keep the record in the return list.
    return produced_records  # Return the produced round records.


def run_fixed_baseline(rounds: int, history_path: str | Path, initial_config: PPOConfig | None = None, train_steps: int = 5_000, eval_episodes: int = 10, seed: int = 0, artifact_dir: str | Path = "artifacts/runs", reset_history: bool = True) -> list[HistoryRecord]:  # Run the fixed-config baseline with no diagnosis-driven changes.
    history_manager = HistoryManager(history_path=history_path)  # Create the rolling history manager.
    if reset_history:  # Start from a clean history file when requested.
        history_manager.clear()  # Reset the history store.
    config = initial_config or PPOConfig()  # Use the provided config or fall back to defaults.
    produced_records: list[HistoryRecord] = []  # Store the produced records for the caller.
    previous_avg_return: float | None = None  # Track the previous round's average return for outcome labeling.
    run_artifact_root = Path(artifact_dir)  # Normalize the artifact root into a Path object.
    run_artifact_root.mkdir(parents=True, exist_ok=True)  # Ensure the artifact root exists.
    for round_id in range(1, rounds + 1):  # Iterate through each baseline round.
        round_seed = seed + round_id - 1  # Derive a deterministic per-round seed from the base seed.
        round_dir = run_artifact_root / f"round_{round_id:02d}"  # Build the round-specific artifact directory.
        round_dir.mkdir(parents=True, exist_ok=True)  # Ensure the round artifact directory exists.
        artifacts = run_training(config=config, train_steps=train_steps, eval_episodes=eval_episodes, seed=round_seed, artifact_dir=round_dir)  # Train PPO and evaluate the current fixed config.
        summary = summarize_training(artifacts.summary_inputs)  # Build the compact summary from evaluation metrics.
        outcome = _classify_outcome(current_avg_return=summary.avg_return, previous_avg_return=previous_avg_return, accepted=True)  # Label the baseline round relative to the previous baseline round.
        previous_avg_return = summary.avg_return  # Update the previous average return for the next round.
        record = HistoryRecord(round_id=round_id, config=config.to_dict(), summary=summary.to_dict(), diagnosis={}, outcome=outcome)  # Build the persisted round record.
        history_manager.append(record)  # Persist the latest baseline record.
        _write_round_artifact(output_path=round_dir / "round_summary.json", record=record, raw_metrics=artifacts.raw_metrics, diagnosis=None)  # Persist a round-level JSON artifact for inspection.
        produced_records.append(record)  # Keep the record in the return list.
    return produced_records  # Return the produced round records.


def _classify_outcome(current_avg_return: float, previous_avg_return: float | None, accepted: bool) -> OutcomeLabel:  # Convert acceptance and return deltas into the required outcome labels.
    if not accepted:  # Guardrail rejection overrides any return comparison.
        return "rejected_by_guard"  # Return the guard rejection label.
    if previous_avg_return is None:  # Use a neutral outcome for the first accepted round.
        return "accepted_neutral"  # Return the neutral label when no baseline comparison exists.
    if current_avg_return > previous_avg_return + 0.1:  # Treat a small positive delta as improvement.
        return "accepted_improved"  # Return the improved label.
    if current_avg_return < previous_avg_return - 0.1:  # Treat a small negative delta as worsening.
        return "accepted_worsened"  # Return the worsened label.
    return "accepted_neutral"  # Return the neutral label when the delta is small.


def _write_round_artifact(output_path: str | Path, record: HistoryRecord, raw_metrics: dict, diagnosis: DiagnosisOutput | None) -> None:  # Persist a JSON artifact that captures round-level outputs.
    payload = {"record": record.to_dict(), "raw_metrics": raw_metrics, "diagnosis": diagnosis.to_dict() if diagnosis is not None else {}}  # Build the artifact payload.
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")  # Persist the artifact JSON.
