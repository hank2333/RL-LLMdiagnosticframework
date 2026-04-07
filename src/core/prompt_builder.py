from __future__ import annotations  # Keep annotations unevaluated until runtime.

from src.core.diagnosis_types import DiagnosisMode, HistoryRecord, TrainingSummary  # Import the shared structured types.


def build_diagnosis_payload(mode: DiagnosisMode, summary: TrainingSummary, history_records: list[HistoryRecord]) -> dict:  # Build the structured payload passed to the LLM client.
    payload = {"mode": mode, "current_summary": summary.to_dict()}  # Start the payload with fields common to both modes.
    if mode == "history_aware":  # Attach the last three records only for the history-aware mode.
        payload["recent_history"] = [record.to_dict() for record in history_records[-3:]]  # Serialize the recent history records.
    return payload  # Return the finished diagnosis payload.
