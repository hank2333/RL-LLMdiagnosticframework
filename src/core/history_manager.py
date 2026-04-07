from __future__ import annotations  # Keep annotations unevaluated until runtime.

import json  # Use JSON for lightweight persistence.
from pathlib import Path  # Use Path for file operations.

from src.core.diagnosis_types import HistoryRecord  # Import the structured history-record type.


class HistoryManager:  # Manage the rolling JSON history file that stores the last three rounds.
    def __init__(self, history_path: str | Path, max_records: int = 3) -> None:  # Initialize the history manager.
        self.history_path = Path(history_path)  # Normalize the history path into a Path object.
        self.max_records = max_records  # Store the maximum number of records to retain.

    def load(self) -> list[HistoryRecord]:  # Load the persisted history records.
        if not self.history_path.exists():  # Return an empty list when the history file does not exist yet.
            return []  # There is no persisted history yet.
        payload = json.loads(self.history_path.read_text(encoding="utf-8"))  # Parse the JSON history file.
        records = payload.get("records", [])  # Read the list of record dictionaries.
        return [HistoryRecord(**record) for record in records[-self.max_records :]]  # Convert the latest records into dataclass objects.

    def append(self, record: HistoryRecord) -> None:  # Append one new record and retain only the latest records.
        records = self.load()  # Load the current history snapshot.
        records.append(record)  # Add the new record to the end.
        trimmed = records[-self.max_records :]  # Keep only the latest configured number of records.
        payload = {"records": [item.to_dict() for item in trimmed]}  # Convert the retained records back into JSON-ready dictionaries.
        self.history_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the history directory exists.
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")  # Persist the trimmed history file.

    def clear(self) -> None:  # Reset the history file to an empty record list.
        self.history_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the history directory exists.
        self.history_path.write_text(json.dumps({"records": []}, indent=2), encoding="utf-8")  # Persist an empty history payload.
