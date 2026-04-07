from __future__ import annotations  # Keep annotations unevaluated until runtime.

from pathlib import Path  # Use Path for file handling.


def load_simple_yaml(path: str | Path) -> dict:  # Load the simple project config format without external YAML dependencies.
    config_path = Path(path)  # Normalize the config path into a Path object.
    data: dict = {}  # Start with an empty config dictionary.
    current_section: str | None = None  # Track the current nested section name.
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():  # Iterate through the config file line by line.
        line_without_comment = raw_line.split("#", 1)[0].rstrip()  # Strip trailing comments and whitespace.
        if not line_without_comment.strip():  # Skip empty lines after comment removal.
            continue  # Move to the next config line.
        indent = len(raw_line) - len(raw_line.lstrip(" "))  # Count indentation to detect nesting.
        key, raw_value = line_without_comment.strip().split(":", 1)  # Split the YAML-like key/value pair once.
        value = raw_value.strip()  # Trim extra whitespace from the value fragment.
        if indent == 0 and value == "":  # Start a nested section when a top-level key has no immediate value.
            data[key] = {}  # Create an empty nested mapping for the section.
            current_section = key  # Remember the section for the following indented keys.
            continue  # Move to the next line after opening the section.
        parsed_value = _parse_scalar(value)  # Convert the scalar string into a Python value.
        if indent == 0:  # Handle a top-level scalar value.
            data[key] = parsed_value  # Store the parsed scalar at the top level.
            current_section = None  # Clear the current nested section.
        else:  # Handle a nested scalar value.
            if current_section is None:  # Reject malformed indentation without an active section.
                raise ValueError(f"Nested key '{key}' is missing a parent section in {config_path}.")  # Raise a clear config error.
            data[current_section][key] = parsed_value  # Store the parsed scalar under the active section.
    return data  # Return the fully parsed configuration dictionary.


def _parse_scalar(value: str) -> object:  # Convert a simple YAML scalar into a Python object.
    if value.lower() in {"true", "false"}:  # Handle booleans first.
        return value.lower() == "true"  # Return the parsed boolean value.
    if value.lower() in {"null", "none"}:  # Handle null-like strings.
        return None  # Return a Python null value.
    if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):  # Handle quoted strings.
        return value[1:-1]  # Strip the matching quote characters.
    try:  # Try to parse integers before floats.
        return int(value)  # Return the parsed integer when possible.
    except ValueError:  # Continue to float parsing when integer parsing fails.
        pass  # Deliberately ignore the failed integer parse.
    try:  # Try to parse a floating-point value next.
        return float(value)  # Return the parsed float when possible.
    except ValueError:  # Fall back to a raw string when numeric parsing fails.
        return value  # Return the original string value unchanged.
