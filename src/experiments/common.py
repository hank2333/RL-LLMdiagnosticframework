from __future__ import annotations  # Keep annotations unevaluated until runtime.

import argparse  # Use argparse for simple experiment CLI handling.
from pathlib import Path  # Use Path for filesystem-safe paths.

from src.core.config_loader import load_simple_yaml  # Load the lightweight YAML-like config files.
from src.core.llm_client import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL  # Reuse the Ollama defaults from the client.
from src.core.train_runner import PPOConfig  # Reuse the structured PPO config.


def build_parser(default_config_path: str, default_group: str) -> argparse.ArgumentParser:  # Build a shared CLI parser for the experiment entrypoints.
    parser = argparse.ArgumentParser(description=f"Run the {default_group} experiment group.")  # Create the parser with a concise description.
    parser.add_argument("--config", default=default_config_path, help="Path to the experiment config file.")  # Allow the config file to be overridden.
    parser.add_argument("--rounds", type=int, default=None, help="Override the number of outer-loop rounds.")  # Allow the round count to be overridden.
    parser.add_argument("--train-steps", type=int, default=None, help="Override PPO training timesteps per round.")  # Allow the train-step budget to be overridden.
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override evaluation episodes per round.")  # Allow the evaluation budget to be overridden.
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for reproducibility.")  # Allow the base seed to be overridden.
    parser.add_argument("--history-path", default=None, help="Override the history JSON path.")  # Allow the history path to be overridden.
    parser.add_argument("--artifact-dir", default=None, help="Override the run artifact directory.")  # Allow the artifact directory to be overridden.
    parser.add_argument("--ollama-model", default=None, help="Override the local Ollama model name.")  # Allow the model name to be overridden.
    parser.add_argument("--ollama-base-url", default=None, help="Override the local Ollama base URL.")  # Allow the base URL to be overridden.
    parser.add_argument("--ollama-timeout-sec", type=float, default=60.0, help="HTTP timeout for the local Ollama request.")  # Allow the Ollama timeout to be overridden.
    parser.add_argument("--keep-history", action="store_true", help="Keep existing history instead of clearing it.")  # Allow the run to append to existing history.
    return parser  # Return the configured parser.


def resolve_settings(args: argparse.Namespace, default_group: str) -> dict:  # Merge CLI overrides with the config file and defaults.
    config = load_simple_yaml(args.config)  # Load the lightweight YAML-like config file.
    ppo_section = config.get("ppo", {})  # Read the nested PPO config section.
    ppo_config = PPOConfig(learning_rate=float(ppo_section.get("learning_rate", 3e-4)), ent_coef=float(ppo_section.get("ent_coef", 0.01)), clip_range=float(ppo_section.get("clip_range", 0.2)), n_steps=int(ppo_section.get("n_steps", 128)), gamma=float(ppo_section.get("gamma", 0.99)))  # Build the structured PPO config from the file.
    experiment_group = str(config.get("experiment_group", default_group))  # Read the experiment group or use the default.
    return {  # Return the merged runtime settings as a plain dictionary.
        "experiment_group": experiment_group,  # Store the resolved experiment group.
        "diagnosis_mode": str(config.get("diagnosis_mode", "single_shot")),  # Store the resolved diagnosis mode.
        "rounds": args.rounds if args.rounds is not None else int(config.get("rounds", 3)),  # Resolve the round count.
        "train_steps": args.train_steps if args.train_steps is not None else int(config.get("train_steps", 5000)),  # Resolve the training budget.
        "eval_episodes": args.eval_episodes if args.eval_episodes is not None else int(config.get("eval_episodes", 10)),  # Resolve the evaluation budget.
        "seed": args.seed if args.seed is not None else int(config.get("seed", 0)),  # Resolve the base seed.
        "history_path": args.history_path or str(config.get("history_path", "artifacts/history.json")),  # Resolve the history path.
        "artifact_dir": args.artifact_dir or str(config.get("artifact_dir", str(Path("artifacts") / "runs" / experiment_group))),  # Resolve the artifact directory.
        "ollama_model": args.ollama_model or str(config.get("ollama_model", DEFAULT_OLLAMA_MODEL)),  # Resolve the Ollama model name.
        "ollama_base_url": args.ollama_base_url or str(config.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)),  # Resolve the Ollama base URL.
        "ollama_timeout_sec": float(args.ollama_timeout_sec),  # Resolve the Ollama timeout.
        "reset_history": not args.keep_history,  # Resolve whether history should be cleared before the run.
        "ppo_config": ppo_config,  # Store the resolved PPO config object.
    }  # Finish the runtime-settings dictionary.
