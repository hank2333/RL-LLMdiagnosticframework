from __future__ import annotations  # Keep annotations unevaluated until runtime.

from src.core.loop import run_closed_loop  # Import the adaptive closed-loop entrypoint.
from src.experiments.common import build_parser, resolve_settings  # Reuse the shared experiment CLI helpers.


if __name__ == "__main__":  # Run the single-shot experiment when executed as a module.
    parser = build_parser(default_config_path="configs/single_shot.yaml", default_group="single_shot")  # Build the shared parser with single-shot defaults.
    args = parser.parse_args()  # Parse command-line arguments.
    settings = resolve_settings(args=args, default_group="single_shot")  # Merge CLI overrides with config defaults.
    run_closed_loop(mode="single_shot", rounds=settings["rounds"], history_path=settings["history_path"], initial_config=settings["ppo_config"], train_steps=settings["train_steps"], eval_episodes=settings["eval_episodes"], seed=settings["seed"], artifact_dir=settings["artifact_dir"], ollama_model=settings["ollama_model"], ollama_base_url=settings["ollama_base_url"], ollama_timeout_sec=settings["ollama_timeout_sec"], reset_history=settings["reset_history"])  # Execute the single-shot adaptive loop.
