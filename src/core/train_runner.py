from __future__ import annotations  # Keep annotations unevaluated until runtime.

from dataclasses import asdict, dataclass  # Use dataclasses for config and artifact records.
from pathlib import Path  # Use Path for artifact output locations.

import numpy as np  # Use NumPy to aggregate evaluation statistics.
from stable_baselines3 import PPO  # Use Stable-Baselines3 PPO for lightweight training.
from stable_baselines3.common.monitor import Monitor  # Wrap environments to standardize episode bookkeeping.

from src.envs.mini_defense_env import ACTION_NAMES, MiniDefenseEnv  # Import the custom environment and action labels.


PARAM_RANGES = {"learning_rate": (1e-5, 1e-3), "ent_coef": (0.0, 0.05), "clip_range": (0.1, 0.3)}  # Constrain the only tunable PPO parameters.


@dataclass  # Store PPO hyperparameters as a serializable dataclass.
class PPOConfig:  # Define the only three tunable PPO parameters.
    learning_rate: float = 3e-4  # Default the learning rate to the requested baseline.
    ent_coef: float = 0.01  # Default the entropy coefficient to the requested baseline.
    clip_range: float = 0.2  # Default the clip range to the requested baseline.

    def to_dict(self) -> dict[str, float]:  # Convert the config into a dictionary.
        return asdict(self)  # Return the dataclass as a plain mapping.


@dataclass  # Store the outputs needed by the rest of the loop.
class TrainingArtifacts:  # Define the training artifact bundle.
    summary_inputs: dict  # Store inputs needed to build the compact summary.
    raw_metrics: dict  # Store extra metrics and artifact paths for inspection.
    config: dict[str, float]  # Store the config actually used for this round.


def run_training(config: PPOConfig, train_steps: int = 5_000, eval_episodes: int = 10, seed: int = 0, artifact_dir: str | Path | None = None) -> TrainingArtifacts:  # Train PPO and collect evaluation metrics for one round.
    train_env = Monitor(MiniDefenseEnv(seed=seed))  # Build the training environment with monitoring.
    model = PPO(policy="MlpPolicy", env=train_env, learning_rate=config.learning_rate, ent_coef=config.ent_coef, clip_range=config.clip_range, n_steps=128, batch_size=64, gamma=0.99, seed=seed, verbose=0)  # Create a small PPO learner suitable for the lightweight environment.
    model.learn(total_timesteps=train_steps, progress_bar=False)  # Train PPO for the requested number of timesteps.
    summary_inputs = _evaluate_model(model=model, eval_episodes=eval_episodes, base_seed=seed + 10_000)  # Run lightweight evaluation after training.
    saved_model_path = _save_model_if_requested(model=model, artifact_dir=artifact_dir)  # Persist the model when an artifact directory is provided.
    train_env.close()  # Close the training environment to release resources.
    return TrainingArtifacts(summary_inputs=summary_inputs, raw_metrics={"train_steps": train_steps, "eval_episodes": eval_episodes, "seed": seed, "model_path": saved_model_path}, config=config.to_dict())  # Return the structured training result bundle.


def _evaluate_model(model: PPO, eval_episodes: int, base_seed: int) -> dict:  # Evaluate the trained policy and extract summary inputs.
    episode_returns: list[float] = []  # Store per-episode returns.
    episode_lengths: list[int] = []  # Store per-episode lengths.
    action_counts = {name: 0 for name in ACTION_NAMES}  # Count selected actions across evaluation.
    phase_entropies: list[list[float]] = [[], [], []]  # Store policy entropy samples for early, middle, and late phases.
    phase_probabilities: list[list[np.ndarray]] = [[], [], []]  # Store action-probability vectors for policy-shift estimates.
    successes = 0  # Count successful evaluation episodes.
    for episode_index in range(eval_episodes):  # Evaluate over the requested number of episodes.
        env = MiniDefenseEnv(seed=base_seed + episode_index)  # Build a fresh environment for this evaluation episode.
        obs, _ = env.reset(seed=base_seed + episode_index)  # Reset the environment with a deterministic episode seed.
        done = False  # Track episode completion.
        episode_return = 0.0  # Accumulate reward for the current episode.
        episode_length = 0  # Count steps for the current episode.
        final_info: dict = {}  # Preserve the last step info for success detection.
        while not done:  # Continue until the episode terminates or truncates.
            action_probs, entropy_value = _extract_policy_stats(model=model, obs=obs)  # Inspect the current policy distribution.
            phase_index = min(2, int((episode_length / env.max_steps) * 3))  # Bucket the current step into one of three episode phases.
            phase_entropies[phase_index].append(entropy_value)  # Record entropy for the current phase.
            phase_probabilities[phase_index].append(action_probs)  # Record action probabilities for shift estimation.
            action, _ = model.predict(obs, deterministic=False)  # Sample an action stochastically to preserve exploration patterns.
            action_int = int(action)  # Normalize the action type to a plain Python integer.
            action_counts[ACTION_NAMES[action_int]] += 1  # Count the selected action.
            obs, reward, terminated, truncated, final_info = env.step(action_int)  # Advance the environment.
            episode_return += float(reward)  # Accumulate episode reward.
            episode_length += 1  # Increase the current episode length.
            done = terminated or truncated  # Stop when the episode ends for either reason.
        if final_info.get("is_success", False):  # Count the episode when the environment marks it as successful.
            successes += 1  # Increase the success counter.
        episode_returns.append(round(episode_return, 4))  # Store the episode return with compact rounding.
        episode_lengths.append(episode_length)  # Store the episode length.
        env.close()  # Close the evaluation environment before the next episode.
    return {"episode_returns": episode_returns, "episode_lengths": episode_lengths, "policy_entropy": [_mean_or_zero(values) for values in phase_entropies], "policy_shift": _compute_policy_shift(phase_probabilities), "action_counts": action_counts, "successes": successes}  # Return the summary inputs expected by the summarizer.


def _extract_policy_stats(model: PPO, obs: np.ndarray) -> tuple[np.ndarray, float]:  # Inspect action probabilities and entropy for one observation.
    obs_tensor, _ = model.policy.obs_to_tensor(obs)  # Convert the observation into the policy's tensor format.
    distribution = model.policy.get_distribution(obs_tensor)  # Build the policy distribution for the observation.
    probs = distribution.distribution.probs.detach().cpu().numpy()[0]  # Read the categorical action probabilities as a NumPy vector.
    entropy = float(distribution.distribution.entropy().mean().item())  # Read the scalar policy entropy.
    return probs.astype(np.float64), entropy  # Return the action probabilities and entropy value.


def _compute_policy_shift(phase_probabilities: list[list[np.ndarray]]) -> list[float]:  # Estimate policy change between early, middle, and late phases.
    mean_phase_probs = [np.mean(phase_values, axis=0) if phase_values else np.zeros(len(ACTION_NAMES), dtype=np.float64) for phase_values in phase_probabilities]  # Compute a representative mean probability vector for each phase.
    return [round(float(np.mean(np.abs(mean_phase_probs[0] - mean_phase_probs[1]))), 4), round(float(np.mean(np.abs(mean_phase_probs[1] - mean_phase_probs[2]))), 4), round(float(np.mean(np.abs(mean_phase_probs[0] - mean_phase_probs[2]))), 4)]  # Return three coarse shift values for the summarizer.


def _save_model_if_requested(model: PPO, artifact_dir: str | Path | None) -> str:  # Save the trained model when a round artifact directory is provided.
    if artifact_dir is None:  # Skip saving when the caller does not request artifact persistence.
        return ""  # Return an empty path when nothing is saved.
    model_dir = Path(artifact_dir)  # Normalize the artifact directory into a Path object.
    model_dir.mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists.
    model_path = model_dir / "ppo_model"  # Build the Stable-Baselines3 save path prefix.
    model.save(str(model_path))  # Save the trained PPO model.
    return str(model_path)  # Return the saved-model path as a string.


def _mean_or_zero(values: list[float]) -> float:  # Compute a mean while tolerating empty inputs.
    if not values:  # Handle empty phase buckets safely.
        return 0.0  # Return zero when there are no values.
    return round(float(np.mean(values)), 4)  # Return the rounded mean value.
