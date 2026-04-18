from __future__ import annotations  # Keep annotations unevaluated until runtime.

from dataclasses import dataclass  # Use a dataclass for the observation snapshot helper.

import gymnasium as gym  # Import Gymnasium for the custom environment base class.
import numpy as np  # Import NumPy for numeric state handling.
from gymnasium import spaces  # Import Gymnasium spaces for action and observation specs.


ACTION_NAMES = (  # Define the fixed discrete action names required by the project.
    "monitor",  # Safe information-gathering action that can become overused.
    "scan",  # Another safe action that slightly reduces threat and can also collapse.
    "isolate_host",  # Stronger defensive action with situational payoff.
    "patch",  # Maintenance action that should help only in the right context.
    "deploy_decoy",  # Defensive deception action that counters higher attack pressure.
    "do_nothing",  # Explicit bad action kept for policy expressiveness.
)  # End the action-name tuple.


@dataclass  # Mark the snapshot helper as a dataclass for compact field storage.
class EnvSnapshot:  # Group the current state values that form the observation vector.
    threat_level: float  # Track how dangerous the current threat state is.
    host_health: float  # Track the defended host's remaining health.
    visibility: float  # Track how much useful information the defender has gathered.
    decoy_level: float  # Track how much decoy protection is currently active.
    patch_level: float  # Track how much patching progress is currently active.
    steps_remaining_ratio: float  # Track progress through the episode as a normalized ratio.


class MiniDefenseEnv(gym.Env[np.ndarray, int]):  # Implement the lightweight custom environment.
    metadata = {"render_modes": []}  # Declare no rendering modes because training is headless.

    def __init__(self, max_steps: int = 40, seed: int | None = None) -> None:  # Initialize the environment.
        super().__init__()  # Initialize the Gym base class.
        self.max_steps = max_steps  # Store the target episode length in the requested 30 to 50 range.
        self.action_space = spaces.Discrete(len(ACTION_NAMES))  # Expose the 6-action discrete action space.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)  # Expose a compact normalized observation space.
        self._rng = np.random.default_rng(seed)  # Create a local random generator for reproducibility.
        self._step_count = 0  # Track the current step inside the episode.
        self._last_action = ACTION_NAMES.index("do_nothing")  # Start with a neutral previous action.
        self._repeat_count = 0  # Track consecutive repeated actions to encourage collapse signals.
        self._threat_level = 0.3  # Initialize a default threat level before the first reset.
        self._host_health = 1.0  # Initialize the host at full health.
        self._visibility = 0.2  # Initialize a modest amount of situational awareness.
        self._decoy_level = 0.0  # Initialize decoy protection at zero.
        self._patch_level = 0.0  # Initialize patch protection at zero.
        self._episode_reward = 0.0  # Track cumulative reward for success labeling.

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:  # Reset the environment for a new episode.
        del options  # Explicitly ignore unused reset options to keep the function clean.
        if seed is not None:  # Rebuild the RNG only when the caller provides a seed.
            self._rng = np.random.default_rng(seed)  # Reset the RNG for deterministic episodes.
        self._step_count = 0  # Start the episode at step zero.
        self._last_action = ACTION_NAMES.index("do_nothing")  # Reset the previous-action tracker.
        self._repeat_count = 0  # Reset the repeated-action counter.
        self._threat_level = float(self._rng.uniform(0.25, 0.5))  # Sample a slightly wider initial threat range.
        self._host_health = 1.0  # Restore full host health at episode start.
        self._visibility = float(self._rng.uniform(0.08, 0.24))  # Start with lower average visibility so information actions matter more.
        self._decoy_level = 0.0  # Reset decoy protection.
        self._patch_level = 0.0  # Reset patch protection.
        self._episode_reward = 0.0  # Reset cumulative reward tracking.
        return self._get_obs(), {}  # Return the initial observation with empty reset info.

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:  # Advance the environment one step.
        assert self.action_space.contains(action)  # Reject invalid action ids during development.
        self._step_count += 1  # Advance the episode clock.
        self._update_repeat_counter(action)  # Update the action repetition tracker.
        attack_pressure = self._sample_attack_pressure()  # Sample the latent attack intensity for this step.
        reward = self._apply_action(action, attack_pressure)  # Apply the action and get its immediate reward.
        self._apply_dynamics(attack_pressure)  # Update the environment state after the action effect.
        self._episode_reward += reward  # Accumulate reward for end-of-episode success logic.
        terminated = self._host_health <= 0.05  # End early if the host is effectively lost.
        truncated = self._step_count >= self.max_steps  # Truncate when the fixed episode horizon is reached.
        is_success = bool((terminated is False) and (self._host_health >= 0.45) and (self._episode_reward >= 1.0))  # Require positive cumulative value for success so poor policies are easier to distinguish.
        info = {"action_name": ACTION_NAMES[action], "threat_level": round(self._threat_level, 4), "host_health": round(self._host_health, 4), "is_success": is_success if (terminated or truncated) else False}  # Build the step info dictionary.
        return self._get_obs(), float(reward), terminated, truncated, info  # Return the standard Gym step tuple.

    def _update_repeat_counter(self, action: int) -> None:  # Track whether the policy is repeating the same action.
        if action == self._last_action:  # Check whether the current action matches the previous one.
            self._repeat_count += 1  # Increase the repeat count when the action repeats.
        else:  # Handle a change in action choice.
            self._repeat_count = 1  # Reset repetition to one occurrence of the new action.
            self._last_action = action  # Store the new action as the previous action.

    def _sample_attack_pressure(self) -> float:  # Sample attack pressure from the current latent threat state.
        base = self._threat_level + self._rng.uniform(-0.06, 0.1)  # Add small random variation around the threat level.
        return float(np.clip(base, 0.0, 1.0))  # Clip the pressure to the normalized range.

    def _apply_action(self, action: int, attack_pressure: float) -> float:  # Translate the chosen action into reward and state changes.
        reward = 0.0  # Start from zero immediate reward each step.
        if action == 0:  # Handle the monitor action.
            self._visibility = min(1.0, self._visibility + 0.12)  # Monitoring improves future visibility.
            reward += 0.16 - 0.06 * max(0, self._repeat_count - 2)  # Repeat use erodes monitor value while keeping it somewhat useful.
        elif action == 1:  # Handle the scan action.
            self._visibility = min(1.0, self._visibility + 0.16)  # Scanning improves visibility more strongly.
            self._threat_level = max(0.0, self._threat_level - 0.05)  # Scanning slightly reduces uncertainty-driven threat.
            reward += 0.14 - 0.06 * max(0, self._repeat_count - 2)  # Repeat use also erodes scan value.
        elif action == 2:  # Handle the isolate_host action.
            timely_bonus = 0.42 if attack_pressure > 0.55 else 0.08  # Isolation should be rewarded only when pressure is meaningfully high.
            self._threat_level = max(0.0, self._threat_level - 0.16)  # Isolation meaningfully reduces immediate threat.
            reward += timely_bonus - 0.1  # Add a moderate action cost so isolation is not universally dominant.
        elif action == 3:  # Handle the patch action.
            patch_ready = self._visibility > 0.55 and attack_pressure > 0.48  # Require both visibility and pressure for patch to be strongly useful.
            patch_bonus = 0.28 if patch_ready else -0.02  # Make poorly timed patching slightly harmful instead of nearly free.
            self._patch_level = min(1.0, self._patch_level + 0.12)  # Build patch mitigation more slowly than before.
            self._threat_level = max(0.0, self._threat_level - 0.07)  # Let patching reduce current threat, but less aggressively.
            reward += patch_bonus - 0.12  # Increase patch cost so repeated patching is not an easy local optimum.
        elif action == 4:  # Handle the deploy_decoy action.
            decoy_ready = attack_pressure > 0.52 and self._visibility < 0.45  # Make decoys strongest when pressure is high but visibility is still limited.
            decoy_bonus = 0.2 if decoy_ready else -0.04  # Make poorly timed decoys slightly harmful instead of broadly safe.
            self._decoy_level = min(1.0, self._decoy_level + 0.16)  # Build decoy protection more slowly than before.
            reward += decoy_bonus - 0.08 * max(0, self._repeat_count - 1)  # Add a repeat-use cost so repeated decoying is not a low-risk default.
        else:  # Handle the do_nothing action.
            reward -= 0.2  # Doing nothing is intentionally bad.
        if self._repeat_count >= 4 and action in (0, 1):  # Add an extra collapse penalty to repeated safe actions.
            reward -= 0.18  # This makes monitor/scan overuse harmful long term.
        if self._repeat_count >= 3 and action == 3:  # Add a dedicated patch-collapse penalty so repeated patching can also be diagnosed.
            reward -= 0.12  # This limits the previous tendency to collapse into patch.
        if self._repeat_count >= 3 and action == 4:  # Add a dedicated decoy-collapse penalty so repeated decoying is also discouraged.
            reward -= 0.12  # This prevents the environment from simply shifting collapse from patch to decoy.
        return reward  # Return the immediate reward after the action effect.

    def _apply_dynamics(self, attack_pressure: float) -> None:  # Advance the latent environment dynamics after the action.
        mitigation = 0.07 * self._visibility + 0.11 * self._decoy_level + 0.09 * self._patch_level  # Combine defensive factors into mitigation with weaker decoy influence than the previous iteration.
        net_pressure = max(0.0, attack_pressure - mitigation)  # Compute the remaining effective attack pressure.
        self._host_health = float(np.clip(self._host_health - net_pressure * 0.2, 0.0, 1.0))  # Damage the host based on net pressure.
        self._threat_level = float(np.clip(self._threat_level + self._rng.uniform(0.02, 0.07), 0.0, 1.0))  # Let threat naturally creep back up.
        self._visibility = float(np.clip(self._visibility * 0.97, 0.0, 1.0))  # Let visibility decay slightly over time.
        self._decoy_level = float(np.clip(self._decoy_level * 0.9, 0.0, 1.0))  # Let decoy effectiveness decay faster so repeated decoying is less sticky.
        self._patch_level = float(np.clip(self._patch_level * 0.96, 0.0, 1.0))  # Let patch effectiveness decay faster so repeated patching is less sticky.

    def _get_obs(self) -> np.ndarray:  # Build the current observation vector.
        snapshot = EnvSnapshot(threat_level=self._threat_level, host_health=self._host_health, visibility=self._visibility, decoy_level=self._decoy_level, patch_level=self._patch_level, steps_remaining_ratio=1.0 - (self._step_count / self.max_steps))  # Gather the normalized state into a named structure.
        return np.array(list(snapshot.__dict__.values()), dtype=np.float32)  # Convert the snapshot into a float32 vector.
