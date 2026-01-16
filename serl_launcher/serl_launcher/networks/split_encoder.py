from typing import Dict, Iterable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from serl_launcher.networks.mlp import MLP, MLPResNetBlock

class SplitObsEncoder(nn.Module):
    """
    Encodes observations by splitting the state into Robot and Env parts,
    processing them with separate MLPs, and concatenating the results.

    Expected Input:
        observations (Dict): Must contain key "state".
        "state" should be a flattened vector where:
            - First 19 elements are Robot State (tcp(3)+vel(3)+grip(1)+jpos(6)+jvel(6)).
            - The rest are Env State (block_pos, target_pos, obstacle_state...).
    """

    # Architecture params hardcoded as per user request
    # Robot Branch: [32, 64]
    robot_hidden_dims = [32, 64]
    # Env Branch: [64, 128, 64] using ResNet blocks
    env_hidden_dims = [64, 128, 64]

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False, # Kept for API compatibility with EncodingWrapper
    ) -> jnp.ndarray:

        # 1. Extract State
        # API Compatibility: EncodingWrapper expects 'state' in observations
        if "state" not in observations:
            raise KeyError("SplitObsEncoder requires 'state' in observations.")

        state = observations["state"]

        # 2. Handle Time/Batch dimensions
        # EncodingWrapper logic:
        # if len(state.shape) == 2: rearrange(state, "T C -> (T C)")
        # if len(state.shape) == 3: rearrange(state, "B T C -> B (T C)")

        if state.ndim == 2: # (T, C) -> (T*C)
             state = rearrange(state, "T C -> (T C)")
             # Now state is 1D (C)
        elif state.ndim == 3: # (B, T, C) -> (B, T*C)
             state = rearrange(state, "B T C -> B (T C)")
             # Now state is 2D (B, C)

        # If obs_horizon=1, T=1, so C is preserved.

        # 3. Split
        # Robot: Indices 0 to 19
        # Env: Indices 19 to end
        robot_state = state[..., :19]
        env_state = state[..., 19:]

        # 4. Robot Branch
        # Dense -> LN -> ReLU
        x_robot = robot_state
        for size in self.robot_hidden_dims:
            x_robot = nn.Dense(size)(x_robot)
            x_robot = nn.LayerNorm()(x_robot)
            x_robot = nn.relu(x_robot)

        # 5. Env Branch (ResNet)
        # Using MLPResNetBlock to process eigen/feature vectors
        x_env = env_state
        # First project to initial size if needed, or rely on first ResBlock to handle dimension change
        for size in self.env_hidden_dims:
            x_env = MLPResNetBlock(
                features=size,
                act=nn.relu,
                use_layer_norm=True
            )(x_env, train=train)

        # 6. Concatenate
        combined = jnp.concatenate([x_robot, x_env], axis=-1)

        if stop_gradient:
            combined = jax.lax.stop_gradient(combined)

        return combined
