import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # FIX: Check if images exist AND are not empty before accessing
        # This prevents JAX tracing errors or recompilation lag when image_obs=False but 'images' key exists
        images_space = {}
        if "images" in self.env.observation_space.spaces:
             # Verify it's not a dummy/empty space
             img_space_candidate = self.env.observation_space["images"]
             if hasattr(img_space_candidate, 'spaces') and len(img_space_candidate.spaces) > 0:
                 images_space = img_space_candidate
             elif isinstance(img_space_candidate, gym.spaces.Box) and img_space_candidate.shape != (0,):
                 # Rare case where 'images' is a single Box, not Dict
                 images_space = {"images": img_space_candidate}

        # If images_space is None or empty, we just have state
        if images_space:
             self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                    **images_space,
                }
            )
        else:
             self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                }
            )

    def observation(self, obs):
        flat_state = flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            )

        new_obs = {"state": flat_state}

        # FIX: Only add images if they exist in the observation and space is defined
        if "images" in obs and obs["images"] and "images" in self.observation_space.spaces:
             new_obs.update(obs["images"])
        elif "images" in obs and obs["images"]:
             # If observation has images but space doesn't (filtered out), check if we should add them
             # Typically we trust the space definition from __init__
             pass

        return new_obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        # This helper also needs fixing if it's used elsewhere
        flat_state = flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            )
        new_obs = {"state": flat_state}
        if "images" in obs and obs["images"]:
             new_obs.update(obs["images"])
        return new_obs
