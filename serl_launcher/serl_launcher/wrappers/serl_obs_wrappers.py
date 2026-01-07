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

        # FIX: Check if images exist before accessing
        images_space = {}
        if "images" in self.env.observation_space.spaces:
             images_space = self.env.observation_space["images"]

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

        # FIX: Only add images if they exist in the observation
        if "images" in obs and obs["images"]:
             new_obs.update(obs["images"])

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
