import os
import jax
import jax.numpy as jnp
import numpy as np
import glfw
import gymnasium as gym
import cv2

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from examples_UR5e.experiments.config import DefaultTrainingConfig

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "left": {
            "serial_number": "127122270146",
            "dim": (128, 128),
            "exposure": 40000,
        },
        "wrist": {
            "serial_number": "127122270350",
            "dim": (128, 128),
            "exposure": 40000,
        },
        "right": {
            "serial_number": "none",
            "dim": (128, 128),
            "exposure": 40000,
        },
    }
    def crop_and_resize(img):
        return cv2.resize(img, (128, 128)) 

    IMAGE_CROP = {
        "left": crop_and_resize,
        "wrist": crop_and_resize,
        "right": crop_and_resize,
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.045, 0.18, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 260
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

        # Initialize gripper state to 'open'
        self.gripper_state = 'open'
        self.intervened = False
        self.action_length = 0.3
        self.current_action = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.flag = False
        self.key_states = {
            'w': False, 'a': False, 's': False, 'd': False,
            'j': False, 'k': False, 'l': False, ';': False,
        }
        self.last_gripper_pos = 0.0

        # Setup GLFW key callback
        if self.env.render_mode == "human" and hasattr(self.env, "_viewer") and self.env._viewer:
             if hasattr(self.env._viewer, "viewer") and self.env._viewer.viewer:
                  if hasattr(self.env._viewer.viewer, "window") and self.env._viewer.viewer.window:
                       glfw.set_key_callback(self.env._viewer.viewer.window, self.glfw_on_key)

    def glfw_on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W: self.key_states['w'] = True
            elif key == glfw.KEY_A: self.key_states['a'] = True
            elif key == glfw.KEY_S: self.key_states['s'] = True
            elif key == glfw.KEY_D: self.key_states['d'] = True
            elif key == glfw.KEY_J: self.key_states['j'] = True
            elif key == glfw.KEY_K: self.key_states['k'] = True
            elif key == glfw.KEY_L: 
                self.key_states['l'] = True
                self.flag = True # Trigger gripper state toggle
            elif key == glfw.KEY_SEMICOLON:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened

                # Immediate sync on toggle to be safe
                if self.intervened and self.gripper_enabled:
                     # Using 0.05 threshold (Open ~0.03, Closed > 0.1)
                     if self.last_gripper_pos > 0.05:
                          self.gripper_state = 'close'
                     else:
                          self.gripper_state = 'open'
                     print(f"Intervention ON. Synced gripper to: {self.gripper_state} (pos={self.last_gripper_pos:.3f})")
                else:
                     print(f"Intervention toggled: {self.intervened}")

        elif action == glfw.RELEASE:
            if key == glfw.KEY_W: self.key_states['w'] = False
            elif key == glfw.KEY_A: self.key_states['a'] = False
            elif key == glfw.KEY_S: self.key_states['s'] = False
            elif key == glfw.KEY_D: self.key_states['d'] = False
            elif key == glfw.KEY_J: self.key_states['j'] = False
            elif key == glfw.KEY_K: self.key_states['k'] = False
            elif key == glfw.KEY_L: self.key_states['l'] = False

        # Update movement action vector (x, y, z)
        self.current_action[:3] = [
            int(self.key_states['w']) - int(self.key_states['s']), # x
            int(self.key_states['a']) - int(self.key_states['d']), # y
            int(self.key_states['j']) - int(self.key_states['k']), # z
        ]
        self.current_action[:3] *= self.action_length

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.current_action.copy()

        if self.gripper_enabled:
            # Handle gripper toggle logic
            if self.flag:
                if self.gripper_state == 'open':
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
                self.flag = False
            
            # Generate gripper action based on state
            if self.gripper_state == 'close':
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
            else:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))

            if self.env.action_space.shape[0] == 4:
                 expert_a = np.concatenate((expert_a[:3], gripper_action), axis=0)
            elif self.env.action_space.shape[0] == 7:
                 expert_a = np.concatenate((expert_a[:3], np.array([0,0,0,1]), gripper_action), axis=0)

        # Action Masking
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # Determine which action to return
        if self.intervened:
            return expert_a, True
        else:
            # Sync state: observe ENV state (physical) to update self.gripper_state
            # Using phys_gripper_pos from info is safer than command in obs
            if self.gripper_enabled:
                # Threshold for physical joint (0-1 normalized).
                # 0 is open (~0.03), 1 is closed.
                # Threshold > 0.05 implies intent to close or successful close.
                if self.last_gripper_pos > 0.05:
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
            
            return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # Capture actual physical gripper position from info for next sync
        try:
            val = None
            if "phys_gripper_pos" in info:
                val = info["phys_gripper_pos"]
            elif "state" in obs:
                 # Fallback to obs (command) if physical info missing
                if "ur5e/gripper_pos" in obs["state"]:
                    val = obs["state"]["ur5e/gripper_pos"]

            if val is not None:
                 if hasattr(val, "__getitem__") and hasattr(val, "__len__") and len(val) > 0:
                     self.last_gripper_pos = val[0]
                 else:
                     self.last_gripper_pos = val
        except Exception:
            pass

        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        self.last_gripper_pos = 0.0

        # Initial capture
        try:
            val = None
            if "phys_gripper_pos" in info:
                val = info["phys_gripper_pos"]
            elif "state" in obs:
                if "ur5e/gripper_pos" in obs["state"]:
                    val = obs["state"]["ur5e/gripper_pos"]

            if val is not None:
                 if hasattr(val, "__getitem__") and len(val) > 0:
                     self.last_gripper_pos = val[0]
                 else:
                     self.last_gripper_pos = val
        except Exception:
            pass

        return obs, info


class TrainConfig(DefaultTrainingConfig):
    image_keys = []
    classifier_keys = ["left", "wrist", "right"]
    proprio_keys = ["ur5e/tcp_pos", "ur5e/tcp_vel", "ur5e/gripper_pos", "ur5e/joint_pos", "ur5e/joint_vel", "block_pos", "target_cube_pos", "obstacle_state"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 200
    encoder_type = "mlp"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    replay_buffer_capacity = 10000

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        if os.environ.get("MUJOCO_GL") == "egl" and render_mode == "human":
            render_mode = "rgb_array"
            print("Switched render_mode to rgb_array due to MUJOCO_GL=egl")

        env = UR5eStackCubeGymEnv(render_mode=render_mode, image_obs=False, hz=12, config=EnvConfig())

        # NOTE: Classifier is force disabled here based on previous code snippets?
        # But 'classifier' arg comes in.
        # Ideally we respect the arg, but user wants to fix lag.
        # I will use the arg but add warmup.
        # classifier=False # This line was overriding the arg in previous snippet!

        if not fake_env:
            env = KeyBoardIntervention2(env)
            pass

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        classifier=False
        if classifier:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            # Warmup to prevent lag during interaction
            print("Compiling reward classifier...")
            dummy_obs = env.observation_space.sample()
            # Ensure dummy obs has correct keys for classifier
            classifier_func(dummy_obs)
            print("Classifier compiled.")

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                pred = sigmoid(classifier_func(obs))
                if hasattr(pred, 'shape') and len(pred.shape) > 0:
                    pred = pred.flatten()[0]
                return int(float(pred) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env