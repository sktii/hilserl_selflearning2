import os
import jax
import jax.numpy as jnp
import numpy as np
import glfw
import gymnasium as gym

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

from examples.experiments.config import DefaultTrainingConfig
# from examples.experiments.ram_insertion.wrapper import RAMEnv # Commented out

from franka_sim.envs.panda_stack_gym_env import PandaStackCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "left": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "right": {
            "serial_number": "none",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "left": lambda img: img,
        "wrist": lambda img: img,
        "right": lambda img: img,
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

        # 預設狀態設為 open
        self.gripper_state = 'open' 
        self.intervened = False
        self.action_length = 0.3
        self.current_action = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.flag = False
        self.key_states = {
            'w': False, 'a': False, 's': False, 'd': False,
            'j': False, 'k': False, 'l': False, ';': False,
        }

        # 設置 GLFW 鍵盤回調
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
                self.flag = True # 觸發夾爪狀態切換
            elif key == glfw.KEY_SEMICOLON:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened
                print(f"Intervention toggled: {self.intervened}")

        elif action == glfw.RELEASE:
            if key == glfw.KEY_W: self.key_states['w'] = False
            elif key == glfw.KEY_A: self.key_states['a'] = False
            elif key == glfw.KEY_S: self.key_states['s'] = False
            elif key == glfw.KEY_D: self.key_states['d'] = False
            elif key == glfw.KEY_J: self.key_states['j'] = False
            elif key == glfw.KEY_K: self.key_states['k'] = False
            elif key == glfw.KEY_L: self.key_states['l'] = False

        # 更新移動動作向量 (x, y, z)
        self.current_action[:3] = [
            int(self.key_states['w']) - int(self.key_states['s']), # x
            int(self.key_states['a']) - int(self.key_states['d']), # y
            int(self.key_states['j']) - int(self.key_states['k']), # z
        ]
        self.current_action[:3] *= self.action_length

    def action(self, action: np.ndarray) -> np.ndarray:
        # 1. 準備人類專家的動作
        expert_a = self.current_action.copy()

        if self.gripper_enabled:
            # 處理夾爪切換邏輯
            if self.flag:
                if self.gripper_state == 'open':
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
                self.flag = False
            
            # Close -> 正值; Open -> 負值
            if self.gripper_state == 'close':
                gripper_action = np.random.uniform(0.9, 1, size=(1,)) 
            else:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            
            expert_a = np.concatenate((expert_a[:3], gripper_action), axis=0)

        # 2. Action Masking
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # 3. 決定返回動作
        if self.intervened:
            return expert_a, True
        else:
            # 同步狀態
            if self.gripper_enabled:
                if action[-1] > 0:
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
            
            return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        return obs, info


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left", "wrist", "right"]
    classifier_keys = ["left", "wrist", "right"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    replay_buffer_capacity = 10000

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        env = PandaStackCubeGymEnv(render_mode=render_mode, image_obs=True, hz=8, config=EnvConfig())
        classifier=False
        
        if not fake_env:
            env = KeyBoardIntervention2(env)
            pass
        
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
