from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import gymnasium # Need gymnasium.spaces for SERL compatibility
import mujoco
import numpy as np
from gym import spaces as gym_spaces # Keep gym spaces for legacy compat
from gymnasium import spaces as gymnasium_spaces # Use gymnasium spaces for env spaces
import time
import cv2
# [新增] 為了讓監控面板能連線
import threading
import logging
from flask import Flask, jsonify

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from dm_robotics.transformations import transformations as tr
from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena2.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class PandaStackCubeGymEnv(MujocoGymEnv, gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 10,
    ):
        self.hz = hz
        # Correctly load action scale from config if available
        if config is not None and hasattr(config, 'ACTION_SCALE'):
            # Assuming config.ACTION_SCALE is (trans, rot, grasp) or similar.
            # Our environment action space is 7D, but action_scale attribute usage depends on step() impl.
            # Currently step() uses action_scale[0] for pos, action_scale[1] for grasp.
            # If config has 3 values, we take 0 and 2? Or 0 and 1?
            # User config has (0.01, 0.06, 1). 0.01 is trans, 0.06 is rot, 1 is grasp.
            # So we should probably store them all.
            # But self._action_scale is expected to be numpy array.
            # Let's update self._action_scale to be the full array or map correctly.
            self._action_scale = np.array(config.ACTION_SCALE)
        else:
            self._action_scale = action_scale

        MujocoGymEnv.__init__(
            self,
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        # Use safe ID lookup
        self.camera_id = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "left"),
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "handcam_rgb"),
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "right"),
        ]
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]
        self._target_cube_id = self._model.body("target_cube").id
        self._target_cube_geom_id = self._model.geom("target_geom").id
        self._target_cube_z = self._model.geom("target_geom").size[2]

        if self.image_obs:
            self.observation_space = gymnasium_spaces.Dict(
                {
                    "state": gymnasium_spaces.Dict(
                        {
                            "tcp_pose": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(7,), dtype=np.float32
                            ),  # xyz + quat
                            "tcp_vel": gymnasium_spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            "gripper_pose": gymnasium_spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                            "tcp_force": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                            "tcp_torque": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                            "target_cube_pos": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        }
                    ),
                    "images": gymnasium_spaces.Dict(
                        {key: gymnasium_spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        else:
            self.observation_space = gymnasium_spaces.Dict(
                {
                    "state": gymnasium_spaces.Dict(
                        {
                            "panda/tcp_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "block_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "target_cube_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                        }
                    ),
                }
            )
        self.action_space = gymnasium_spaces.Box(
                    low=np.asarray([-1.0, -1.0, -1.0, -1.0]),
                    high=np.asarray([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
        )
        
        try:
             # NOTE: gymnasium is used here since MujocoRenderer is not available in gym.
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(
                self.model,
                self.data,
                # width=render_spec.width, # Removed to avoid unexpected argument error
                # height=render_spec.height,
            )
            # manually set width/height if possible/needed
            if hasattr(self._viewer, 'width'):
                self._viewer.width = render_spec.width
            if hasattr(self._viewer, 'height'):
                self._viewer.height = render_spec.height

            if self.render_mode == "human":
                self._viewer.render(self.render_mode)
            # [新增] 啟動監控 Server
            self._start_monitor_server()
        except ImportError:
            # Fallback or error if gymnasium not available or headless issue
            # In headless environment without GL, this might fail.
            print("Warning: Could not initialize MujocoRenderer. Rendering might be disabled.")
            self._viewer = None
        except Exception as e:
             print(f"Warning: Failed to initialize MujocoRenderer: {e}")
             self._viewer = None

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        # Sample a new target_cube position.
        # Ensure it's not too close to the block
        # while True:
        #     target_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        #     if np.linalg.norm(target_xy - block_xy) > 0.3:
        #         breakF
        target_xy = np.array([0.4, 0.25])
        # 2. 隨機生成方塊位置，直到它與目標距離 > 0.15 (或您想要的最小距離)
        while True:
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            # 檢查距離
            if np.linalg.norm(block_xy - target_xy) > 0.35: 
                break
        # 3. 應用位置
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        # Since target_cube is static (no joint), we modify its body position in the model
        # Note: changing model affects all future steps, but we reset it every time here.
        self._model.body_pos[self._target_cube_id][:2] = target_xy
        # Z position is fixed as in XML (0.025)

        # Randomize pillars
        self._randomize_pillars(block_xy, target_xy)

        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + self._target_cube_z * 2

        self.env_step = 0
        self.success_counter = 0
        # [新增] 初始化階段性獎勵標記
        self._stage_rewards = {
            "touched": False,
            "lifted": False,
            "hovered": False
        }
        obs = self._compute_observation()
        return obs, {"succeed": False}

    def _randomize_pillars(self, block_xy, target_xy):
        # Workspace bounds: x=[0.25, 0.55], y=[-0.25, 0.25] from _SAMPLING_BOUNDS
        # Pillars should avoid the target and block AND the robot start position.
        safe_dist = 0.14
        start_pos = np.array([0.3, 0.0]) # Approx home X,Y

        # Helper to get random pos avoiding block, target, and start
        def get_safe_pos():
            for _ in range(100):
                px = self._random.uniform(0.2, 0.6)
                py = self._random.uniform(-0.3, 0.3)
                pos = np.array([px, py])
                if (np.linalg.norm(pos - block_xy) > safe_dist and
                    np.linalg.norm(pos - target_xy) > safe_dist and
                    np.linalg.norm(pos - start_pos) > safe_dist):
                    return pos
            return np.array([0.8, 0.8]) # Fallback outside

        # Pillar cylinders 1-2
        for i in range(1, 3):
            name = f"pillar_cyl_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                # Random pos
                pos = get_safe_pos()
                self._model.geom_pos[body_id][:2] = pos

                # Random size: cylinder size is [radius, half_height]
                # radius ~ 0.02 to 0.03 (increased min radius for better collision)
                # height ~ 0.1025 to 0.1675 (reduced range to half of 0.07-0.2, mean 0.135)
                radius = self._random.uniform(0.02, 0.03)
                half_height = self._random.uniform(0.0625, 0.1075)
                self._model.geom_size[body_id] = [radius, half_height, 0]

                # Adjust Z pos to sit on floor
                self._model.geom_pos[body_id][2] = half_height

                # Set color to black
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]

        # Pillar boxes 1-2
        for i in range(1, 3):
            name = f"pillar_box_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                pos = get_safe_pos()
                self._model.geom_pos[body_id][:2] = pos

                # Random size: box size is [hx, hy, hz]
                hx = self._random.uniform(0.02, 0.03)
                hy = self._random.uniform(0.02, 0.03)
                hz = self._random.uniform(0.1025, 0.1675)
                self._model.geom_size[body_id] = [hx, hy, hz]

                # Adjust Z pos to sit on floor
                self._model.geom_pos[body_id][2] = hz

                # Set color to black
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]
    def _start_monitor_server(self):
        """啟動一個背景 HTTP Server 讓 dashboard 讀取數據"""
        try:
            app = Flask("SimMonitor")
            # 關閉 Flask 的囉嗦日誌
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            @app.route('/getstate', methods=['POST'])
            def get_state():
                # 1. 獲取位置 (Position)
                # Stack 環境通常也有這個 sensor，如果報錯改用 self._data.site_xpos[self._pinch_site_id]
                try:
                    pos = self._data.sensor("2f85/pinch_pos").data.tolist()
                except:
                     # 備用方案：直接抓 site 位置
                    pos = self._data.site_xpos[self._pinch_site_id].tolist()

                # 2. 獲取旋轉 (Rotation Quat) [w, x, y, z] -> [x, y, z, w]
                # MuJoCo 的 site_xmat 是 9個數的旋轉矩陣
                site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
                quat_mujoco = np.zeros(4)
                mujoco.mju_mat2Quat(quat_mujoco, site_mat)
                
                # Dashboard 預期 [x, y, z, qx, qy, qz, qw]
                pose = [
                    pos[0], pos[1], pos[2],      # x, y, z
                    quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0] # qx, qy, qz, qw
                ]

                # 3. 獲取夾爪狀態 (0~1)
                g = self._data.ctrl[self._gripper_ctrl_id] / 255.0

                return jsonify({
                    "pose": pose,
                    "gripper_pos": g,
                    "vel": [0]*6, 
                    "force": [0]*3,
                    "torque": [0]*3
                })

            def run_app():
                try:
                    # 嘗試開啟 Port 5000
                    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
                except Exception as e:
                    print(f"[SimMonitor] Port 5000 被佔用或無法啟動: {e}")

            # 在背景執行
            t = threading.Thread(target=run_app)
            t.daemon = True
            t.start()
            print("[SimMonitor] 監控 Server 已啟動於 http://127.0.0.1:5000")
            
        except Exception as e:
            print(f"[SimMonitor] 初始化失敗: {e}")
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        start_time = time.time()

        # Handle scaling based on config (3 values) or default (2 values)
        if len(self._action_scale) == 3:
            trans_scale = self._action_scale[0]
            # rot_scale = self._action_scale[1] # Unused now
            grasp_scale = self._action_scale[2]
        else:
            trans_scale = self._action_scale[0]
            # rot_scale = 1.0 # Default/Fallback
            grasp_scale = self._action_scale[1]

        # Action components
        x, y, z, grasp = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * trans_scale
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * grasp_scale
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()

        if self.image_obs:
            gripper_key = "gripper_pose"
            gripper_val = obs["state"]["gripper_pose"]
        else:
            gripper_key = "panda/gripper_pos"
            gripper_val = obs["state"]["panda/gripper_pos"]

        if (action[-1] < -0.5 and gripper_val > 0.9) or (
            action[-1] > 0.5 and gripper_val < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0

        # terminated = self.time_limit_exceeded()
        self.env_step += 1
        terminated = False
        if self.env_step >= 260:
            terminated = True

        if self.render_mode == "human" and self._viewer:
            self._viewer.render(self.render_mode)
            dt = time.time() - start_time
            if self.intervened == True:
                time.sleep(max(0, (1.0 / self.hz) - dt))

        # Check collision
        collision = self._check_collision()
        if collision:
            # print("Collision with pillar detected!")
            terminated = False # Don't fail mission, just penalize
            rew = -5.0 # Heavy penalty per step
            success = False
            self.success_counter = 0
            # Continue episode
            # return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": grasp_penalty}
            # We continue execution logic below?
            # If we don't return, we execute success check. Success will likely be false if collision?
            # But we reset success_counter.
            # We should probably return here to avoid overriding rew?
            return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": grasp_penalty}

        instant_success = self._compute_success(gripper_val)
        if instant_success:
            self.success_counter += 1
        else:
            self.success_counter = 0

        success = self.success_counter >= (1.0 / self.control_dt)

        if success:
            print(f'success!')
        else:
            pass
        terminated = terminated or success
        if success:
            rew += 100.0
        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": grasp_penalty}

    def _check_collision(self):
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            # Check if either geometry is a pillar
            is_g1_pillar = geom1_name and "pillar" in geom1_name
            is_g2_pillar = geom2_name and "pillar" in geom2_name

            if is_g1_pillar or is_g2_pillar:
                # Identify the other object
                other = geom2_name if is_g1_pillar else geom1_name

                # If other is None (unnamed, likely robot) or not in allowed list
                if other is None or other not in ["block", "floor", "target_geom", "target"]:
                    return True
        return False

    def _compute_success(self, gripper_val):
        block_pos = self._data.sensor("block_pos").data
        # Target cube position. Note: self._data.body("target_cube").xpos gives current global pos
        target_pos = self._data.body("target_cube").xpos

        # Check XY overlap
        # target geom size is 0.025, block geom size is 0.02 (from xml)
        # Total width 0.045
        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.04

        # Check Z height
        # Block should be above target cube. Target cube top is at z ~ 0.05 (pos 0.025 + size 0.025)
        # Block z pos is center of block. Block size is 0.02. So block bottom is z - 0.02.
        # We want block bottom > target top approx.
        # target_pos[2] is center of target. target top is target_pos[2] + target_cube_z.
        # We require block center to be higher than target top + some margin (half block size).
        z_success = block_pos[2] > (target_pos[2] + self._target_cube_z + self._block_z * 0.9)

        # Check if gripper is open (released)
        # gripper_val is ~0 (closed) to 1 (open) or width.
        # If block width is 0.02, holding it means width ~0.02. If open, width > 0.02
        # But 'gripper_val' comes from observation which is normalized ctrl?
        # In _compute_observation: gripper_pos = ctrl / 255.
        # If open, ctrl is 255 -> 1.0. If closed on block, ctrl might still be 255?
        # No, 'fingers_actuator' is position controlled (usually) or force?
        # If position controlled, 255 = max width (0.08).
        # If we just released it, we commanded 0.0 (Open).
        # If we rely on obs["state"]["gripper_pose"], it is the commanded value.
        # If we commanded open, it is < 0.1.
        gripper_open = gripper_val < 0.1

        # Check if block is static
        # Joint 'block' is a freejoint.
        # qvel has 6 dims.
        block_vel = self._data.jnt("block").qvel[:3]
        is_static = np.linalg.norm(block_vel) < 0.05

        return xy_success and z_success and gripper_open and is_static

    def render(self):
        if self._viewer is None:
             return []

        try:
            rendered_frames = []
            for cam_id in self.camera_id:
                rendered_frames.append(
                    self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
                )
            return rendered_frames
        except Exception:
             return []

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        target_pos = self._data.body("target_cube").xpos.astype(np.float32)
        obs["state"]["target_cube_pos"] = target_pos

        if self.image_obs:
            obs["images"] = {}
            rendered = self.render()
            # We expect 3 frames: [front, wrist, back] based on self.camera_id
            if rendered and len(rendered) == 3:
                 obs["images"]["left"] = rendered[0]
                 obs["images"]["wrist"] = rendered[1]
                 obs["images"]["right"] = rendered[2]
            else:
                 # Provide empty images if rendering fails (e.g. headless)
                 obs["images"]["left"] = np.zeros((128, 128, 3), dtype=np.uint8)
                 obs["images"]["wrist"] = np.zeros((128, 128, 3), dtype=np.uint8)
                 obs["images"]["right"] = np.zeros((128, 128, 3), dtype=np.uint8)

        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos


        # startear add (keeping structure but filling with meaningful/consistent data where possible)
        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )

        if self.image_obs:
            # Reconstruct tcp_pose from tcp_pos and some orientation.
            # We don't have full pose easily available without computation or sensor.
            # Using tcp_pos for position. For orientation, we could use mocap_quat or keep random if not critical?
            # Reviewer complained about random noise.
            # Let's use mocap_quat for orientation if it's close enough, or just 0s if we don't track it.
            # Ideally we should add a sensor for it.
            # Existing code used random samples.

            # [修復] 使用真實的末端姿態，而非 mocap 控制指令
            site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, site_mat)
            final_tcp_pos[3:] = current_quat[[1, 2, 3, 0]]

            final_tcp_vel = np.zeros(6, dtype=np.float32)
            final_tcp_vel[:3] = tcp_vel
            # rotational vel?

            # Force/Torque? We don't have sensors configured in xml for force/torque at wrist?
            # XML has:
            # <sensor>
            # <framepos name="block_pos" objtype="geom" objname="block"/>
            # <framequat name="block_quat" objtype="geom" objname="block"/>
            # </sensor>
            # And panda.xml usually has standard sensors.
            # If we don't have them, zeros is better than random noise.

            # Try to get force/torque if available
            try:
                tcp_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
            except Exception:
                tcp_force = np.zeros(3, dtype=np.float32)

            try:
                tcp_torque = self._data.sensor("panda/wrist_torque").data.astype(np.float32)
            except Exception:
                tcp_torque = np.zeros(3, dtype=np.float32)

            obs['state'] = {
                "tcp_pose": final_tcp_pos,
                "tcp_vel": final_tcp_vel,
                "gripper_pose": gripper_pos,
                "tcp_force": tcp_force,
                "tcp_torque": tcp_torque,
                "target_cube_pos": target_pos
            }

            # If images are somehow missing in obs["images"] but we rendered them (e.g. if we want to overwrite),
            # we can do it here. But logic above already populates obs["images"].
            # The previous bug was unpacking 3 items into 2 variables.
            # We already fixed it in the 'if self.image_obs:' block at the top of this function
            # where we correctly assign front, wrist, back.
            # So we don't need to do anything here.
            pass
        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos

        # 1. Reach reward: Approach the block
        dist_to_block = np.linalg.norm(block_pos - tcp_pos)
        r_reach = (1 - np.tanh(5.0 * dist_to_block))

        # 2. Pick reward: Lift the block
        # Only reward lifting if we are close to the block
        is_grasped = dist_to_block < 0.03
        r_lift = 0.0
        if is_grasped or block_pos[2] > self._z_init + 0.01:
             r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
             r_lift = np.clip(r_lift, 0, 1)

        # 3. Place reward: Move block to target
        dist_block_to_target = np.linalg.norm(block_pos[:2] - target_pos[:2])
        r_place = 0.0
        if block_pos[2] > self._z_init + 0.02: # If lifted
             r_place = (1 - np.tanh(5.0 * dist_block_to_target))

        # 4. Stack reward: Align Z with target
        r_stack = 0.0
        target_z = target_pos[2] + self._target_cube_z + self._block_z
        if r_place > 0.95: # If close to target XY
            dist_z = np.abs(block_pos[2] - target_z)
            r_stack = (1 - np.tanh(10.0 * dist_z))

        rew = 0.5 * r_reach + 0.3 * r_lift + 0.3 * r_place + 0.3 * r_stack
        # Time penalty to encourage speed

        # ============================================================
        # [新增] 階段性一次性獎勵 (One-time Stage Rewards)
        # ============================================================
        
        # 1. 碰到方塊 (Touch): 距離小於 3cm 且還沒領過
        if not self._stage_rewards["touched"]:
            if dist_to_block < 0.03:
                rew += 10.0  # 給予小獎勵
                self._stage_rewards["touched"] = True
                print(">>> Reward: Touched Block (+10)")

        # 2. 夾起方塊 (Lift): 高度上升 且還沒領過
        # 判斷條件：方塊高度比初始高度高 3cm
        if not self._stage_rewards["lifted"]:
            if block_pos[2] > self._z_init + 0.03:
                rew += 25.0  # 給予中獎勵
                self._stage_rewards["lifted"] = True
                print(">>> Reward: Lifted Block (+25)")

        # 3. 移到目標上方 (Hover): XY 平面距離接近目標 且 已經夾起來了
        if not self._stage_rewards["hovered"]:
            # 只有在已經夾起來的情況下，移過去才給分 (避免它推著方塊過去)
            if self._stage_rewards["lifted"]: 
                dist_xy_to_target = np.linalg.norm(block_pos[:2] - target_pos[:2])
                if dist_xy_to_target < 0.05:
                    rew += 25.0 # 給予中獎勵
                    self._stage_rewards["hovered"] = True
                    print(">>> Reward: Hovered above Goal (+25)")

        return rew


if __name__ == "__main__":
    env = PandaStackCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()