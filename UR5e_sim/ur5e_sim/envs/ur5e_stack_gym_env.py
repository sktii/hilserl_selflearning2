from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import gymnasium # Need gymnasium.spaces for SERL compatibility
import mujoco
import numpy as np
from gym import spaces as gym_spaces # Keep gym spaces for legacy compat
from gymnasium import spaces as gymnasium_spaces # Use gymnasium spaces for env spaces
import time
import threading
import logging
from flask import Flask, jsonify
import requests

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from dm_robotics.transformations import transformations as tr
from ur5e_sim.controllers import opspace
from ur5e_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_ur5e.xml"
_UR5E_HOME = np.asarray([0, -1.57, 1.57, -1.57, -1.57, 0])

_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])
_MAX_OBSTACLES = 6

class UR5eStackCubeGymEnv(MujocoGymEnv, gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 10,
        real_robot: bool = False,
        real_robot_ip: str = "127.0.0.1",
    ):
        self.hz = hz
        self.real_robot = real_robot
        self.real_robot_ip = real_robot_ip

        if config is not None and hasattr(config, 'ACTION_SCALE'):
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
        self.camera_id = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "left"),
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "right"),
        ]
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # UR5e has 6 joints
        self._ur5e_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 7)]
        )
        self._ur5e_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 7)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]
        self._target_cube_id = self._model.body("target_cube").id
        self._target_cube_geom_id = self._model.geom("target_geom").id
        self._target_cube_z = self._model.geom("target_geom").size[2]

        # Find physical gripper joint for accurate state reading
        # Typically "robot0:2f85:right_driver_joint" for Robotiq 2F-85
        self._gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:2f85:right_driver_joint")

        # Pre-cache pillar IDs for fast collision checking
        self._pillar_geom_ids = []
        self._pillar_info = [] # Cache for _get_obstacle_state: list of (id, type)
        for i in range(1, 3):
            id_cyl = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_cyl_{i}")
            if id_cyl != -1:
                self._pillar_geom_ids.append(id_cyl)
                self._pillar_info.append((id_cyl, 'cyl'))

            id_box = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_box_{i}")
            if id_box != -1:
                self._pillar_geom_ids.append(id_box)
                self._pillar_info.append((id_box, 'box'))

        # Cache block ID
        self._block_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "block")

        # Cache gripper geom IDs to avoid string lookups in _check_grasp
        self._gripper_geom_ids = set()
        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ("pad" in name or "finger" in name or "2f85" in name):
                self._gripper_geom_ids.add(i)

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
                            "ur5e/tcp_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "ur5e/tcp_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "ur5e/gripper_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "block_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "target_cube_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "obstacle_state": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(_MAX_OBSTACLES * 7,), dtype=np.float32
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
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(
                self.model,
                self.data,
            )
            if hasattr(self._viewer, 'width'):
                self._viewer.width = render_spec.width
            if hasattr(self._viewer, 'height'):
                self._viewer.height = render_spec.height

            if self.render_mode == "human":
                self._viewer.render(self.render_mode)
        except ImportError:
            print("Warning: Could not initialize MujocoRenderer. Rendering might be disabled.")
            self._viewer = None
        except Exception as e:
             print(f"Warning: Failed to initialize MujocoRenderer: {e}")
             self._viewer = None

        self._safe_geom_ids = set()
        safe_names = ["block", "floor", "target_geom", "target"]
        for name in safe_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._safe_geom_ids.add(gid)
            else:
                print(f"Warning: Safe geom '{name}' not found in model.")

        if self.real_robot:
            self._start_monitor_server()
            self._connect_real_robot()

    def _connect_real_robot(self):
        url = f"http://{self.real_robot_ip}:5000/clearerr"
        print(f"[Sim] Connecting to Real Robot Server at {url}...")
        try:
            requests.post(url, timeout=2.0)
            print("[Sim] Connected to Real Robot Server.")
        except Exception as e:
            print(f"[Sim] Failed to connect to Real Robot Server: {e}")

    def _send_real_robot_command(self, tcp_pos, tcp_ori, gripper_pos):
        url = f"http://{self.real_robot_ip}:5000/pose"
        payload = np.concatenate([tcp_pos, tcp_ori]).tolist()
        try:
            requests.post(url, json={"arr": payload}, timeout=0.05)
            g_int = int(np.clip(gripper_pos * 255, 0, 255))
            requests.post(f"http://{self.real_robot_ip}:5000/move_gripper",
                          json={"gripper_pos": g_int}, timeout=0.05)

        except Exception as e:
             pass

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[self._ur5e_dof_ids] = _UR5E_HOME
        mujoco.mj_forward(self._model, self._data)

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        target_xy = np.array([0.4, 0.25])

        while True:
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            if np.linalg.norm(block_xy - target_xy) > 0.15:
                break

        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        self._model.body_pos[self._target_cube_id][:2] = target_xy

        self._randomize_pillars(block_xy, target_xy)

        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + self._target_cube_z * 2

        self.env_step = 0
        self.success_counter = 0
        self._stage_rewards = {
            "touched": False,
            "lifted": False,
            "hovered": False
        }
        obs = self._compute_observation()

        # Add physical gripper state to info
        info = {"succeed": False}
        if self._gripper_joint_id != -1:
            raw_pos = self._data.qpos[self._gripper_joint_id]
            # Normalize approx 0~0.8 to 0~1
            info["phys_gripper_pos"] = np.clip(raw_pos / 0.8, 0, 1)
        else:
             info["phys_gripper_pos"] = 0.0

        return obs, info

    def _get_obstacle_state(self):
        obs_state = np.zeros((_MAX_OBSTACLES, 7), dtype=np.float32)
        idx = 0

        for gid, ptype in self._pillar_info:
            if idx >= _MAX_OBSTACLES:
                break

            pos = self._model.geom_pos[gid]
            size = self._model.geom_size[gid]

            if ptype == 'cyl':
                # Cylinder: [radius, half_height, 0] -> [radius, radius, half_height]
                obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[0], size[1]]
            else:
                # Box: [hx, hy, hz] -> [hx, hy, hz]
                obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[1], size[2]]
            idx += 1

        return obs_state.flatten()

    def _randomize_pillars(self, block_xy, target_xy):
        safe_dist = 0.14
        start_pos = np.array([0.3, 0.0])

        def get_safe_pos():
            for _ in range(100):
                px = self._random.uniform(0.2, 0.6)
                py = self._random.uniform(-0.3, 0.3)
                pos = np.array([px, py])
                if (np.linalg.norm(pos - block_xy) > safe_dist and
                    np.linalg.norm(pos - target_xy) > safe_dist and
                    np.linalg.norm(pos - start_pos) > safe_dist):
                    return pos
            return np.array([0.8, 0.8])

        for i in range(1, 3):
            name = f"pillar_cyl_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                pos = get_safe_pos()
                self._model.geom_pos[body_id][:2] = pos
                radius = self._random.uniform(0.02, 0.03)
                half_height = self._random.uniform(0.0625, 0.1075)
                self._model.geom_size[body_id] = [radius, half_height, 0]
                self._model.geom_pos[body_id][2] = half_height
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]

        for i in range(1, 3):
            name = f"pillar_box_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                pos = get_safe_pos()
                self._model.geom_pos[body_id][:2] = pos
                hx = self._random.uniform(0.02, 0.03)
                hy = self._random.uniform(0.02, 0.03)
                hz = self._random.uniform(0.1025, 0.1675)
                self._model.geom_size[body_id] = [hx, hy, hz]
                self._model.geom_pos[body_id][2] = hz
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]

    def _start_monitor_server(self):
        import socket
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        try:
            app = Flask("SimMonitor")
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            @app.route('/getstate', methods=['POST'])
            def get_state():
                try:
                    pos = self._data.sensor("2f85/pinch_pos").data.tolist()
                except:
                    pos = self._data.site_xpos[self._pinch_site_id].tolist()

                site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
                quat_mujoco = np.zeros(4)
                mujoco.mju_mat2Quat(quat_mujoco, site_mat)

                pose = [
                    pos[0], pos[1], pos[2],      # x, y, z
                    quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0] # qx, qy, qz, qw
                ]
                g = self._data.ctrl[self._gripper_ctrl_id] / 255.0
                return jsonify({
                    "pose": pose,
                    "gripper_pos": g,
                    "vel": [0]*6,
                    "force": [0]*3,
                    "torque": [0]*3
                })

            def run_app():
                target_port = 5000
                while is_port_in_use(target_port):
                    print(f"[SimMonitor] Port {target_port} is busy, trying next...")
                    target_port += 1
                    if target_port > 5010:
                        print("[SimMonitor] No available ports found (5000-5010).")
                        return

                print(f"[SimMonitor] Monitor Server started at http://127.0.0.1:{target_port}")
                try:
                    app.run(host='0.0.0.0', port=target_port, debug=False, use_reloader=False)
                except Exception as e:
                    print(f"[SimMonitor] Server crashed: {e}")

            t = threading.Thread(target=run_app)
            t.daemon = True
            t.start()

        except Exception as e:
            print(f"[SimMonitor] Initialization failed: {e}")

    def _check_grasp(self):
        has_contact = False
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            # Use cached block ID
            if contact.geom1 == self._block_geom_id or contact.geom2 == self._block_geom_id:
                other_id = contact.geom2 if contact.geom1 == self._block_geom_id else contact.geom1
                # Use cached gripper IDs (set lookup is O(1))
                if other_id in self._gripper_geom_ids:
                    has_contact = True
                    break
        return has_contact

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()

        if len(self._action_scale) == 3:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[2]
        else:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[1]

        x, y, z, grasp = action

        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * trans_scale
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * grasp_scale
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._ur5e_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_UR5E_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4
            )
            self._data.ctrl[self._ur5e_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()

        if self.image_obs:
            gripper_key = "gripper_pose"
            gripper_val = obs["state"]["gripper_pose"]
        else:
            gripper_key = "ur5e/gripper_pos"
            gripper_val = obs["state"]["ur5e/gripper_pos"]

        if (action[-1] < -0.5 and gripper_val > 0.9) or (
            action[-1] > 0.5 and gripper_val < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0

        self.env_step += 1
        terminated = False
        if self.env_step >= 280:
            terminated = True

        if self.render_mode == "human" and self._viewer:
            self._viewer.render(self.render_mode)

        if self.render_mode == "human" or self.real_robot:
            dt = time.time() - start_time
            target_dt = 1.0 / self.hz
            sleep_time = max(0, target_dt - dt)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.real_robot:
            try:
                sim_tcp_pos = self._data.site_xpos[self._pinch_site_id]
                sim_site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
                sim_tcp_quat = np.zeros(4)
                mujoco.mju_mat2Quat(sim_tcp_quat, sim_site_mat)
                sim_tcp_quat_scipy = sim_tcp_quat[[1, 2, 3, 0]]
                sim_gripper_val = self._data.ctrl[self._gripper_ctrl_id] / 255.0

                self._send_real_robot_command(sim_tcp_pos, sim_tcp_quat_scipy, sim_gripper_val)
            except Exception as e:
                pass
        
        collision = self._check_collision()
        info = {"succeed": False, "grasp_penalty": grasp_penalty}

        # Add physical gripper state to info
        if self._gripper_joint_id != -1:
            raw_pos = self._data.qpos[self._gripper_joint_id]
            # Normalize approx 0~0.8 to 0~1
            info["phys_gripper_pos"] = np.clip(raw_pos / 0.8, 0, 1)
        else:
             info["phys_gripper_pos"] = 0.0

        if collision:
            terminated = False
            rew = -5.0
            success = False
            self.success_counter = 0
            return obs, rew, terminated, False, info

        instant_success = self._compute_success(gripper_val)
        if instant_success:
            self.success_counter += 1
        else:
            self.success_counter = 0

        success = self.success_counter >= (1.0 / self.control_dt)

        if success:
            print(f'success!')
            info["succeed"] = True

        terminated = terminated or success
        if success:
            rew += 100.0
        return obs, rew, terminated, False, info

    def _check_collision(self):
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            is_g1_pillar = g1 in self._pillar_geom_ids
            is_g2_pillar = g2 in self._pillar_geom_ids

            if is_g1_pillar or is_g2_pillar:
                other_id = g2 if is_g1_pillar else g1
                if other_id not in self._safe_geom_ids:
                    return True
        return False

    def _compute_success(self, gripper_val):
        block_pos = self._data.sensor("block_pos").data
        target_pos = self._data.body("target_cube").xpos

        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.04

        z_success = block_pos[2] > (target_pos[2] + self._target_cube_z + self._block_z * 0.8)

        gripper_open = gripper_val < 0.1

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

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        if self.image_obs:
            obs["images"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["ur5e/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["ur5e/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["ur5e/gripper_pos"] = gripper_pos

        target_pos = self._data.body("target_cube").xpos.astype(np.float32)
        obs["state"]["target_cube_pos"] = target_pos

        if self.image_obs:
            rendered = self.render()
            if rendered:
                 if len(rendered) >= 1: obs["images"]["left"] = rendered[0]
                 else: obs["images"]["left"] = np.zeros((128, 128, 3), dtype=np.uint8)

                 if len(rendered) >= 2: obs["images"]["right"] = rendered[1]
                 else: obs["images"]["right"] = np.zeros((128, 128, 3), dtype=np.uint8)

                 obs["images"]["wrist"] = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                 obs["images"]["left"] = np.zeros((128, 128, 3), dtype=np.uint8)
                 obs["images"]["wrist"] = np.zeros((128, 128, 3), dtype=np.uint8)
                 obs["images"]["right"] = np.zeros((128, 128, 3), dtype=np.uint8)

        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos
            obs["state"]["obstacle_state"] = self._get_obstacle_state()

        if self.image_obs:
            gripper_pos = np.array(
                [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
            )
            site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, site_mat)
            final_tcp_pos = np.zeros(7, dtype=np.float32)
            final_tcp_pos[:3] = tcp_pos
            final_tcp_pos[3:] = current_quat[[1, 2, 3, 0]]

            final_tcp_vel = np.zeros(6, dtype=np.float32)
            final_tcp_vel[:3] = tcp_vel

            try:
                tcp_force = self._data.sensor("robot0:eef_force").data.astype(np.float32)
            except Exception:
                tcp_force = np.zeros(3, dtype=np.float32)

            try:
                tcp_torque = self._data.sensor("robot0:eef_torque").data.astype(np.float32)
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

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos

        dist_to_block = np.linalg.norm(block_pos - tcp_pos)
        r_reach = (1 - np.tanh(5.0 * dist_to_block))

        is_grasped = self._check_grasp()

        r_lift = 0.0
        if is_grasped and block_pos[2] > self._z_init + 0.01:
             r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
             r_lift = np.clip(r_lift, 0, 1)

        dist_block_to_target = np.linalg.norm(block_pos[:2] - target_pos[:2])
        r_place = 0.0
        if block_pos[2] > self._z_init + 0.02:
             r_place = (1 - np.tanh(5.0 * dist_block_to_target))

        r_stack = 0.0
        target_z = target_pos[2] + self._target_cube_z + self._block_z
        if r_place > 0.95:
            dist_z = np.abs(block_pos[2] - target_z)
            r_stack = (1 - np.tanh(10.0 * dist_z))

        rew = 0.5 * r_reach + 0.3 * r_lift + 0.3 * r_place + 0.3 * r_stack

        if not self._stage_rewards["touched"]:
            if dist_to_block < 0.03:
                rew += 10.0
                self._stage_rewards["touched"] = True
                print(">>> Reward: Touched Block (+10)")

        if not self._stage_rewards["lifted"]:
            if is_grasped and block_pos[2] > self._z_init + 0.03:
                rew += 25.0
                self._stage_rewards["lifted"] = True
                print(">>> Reward: Lifted Block (+25)")

        if not self._stage_rewards["hovered"]:
            if self._stage_rewards["lifted"]:
                dist_xy_to_target = np.linalg.norm(block_pos[:2] - target_pos[:2])
                if dist_xy_to_target < 0.05:
                    rew += 25.0
                    self._stage_rewards["hovered"] = True
                    print(">>> Reward: Hovered above Goal (+25)")

        return rew
