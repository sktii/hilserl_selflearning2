from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import os
import glfw

# Prevent JAX from hogging GPU memory, allowing MuJoCo EGL to run smoothly
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"

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
from ur5e_sim.controllers.opspace import OpSpaceController
from ur5e_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

class RealRobotInterface:
    def __init__(self, ip):
        self.url = f"http://{ip}:5000"
        self.target_q = None
        self.target_g = None
        self.lock = threading.Lock()
        self.running = True
        # Start sender thread
        self.thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.thread.start()

    def update(self, q, gripper):
        """Update target data, returns immediately."""
        with self.lock:
            self.target_q = q
            self.target_g = gripper

    def _sender_loop(self):
        while self.running:
            q, g = None, None
            with self.lock:
                if self.target_q is not None:
                    q = self.target_q.tolist()
                    g = int(np.clip(self.target_g * 255, 0, 255))

            if q is not None:
                try:
                    # Send servoj command
                    requests.post(f"{self.url}/servoj", json={"q": q}, timeout=0.02)
                    requests.post(f"{self.url}/move_gripper", json={"gripper_pos": g}, timeout=0.02)
                except:
                    pass # Ignore errors to prevent lag

            time.sleep(0.01) # 100Hz

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_ur5e.xml"
_UR5E_HOME = np.asarray([0, -1.57, 1.57, -1.57, -1.57, 0])

_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])
# User requested max obstacles = 64
_MAX_OBSTACLES = 64

class UR5eStackCubeGymEnv(MujocoGymEnv, gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    POTENTIAL_SCALE = 1.0

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
        self.image_obs = image_obs

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

        print(f"[UR5eEnv] Initialized with _MAX_OBSTACLES={_MAX_OBSTACLES}")

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.env_step = 0
        self.intervened = False
        self._grasp_counter = 0

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

        # Initialize state variables
        self._z_init = 0.0
        self._floor_collision = False

        # Find physical gripper joint for accurate state reading
        # Typically "robot0:2f85:right_driver_joint" for Robotiq 2F-85
        self._gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:2f85:right_driver_joint")

        # Pre-cache pillar IDs for fast collision checking
        # Use SET for O(1) collision checks (critical for performance)
        self._pillar_geom_ids = set()
        self._pillar_info = [] # Cache for _get_obstacle_state: list of (id, type)
        # Search for all pillar geoms up to _MAX_OBSTACLES or until not found
        # Typically XML has limited number, but we scan robustly
        for i in range(1, _MAX_OBSTACLES + 1): # Scan for potential pillars
            id_cyl = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_cyl_{i}")
            if id_cyl != -1:
                self._pillar_geom_ids.add(id_cyl)
                self._pillar_info.append((id_cyl, 'cyl'))

            id_box = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_box_{i}")
            if id_box != -1:
                self._pillar_geom_ids.add(id_box)
                self._pillar_info.append((id_box, 'box'))

        # Cache block ID
        self._block_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "block")

        # Cache floor ID for non-terminal collision checks
        self._floor_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Cache gripper and robot geom IDs
        self._gripper_geom_ids = set()
        self._gripper_pad_geom_ids = set() # Specific for strict grasp detection
        self._robot_geom_ids = set()
        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body_id = self._model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if name and ("pad" in name or "finger" in name or "2f85" in name):
                self._gripper_geom_ids.add(i)

            # Identify inner pads for strict grasp logic
            # Using "pad" is safer than strict "pad1"/"pad2" based on code review
            if name and "pad" in name:
                self._gripper_pad_geom_ids.add(i)

            # Robustly identify robot parts
            if body_name and ("robot0" in body_name or "ur5e" in body_name or "2f85" in body_name):
                self._robot_geom_ids.add(i)

        # Force collision properties for all robot geoms to ensure they interact with pillars
        # Default XML might have them as visual-only (contype=0)
        for i in self._robot_geom_ids:
            self._model.geom_contype[i] = 1
            self._model.geom_conaffinity[i] = 1
            self._model.geom_solimp[i] = np.array([0.99, 0.999, 0.001, 0.5, 2])
            self._model.geom_solref[i] = np.array([0.005, 1])

        # Initialize Persistent Controller (Zero-Allocation)
        self._opspace_controller = OpSpaceController(self._model, self._ur5e_dof_ids)

        print(f"[UR5eEnv] Cached {len(self._robot_geom_ids)} Robot Geoms, {len(self._pillar_geom_ids)} Pillar Geoms.")

        # User requested to remove all image observation logic to prevent overhead
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
                            "ur5e/joint_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "ur5e/joint_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
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
            # Optimize: Force lower resolution for 'human' render on WSL to reduce X Server lag
            if self.render_mode == "human":
                if hasattr(self._viewer, 'width'):
                    self._viewer.width = 640 # Reduced from potentially high defaults
                if hasattr(self._viewer, 'height'):
                    self._viewer.height = 480
            else:
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

        self._safe_geom_ids = set() # Safe for PILLARS (static env)
        safe_names = ["block", "floor", "target_geom", "target"]
        for name in safe_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._safe_geom_ids.add(gid)
            else:
                print(f"Warning: Safe geom '{name}' not found in model.")

        self._robot_safe_geom_ids = set() # Safe for ROBOT
        robot_safe_names = ["block", "target_geom", "target"] # Floor is NOT safe for robot
        for name in robot_safe_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._robot_safe_geom_ids.add(gid)

        if self.real_robot:
            self._robot_interface = RealRobotInterface(self.real_robot_ip)
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

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        if self.real_robot:
            try:
                # Align Sim to Real Robot
                print("[Sim] Aligning Simulation to Real Robot...")
                resp = requests.post(f"http://{self.real_robot_ip}:5000/getq", timeout=1.0)
                real_q = np.array(resp.json()['q'])
                self._data.qpos[self._ur5e_dof_ids] = real_q
                print(f"[Sim] Aligned to Q: {real_q}")
            except Exception as e:
                print(f"[Sim] Failed to align to real robot: {e}")
                self._data.qpos[self._ur5e_dof_ids] = _UR5E_HOME
        else:
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
        self._cached_obstacle_state = self._compute_obstacle_state_once()

        # Capture actual initial Z of the block for lift reward logic
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + self._target_cube_z * 2

        # Initialize distances for potential normalization
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos

        self._init_dist_reach = np.linalg.norm(block_pos - tcp_pos) + 1e-6
        self._init_dist_move = np.linalg.norm(block_pos - target_pos) + 1e-6

        self._grasp_counter = 0

        # Initialize previous potential
        self._prev_potential, self._latest_potentials = self._calculate_potential(block_pos, tcp_pos, target_pos, False)

        self._floor_collision = False
        self.env_step = 0
        self.episode_reward = 0.0
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
    def _compute_obstacle_state_once(self):
            obs_state = np.zeros((_MAX_OBSTACLES, 7), dtype=np.float32)
            idx = 0
            for gid, ptype in self._pillar_info:
                if idx >= _MAX_OBSTACLES:
                    break

                # 直接讀取 MuJoCo 數據
                pos = self._model.geom_pos[gid]
                size = self._model.geom_size[gid]

                if ptype == 'cyl':
                    obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[0], size[1]]
                else:
                    obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[1], size[2]]
                idx += 1
            return obs_state.flatten()
    def _get_obstacle_state(self):
        if not hasattr(self, '_cached_obstacle_state'):
            self._cached_obstacle_state = self._compute_obstacle_state_once()
        return self._cached_obstacle_state

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

                # Enforce collision properties to prevent passthrough
                self._model.geom_contype[body_id] = 1
                self._model.geom_conaffinity[body_id] = 1
                # Relaxed solimp for better solver stability on WSL
                self._model.geom_solimp[body_id] = np.array([0.95, 0.99, 0.001, 0.5, 2])
                self._model.geom_solref[body_id] = np.array([0.005, 1])
                self._model.geom_margin[body_id] = 0.005 # 5mm margin to prevent visual penetration

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

                # Enforce collision properties to prevent passthrough
                self._model.geom_contype[body_id] = 1
                self._model.geom_conaffinity[body_id] = 1
                # Relaxed solimp for better solver stability on WSL
                self._model.geom_solimp[body_id] = np.array([0.95, 0.99, 0.001, 0.5, 2])
                self._model.geom_solref[body_id] = np.array([0.005, 1])
                self._model.geom_margin[body_id] = 0.005 # 5mm margin to prevent visual penetration

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
                # Use cached gripper PAD IDs for strict grasp (inner pads only)
                if other_id in self._gripper_pad_geom_ids:
                    has_contact = True
                    break

        # Hysteresis Logic (Fix Flicker)
        if has_contact:
            self._grasp_counter = 5
            return True
        else:
            if self._grasp_counter > 0:
                self._grasp_counter -= 1
                return True
            else:
                return False

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        self._floor_collision = False # Reset collision flag

        if len(self._action_scale) == 3:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[2]
        else:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[1]

        x, y, z, grasp = action

        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * trans_scale

        # [Fix] Add safety margin (e.g., 5mm) to prevent solver fighting at boundaries
        margin = 0.005
        bounds_low = _CARTESIAN_BOUNDS[0] + margin
        bounds_high = _CARTESIAN_BOUNDS[1] - margin
        npos = np.clip(pos + dpos, bounds_low, bounds_high)

        # [Fix] Enforce minimum Z height to avoid floor penetration fighting
        if npos[2] < 0.02:
             npos[2] = 0.02

        self._data.mocap_pos[0] = npos

        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * grasp_scale
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        # Timing vars
        t_ctrl = 0.0
        t_physics = 0.0

        for i in range(self._n_substeps):
            if i % 2 == 0:
                t_c0 = time.time()
                tau = self._opspace_controller(
                    model=self._model,
                    data=self._data,
                    site_id=self._pinch_site_id,
                    pos=self._data.mocap_pos[0],
                    ori=self._data.mocap_quat[0],
                    joint=_UR5E_HOME,
                    gravity_comp=True,
                    pos_gains=(400.0, 400.0, 400.0),
                    damping_ratio=4
                )
                t_ctrl += time.time() - t_c0
            self._data.ctrl[self._ur5e_ctrl_ids] = tau

            t_p0 = time.time()
            mujoco.mj_step(self._model, self._data)
            t_physics += time.time() - t_p0

            # Optimize: Fail fast on collision to prevent solver explosion/lag
            if self._check_collision():
                break

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

        t_poll = 0.0
        t_draw = 0.0

        if self.render_mode == "human" and self._viewer:
            # 0. Explicitly disable VSync to reduce blocking time on WSL X Server (Lazy Init)
            if not getattr(self, '_vsync_set', False):
                if glfw.get_current_context():
                    glfw.swap_interval(0)
                    self._vsync_set = True

            # 1. Always poll events to keep window responsive and prevent queue flooding
            t_p0 = time.time()
            glfw.poll_events()
            t_poll = time.time() - t_p0

            # 2. Throttle rendering to max 20Hz (50ms) to prevent VSync/SwapBuffers blocking the physics loop
            # Initialize last_render_time if not present
            if not hasattr(self, '_last_render_time'):
                self._last_render_time = 0.0

            curr_time = time.time()
            # Dynamic throttling: Limit render FPS to self.hz (18) to match simulation speed
            # and prevent X Server queue flooding.
            target_render_dt = 1.0 / self.hz
            if curr_time - self._last_render_time > target_render_dt:
                t_d0 = time.time()
                self._viewer.render(self.render_mode)
                t_draw = time.time() - t_d0
                self._last_render_time = time.time()

        t_sleep = 0.0
        total_time_ms = (time.time() - start_time) * 1000

        if self.render_mode == "human" or self.real_robot:
            dt = time.time() - start_time
            target_dt = 1.0 / self.hz
            sleep_time = max(0, target_dt - dt)
            if sleep_time > 0:
                t_s0 = time.time()
                time.sleep(sleep_time)
                t_sleep = time.time() - t_s0

        # Log if slow (>100ms)
        final_total = (time.time() - start_time) * 1000
        if final_total > 100:
            logging.warning(
                f"[LAG DETECTED] Total={final_total:.1f}ms | "
                f"Phys={t_physics*1000:.1f}ms | "
                f"Ctrl={t_ctrl*1000:.1f}ms | "
                f"Poll={t_poll*1000:.1f}ms | "
                f"Draw={t_draw*1000:.1f}ms | "
                f"Sleep={t_sleep*1000:.1f}ms"
            )

        if self.real_robot:
            try:
                sim_q = self._data.qpos[self._ur5e_dof_ids]
                sim_g = self._data.ctrl[self._gripper_ctrl_id] / 255.0
                self._robot_interface.update(sim_q, sim_g)
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
            terminated = True
            rew = -0.05
            success = False
            self.success_counter = 0
            return obs, rew, terminated, False, info

        # Non-fatal floor penalty
        if self._floor_collision:
            rew -= 0.005

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
            rew += 1.0

        self.episode_reward += rew
        if terminated:
            # Breakdown: Scale potentials by POTENTIAL_SCALE (200.0)
            p_reach, p_grasp, p_lift, p_move = self._latest_potentials
            print(f"\nEpisode Finished.")
            print(f"Total Reward: {self.episode_reward:.2f}")
            print(f"Breakdown:")
            print(f"  Reach: {p_reach * self.POTENTIAL_SCALE:.1f} / {1.0 * self.POTENTIAL_SCALE:.1f}")
            print(f"  Grasp: {p_grasp * self.POTENTIAL_SCALE:.1f} / {1.0 * self.POTENTIAL_SCALE:.1f}")
            print(f"  Lift:  {p_lift * self.POTENTIAL_SCALE:.1f} / {1.0 * self.POTENTIAL_SCALE:.1f}")
            print(f"  Move:  {p_move * self.POTENTIAL_SCALE:.1f} / {2.0 * self.POTENTIAL_SCALE:.1f}")
            print(f"  Success: {info['succeed']} (+100 if true)")

        return obs, rew, terminated, False, info

    def _check_collision(self):
        if self._data.ncon == 0:
            return False

        # [Optimization] Limit contact checks to avoid O(N) loop lag
        check_limit = min(self._data.ncon, 50)

        for i in range(check_limit):
            contact = self._data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            # Explicit check: Robot vs Pillar (Forbidden Region)
            if (g1 in self._robot_geom_ids and g2 in self._pillar_geom_ids) or \
               (g2 in self._robot_geom_ids and g1 in self._pillar_geom_ids):
                print(f"Collision detected: Robot hit Pillar (Forbidden Region)")
                return True

            # 1. Pillar Collisions
            is_g1_pillar = g1 in self._pillar_geom_ids
            is_g2_pillar = g2 in self._pillar_geom_ids

            if is_g1_pillar or is_g2_pillar:
                other_id = g2 if is_g1_pillar else g1
                if other_id not in self._safe_geom_ids:
                    return True

            # 2. Robot Collisions (including Floor)
            is_g1_robot = g1 in self._robot_geom_ids
            is_g2_robot = g2 in self._robot_geom_ids

            if is_g1_robot or is_g2_robot:
                other_id = g2 if is_g1_robot else g1

                # Treat FLOOR collision as non-fatal warning (fix "hands up")
                if other_id == self._floor_geom_id:
                    self._floor_collision = True
                    continue

                # Collision if other is NOT safe (e.g. pillar) AND NOT part of robot (self-collision)
                if other_id not in self._robot_safe_geom_ids and other_id not in self._robot_geom_ids:
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

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["ur5e/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["ur5e/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["ur5e/gripper_pos"] = gripper_pos

        # Add joint state
        obs["state"]["ur5e/joint_pos"] = self._data.qpos[self._ur5e_dof_ids].astype(np.float32)
        obs["state"]["ur5e/joint_vel"] = self._data.qvel[self._ur5e_dof_ids].astype(np.float32)

        target_pos = self._data.body("target_cube").xpos.astype(np.float32)
        obs["state"]["target_cube_pos"] = target_pos

        # Removed image observation logic to prevent overhead
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        obs["state"]["block_pos"] = block_pos
        obs["state"]["obstacle_state"] = self._get_obstacle_state()

        return obs

    def _is_block_placed(self, block_pos, target_pos):
        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.04
        # Relax Z threshold to 0.5 to prevent flickering "placed" state
        z_success = block_pos[2] > (target_pos[2] + self._target_cube_z + self._block_z * 0.5)
        return xy_success and z_success

    def _calculate_potential(self, block_pos, tcp_pos, target_pos, is_grasped):
        # 1. Reach Potential
        dist_reach = np.linalg.norm(block_pos - tcp_pos)
        # Normalize: 1.0 when close, 0.0 when at start distance (or further)
        # Using tanh to smoothly saturate
        phi_reach = 1 - np.tanh(5.0 * dist_reach / self._init_dist_reach)

        # Determine effective grasp: Real grasp OR Success state (placed)
        is_placed = self._is_block_placed(block_pos, target_pos)

        # If placed, we force maximal potentials to represent "Task Complete"
        if is_placed:
            phi_reach = 1.0
            phi_move = 1.0
            phi_lift = 1.0
            effective_grasp = 1.0
        else:
            effective_grasp = 1.0 if is_grasped else 0.0
            phi_move = 0.0
            phi_lift = 0.0

            # 2. Lift Potential (only if grasped)
            if effective_grasp > 0.5:
                # Lift target: ~5cm above initial height
                dist_lift = max(0.0, block_pos[2] - self._z_init)
                # Saturate at 5cm using tanh
                phi_lift = np.tanh(60.0 * dist_lift)

                # 3. Move Potential (only if lifted > 2cm)
                if dist_lift > 0.02:
                     dist_move = np.linalg.norm(block_pos - target_pos)
                     phi_move = 1 - np.tanh(5.0 * dist_move / self._init_dist_move)

        # Total Potential (Normalized to 1.0)
        # Weights: Reach=0.1, Grasp=0.1, Lift=0.2, Move=0.6
        w_reach = 0.1
        w_grasp = 0.1
        w_lift = 0.2
        w_move = 0.6

        potential = w_reach * phi_reach + effective_grasp * (w_grasp + w_lift * phi_lift + w_move * phi_move)
        return potential, (w_reach * phi_reach, effective_grasp * w_grasp, effective_grasp * w_lift * phi_lift, effective_grasp * w_move * phi_move)

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos
        is_grasped = self._check_grasp()

        current_potential, potentials = self._calculate_potential(block_pos, tcp_pos, target_pos, is_grasped)
        self._latest_potentials = potentials

        # Step reward is difference in potential
        step_rew = (current_potential - self._prev_potential) * self.POTENTIAL_SCALE

        # Update previous potential
        self._prev_potential = current_potential

        return step_rew
