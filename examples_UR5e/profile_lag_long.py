
import sys
import os
import numpy as np
import time
import cProfile
import pstats
import io
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Set threading limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

if os.environ.get("MUJOCO_GL") == "egl":
    del os.environ["MUJOCO_GL"]

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv
from examples_UR5e.experiments.stack_cube_sim.config import EnvConfig

def profile_run_long():
    print("Initializing environment for LONG profiling...")
    # Set HZ to 1000 to effectively disable the sleep logic in step()
    # step() logic: target_dt = 1.0 / self.hz. If hz=1000, target_dt=1ms. Physics takes ~0.14ms.
    # So sleep will be minimal/zero if overhead > 1ms.
    env = UR5eStackCubeGymEnv(render_mode="human", image_obs=False, hz=1000, config=EnvConfig())
    obs, info = env.reset()

    print("Starting profiling loop (1000 steps)...")

    times = []

    start_time = time.time()

    for i in range(1000):
        t0 = time.time()

        # Action: random walk
        action = env.action_space.sample()
        next_obs, rew, done, truncated, info = env.step(action)

        dt = time.time() - t0
        times.append(dt)

        if done:
            env.reset()

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

    # Analyze trends
    times = np.array(times) * 1000 # to ms
    avg = np.mean(times)
    std = np.std(times)

    print(f"Step Time Stats: Mean={avg:.2f}ms, Std={std:.2f}ms, Max={np.max(times):.2f}ms, Min={np.min(times):.2f}ms")

    # Check for trend
    first_half = np.mean(times[:500])
    second_half = np.mean(times[500:])
    print(f"First Half Mean: {first_half:.2f}ms")
    print(f"Second Half Mean: {second_half:.2f}ms")

    if second_half > first_half * 1.1:
        print("!! DETECTED SLOWDOWN: Second half is >10% slower than first half.")
    else:
        print("No significant slowdown detected.")

    env.close()

if __name__ == "__main__":
    profile_run_long()
