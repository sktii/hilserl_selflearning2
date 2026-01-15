
import sys
import os
import numpy as np
import time
import cProfile
import pstats
import io

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Set threading limits (keep previous fix)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Unset MUJOCO_GL (keep previous fix)
if os.environ.get("MUJOCO_GL") == "egl":
    del os.environ["MUJOCO_GL"]

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv
from examples_UR5e.experiments.stack_cube_sim.config import EnvConfig

def profile_run():
    print("Initializing environment for profiling...")
    # Use human render mode to mimic user scenario
    env = UR5eStackCubeGymEnv(render_mode="human", image_obs=False, hz=12, config=EnvConfig())
    obs, info = env.reset()

    print("Starting profiling loop...")

    # We will profile a sequence of steps
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()

    # Run for 200 steps (enough to trigger lag if it's cumulative or boundary related)
    for i in range(200):
        t0 = time.time()

        # Action: oscillating movement to hit boundaries potentially
        # Sweep X and Y
        action = np.zeros(env.action_space.sample().shape)
        action[0] = np.sin(i * 0.1) # X sweep
        action[1] = np.cos(i * 0.1) # Y sweep
        action[2] = -0.5 # Push down slightly

        next_obs, rew, done, truncated, info = env.step(action)

        # Mimic render (env.step already calls it if human, but let's be sure)
        # env.render()

        dt = time.time() - t0

        # Log slow frames (> 50ms)
        if dt > 0.05:
            print(f"[LAG] Step {i}: {dt*1000:.2f}ms. Contacts: {env._data.ncon}")

        if done:
            env.reset()

    pr.disable()
    print(f"Total time: {time.time() - start_time:.2f}s")

    # Save stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30) # Top 30 consumers
    print(s.getvalue())

    env.close()

if __name__ == "__main__":
    profile_run()
