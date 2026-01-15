# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import sys
sys.path.insert(0, '../../../')
import os

# Fix for WSL/Lag: Limit NumPy/OpenMP threading to prevent explosion during opspace control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Fix for WSL/Lag: Unset MUJOCO_GL=egl if detected, to allow windowed rendering (GLFW)
if os.environ.get("MUJOCO_GL") == "egl":
    print("Pre-emptive fix: Unsetting MUJOCO_GL=egl to allow windowed rendering in record_demos.py")
    del os.environ["MUJOCO_GL"]

# Force JAX to use CPU to avoid GPU/GLFW conflicts in WSL
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Prevent JAX from hogging GPU memory, allowing MuJoCo EGL to run smoothly
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import gc

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # Disable classifier to rely on simulation ground truth success for demo recording
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    #pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    step_count = 0

    # Disable automatic GC to prevent stuttering/accumulation during episode
    # NOTE: User reported "reset to flow running" behavior, which implies GC pauses might be the "flow".
    # But "accumulating" implies memory growth.
    # We keep gc.disable() because it is best practice for high-freq loops.
    # gc.disable()

    # Recursive function to force deep copy of numpy arrays (detaching from MuJoCo views)
    def force_copy(obj):
        if isinstance(obj, np.ndarray):
            return obj.copy()
        elif isinstance(obj, dict):
            return {k: force_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [force_copy(v) for v in obj]
        return obj

    while success_count < success_needed:
        step_count += 1
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # Use force_copy to ensure we don't hold references to MuJoCo memory views
        transition = {
            "observations": force_copy(obs),
            "actions": force_copy(actions),
            "next_observations": force_copy(next_obs),
            "rewards": rew,
            "masks": 1.0 - done,
            "dones": done,
            "infos": force_copy(info),
        }

        trajectory.append(transition)
        
        # if step_count % 20 == 0:
        #     pbar.set_description(f"Return: {returns:.2f}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    # Trajectory items are already copied, just append
                    transitions.append(transition)
                success_count += 1
                #pbar.update(1)

            # Explicitly clear trajectory to free memory
            del trajectory[:]
            trajectory = []
            returns = 0

            # Manually run GC between episodes when it's safe
            gc.collect()

            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)