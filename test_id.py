import gymnasium as gym
import numpy as np
from src.world.World import World
import os
from src.utils.Filesys import get_project_root

ROOT_DIR = get_project_root()

class AntWorld(World):
    def __init__(self):
        action_space = 16  # 8 actions per robot
        state_space = 56  # 27 observations per robot

        self.n_repeats = 3
        self.n_steps = 1000
        # self.controller = MLP.NNController(state_space, action_space)
        # self.n_weights = self.controller.n_params
        # self.n_params = self.n_weights
        self.world_file = os.path.join(ROOT_DIR, 'AntEnv.xml')

# env = gym.make('Ant_custom', robot_path='path/to/your/xml', render_mode="human")
ENV_NAME = 'Ant_custom'
world = AntWorld()
env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")

raw_env = env.unwrapped

print("Bodies in MuJoCo model:")
for i, name in enumerate(raw_env.sim.model.body_names):
    print(f"  ID {i}: {name}")

print("\nJoints in MuJoCo model:")
for i, name in enumerate(raw_env.sim.model.joint_names):
    print(f"  ID {i}: {name}")

print("\nActuators in MuJoCo model:")
for i, name in enumerate(raw_env.sim.model.actuator_names):
    print(f"  ID {i}: {name}")

env.close()