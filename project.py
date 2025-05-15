from src.EA.CMAES_sol import CMAES, CMAES_opts
from src.EA.NSGA_sol import NSGAII_sol, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os


ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

class AntWorld(World):
    def __init__(self):
        action_space = 16  # 8 actions per robot
        state_space = 56  # 27 observations per robot

        self.n_repeats = 3
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params
        self.n_params = self.n_weights
        self.world_file = os.path.join(ROOT_DIR, 'AntEnv.xml')

    def evaluate_individual(self, genotype):
        # Load the NN weights
        self.controller.geno2pheno(genotype)

        # Define the environment including both robots
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', 'ant_world.xml'))
        robot_env = world.getroot()

        robot_env.append(xml.Element('include', attrib={'file': 'AntRobot.xml'}))
        robot_env.append(xml.Element('include', attrib={'file': 'AntRobot2.xml'}))
        world_xml = xml.tostring(robot_env, encoding='unicode') #write the final world file

        with open(self.world_file, 'w') as f:
            f.write(world_xml)

        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)

        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)
            
            # Store rewards for active environments only
            rewards_full[step, done_mask == False] = rewards[done_mask == False]
            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break

        final_rewards = np.sum(rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards)
    
# def run_EA_single(ea_single, world):
#     for gen in range(ea_single.n_gen):
#         pop = ea_single.ask()
#         fitnesses_gen = np.empty(len(pop))
#         for index, genotype in enumerate(pop):
#             fit_ind, _ = world.evaluate_individual(genotype)
#             fitnesses_gen[index] = fit_ind
#         ea_single.tell(pop, fitnesses_gen)
def run_EA_single(ea_single, world):
    for gen in range(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.array([world.evaluate_individual(ind) for ind in pop])
        ea_single.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()
    
def visualise_individual(genotype):
    world = AntWorld()
    # robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    # robot.xml = robot.define_robot()
    # robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    robot_env.append(xml.Element("include", attrib={"file": "AntRobot2.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(1000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    print(np.sum(rewards_list))

def main():
    # %% Understanding the world
    genotype = np.random.uniform(-1, 1, 953)  # 8 body parameters, 945 NN weights
    visualise_individual(genotype)

    # %% Optimise single-objective
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 250
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 100
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 0.33

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
    ea_single = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    run_EA_single(ea_single, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    # best_individual = np.load(os.path.join(results_dir, "99", "x_best.npy"))

    # points, connectivity_mat = world.geno2pheno(best_individual)
    # robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    # robot.xml = robot.define_robot()
    # robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    robot_env.append(xml.Element("include", attrib={"file": "AntRobot2.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world)


if __name__ == "__main__":
    main()


