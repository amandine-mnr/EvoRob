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
import matplotlib.pyplot as plt
ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

class AntWorld(World):
    def __init__(self):
        # self.env = gym.make(ENV_NAME)
        self.world_file = os.path.join(ROOT_DIR, 'AntEnv.xml')
        self.n_steps = 5000
        self.env = gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
        action_space = 16  # 8 actions per robot
        state_space = 56  # 28 observations per robot
        self.controller = MLP.NNController(state_space, action_space) #multi layer perceptrons
        self.n_params = self.controller.n_params  
        self.n_weights = self.controller.n_params
        self.world_file = os.path.join(ROOT_DIR, 'AntEnv.xml')

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        n_sim_steps = self.n_steps
        self.geno2pheno(genotype)

        rewards_list = []
        observations, info = self.env.reset()
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
        return np.sum(rewards_list)

def run_EA(ea, world):
    fitness_hist_max = [] 
    fitness_hist_mean = []
    for gen in range(ea.n_gen):
        print("Generation ", gen)
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        # env.reset()
        for index, genotype in enumerate(pop):
            world.geno2pheno(genotype)
            fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        best_fitness = np.max(fitnesses_gen)
        mean_fitness = np.mean(fitnesses_gen)
        fitness_hist_max.append(best_fitness)
        fitness_hist_mean.append(mean_fitness)
        ea.tell(pop, fitnesses_gen)
    # env.close()
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_hist_max, label='Best fitness per generation')
    plt.plot(fitness_hist_mean, label='Mean fitness per generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of fitness over generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def generate_best_individual_video(controller, video_name: str = 'EvoRob4_video12.mp4'):
    robot_path = os.path.join(get_project_root(), 'AntEnv.xml')
    env = gym.make(
        ENV_NAME,
        robot_path=robot_path,
        render_mode="rgb_array",
        reset_noise_scale=0.1,
        max_episode_steps=1000,
    )
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        # if terminated:
        #     break
    print(np.sum(rewards_list))

    cumulative_rewards = np.cumsum(rewards_list)

    # Plotting the fitness curve
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards, label='Cumulative reward')
    plt.xlabel('Simulation step')
    plt.ylabel('Cumulative reward')
    plt.title(f'Fitness Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()

def main():
    world = AntWorld()
    n_parameters = world.n_params

    # # TODO: improve the ES settings
    # ES_opts["min"] = -1
    # ES_opts["max"] = 1
    # ES_opts["num_parents"] = 100
    # ES_opts["num_generations"] = 100
    # ES_opts["mutation_sigma"] = .5
    population_size = 40 #250
    CMAES_opts["min"] = -10
    CMAES_opts["max"] = 10
    CMAES_opts["num_generations"] = 20
    CMAES_opts["mutation_sigma"] = 0.33

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
    # ea = ES(population_size, n_parameters, ES_opts, results_dir)
    ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    run_EA(ea, world)

    # %% Make video of best behaviour
    best_individual = np.load(os.path.join(results_dir, "19", "x_best.npy"))
    world.controller.geno2pheno(best_individual)

    generate_best_individual_video(world.controller, 'EA_best12.mp4')


if __name__ == "__main__":
    main()
