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
        action_space = 16  # 8 actions per robot
        state_space = 56  # 28 observations per robot

        self.n_repeats = 3 #3
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params
        self.n_params = self.n_weights
        self.world_file = os.path.join(ROOT_DIR, 'AntEnv.xml')

        # Prepare the XML only once during initialization
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', 'ant_world.xml'))
        robot_env = world.getroot()
        robot_env.append(xml.Element('include', attrib={'file': 'AntRobot.xml'}))
        robot_env.append(xml.Element('include', attrib={'file': 'AntRobot2.xml'}))
        world_xml = xml.tostring(robot_env, encoding='unicode')
        with open(self.world_file, 'w') as f:
            f.write(world_xml)

    def evaluate_individual(self, genotype):
        self.controller.geno2pheno(genotype) #load the NN weights

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
            # print("observation : ", observations.shape)

            rewards_full[step, done_mask == False] = rewards[done_mask == False]
            done_mask = done_mask | dones | truncated

            if np.all(done_mask):
                break

        final_rewards = np.sum(rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards)

    def evaluate_individual_multi(self, genotype):
        
        self.controller.geno2pheno(genotype) #load the NN weights

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
        multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))  

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)

        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)
            rewards_full[step, done_mask == False] = rewards[done_mask == False]

            multi_obj_reward = np.array([infos['vel_x'], -infos['inter_distance']]).T  
            multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward[done_mask == False]

            done_mask = done_mask | dones | truncated
            if np.all(done_mask):
                break

        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)

    def visualize_pareto_front(self, pareto_front, fitnesses, output_file='pareto_front3.png'):
        plt.figure()
        distances = [fit[0] for fit in fitnesses]  
        separations = [fit[1] for fit in fitnesses]  
        plt.scatter(separations, distances, color='blue')
        plt.xlabel('Separation') #(lower is better)
        plt.ylabel('Velocity') #(higher is better)
        plt.title('Pareto Front: Velocity vs Separation')
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()

########################
def run_EA_single(ea_single, world):
    best_individual = None
    best_fitness = -np.inf

    for gen in range(ea_single.n_gen):
        print("generation ", gen)
        pop = ea_single.ask()
        fitnesses_gen = np.array([world.evaluate_individual(ind) for ind in pop])

        max_fitness_idx = np.argmax(fitnesses_gen)
        if fitnesses_gen[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses_gen[max_fitness_idx]
            best_individual = pop[max_fitness_idx]

        ea_single.tell(pop, fitnesses_gen)

    return best_individual, best_fitness

def generate_best_individual_video(world, best_individual, video_name: str = 'EvoRob4_video11.mp4'):
    world.controller.geno2pheno(best_individual)
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(5000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break

    print(f"Total reward of the best individual: {np.sum(rewards_list)}")

    import imageio
    imageio.mimsave(video_name, frames, fps=30)
    env.close()

def visualise_individual(genotype):
    world = AntWorld()

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(2000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            print("termination")
            break
    env.close()
    print(np.sum(rewards_list))

def run_EA_multi(ea_multi, world):
    pareto_front = []
    pareto_fitnesses = []

    for gen in range(ea_multi.n_gen):
        print(f"Generation {gen}")
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual_multi(genotype)
            fitnesses_gen[index] = fit_ind

        ea_multi.tell(pop, fitnesses_gen)

        # Update Pareto front
        for idx, ind in enumerate(ea_multi.x):            
            pareto_front.append(ind)
            pareto_fitnesses.append(fitnesses_gen[idx])

    return pareto_front, pareto_fitnesses

def main():
    world = AntWorld()
    n_parameters = world.n_params

    # genotype = np.random.uniform(-1, 1, (56*56+56*16))  # 8 body parameters, 945 NN weights
    genotype = np.random.uniform(-1, 1, n_parameters)  
    visualise_individual(genotype)
    world = AntWorld()
    n_parameters = world.n_params
    # print("n_parameters : ", n_parameters)

    population_size = 30 #250
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_generations"] = 10
    CMAES_opts["mutation_sigma"] = 0.33
    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
    ea_single = CMAES(population_size, n_parameters, CMAES_opts, results_dir)
    best_individual, best_fitness = run_EA_single(ea_single, world)
    print(f"Best fitness achieved: {best_fitness}")
    generate_best_individual_video(world, best_individual)
    
    # population_size = 20
    # NSGA_opts["min"] = -1
    # NSGA_opts["max"] = 1
    # NSGA_opts["num_parents"] = population_size
    # NSGA_opts["num_generations"] = 20
    # NSGA_opts["mutation_prob"] = 0.3
    # NSGA_opts["crossover_prob"] = 0.5

    # results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'multi')
    # ea_multi_obj = NSGAII_sol(population_size, n_parameters, NSGA_opts, results_dir)

    # pareto_front, pareto_fitnesses = run_EA_multi(ea_multi_obj, world)
    # world.visualize_pareto_front(pareto_front, pareto_fitnesses)
    # best_individual = np.load(os.path.join(results_dir, "19", "x_best.npy"))
    # generate_best_individual_video(world, best_individual)

    # print("Multi-objective optimization and video generation complete.")

if __name__ == "__main__":
    main()
