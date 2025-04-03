import matplotlib.pyplot as plt
import numpy as np
import os

from src.EA.ES_sol import ES, ES_opts
from src.world.World import World
from src.world.envs.TestFunctions import f_reversed_ackley, f_rosenbrock
from src.utils.Filesys import get_project_root

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise0 warm-up: This exercise is a warm-up to understanding the flow of information in the software. 
    In the previous exercise you built your own Evolutionary Strategy which will be used to optimise the parameters
    in the reversed Ackley environment.
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'InverseAckley'


# %% Q0.1
# TODO: understanding the world
class AckleyWorld(World):
    def __init__(self):
        self.n_params = 2

    def geno2pheno(self, genotype):
        x, y = genotype
        return np.array([x, y])

    def evaluate_individual(self, genotype):
        x, y = self.geno2pheno(genotype)
        fitness = f_reversed_ackley(x, y)
        return fitness


class MyWorld(World):
    def __init__(self):
        self.n_params = 2

    def geno2pheno(self, genotype):
        x, y = genotype
        return np.array([x, y])

    def evaluate_individual(self, genotype):
        x, y = self.geno2pheno(genotype)
        fitness = f_rosenbrock(x, y)
        return fitness


def run_EA(ea, world):
    # TODO: Understand the EA ask/tell interface.
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        for index, individual in enumerate(pop):
            fit_ind = world.evaluate_individual(individual)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)


def main():
    # %% Q0.1
    world = AckleyWorld()
    n_parameters = world.n_params  # only x,y params are optimised

    # %% Q0.2
    # TODO: Take your previous exercise code and add it to the ES python class
    ES_opts["min"] = -4
    ES_opts["max"] = 4
    ES_opts["num_parents"] = 100
    ES_opts["num_generations"] = 100
    ES_opts["mutation_sigma"] = 2.5
    population_size = ES_opts["num_parents"]
    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'ES')
    ea = ES(100, n_parameters, ES_opts, results_dir)

    # %% Optimise
    run_EA(ea, world)

    # %% Report results
    # TODO: Load the results and make a fitness curve plot.
    fitnesses_full = np.load(os.path.join(results_dir, 'full_f.npy'))

    mean_f = np.mean(fitnesses_full, axis=1)
    std_f = np.std(fitnesses_full, axis=1)
    gens = np.arange(0, 100, 1)
    plt.plot(gens, mean_f, color='r')
    plt.fill_between(gens, mean_f - std_f, mean_f + std_f, alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig('Ackley_f.pdf')
    plt.close()

    # %% Change the World
    # TODO: Implement your world
    myworld = MyWorld()
    n_parameters = myworld.n_params

    ea = ES(100, n_parameters, ES_opts, results_dir)
    run_EA(ea, myworld)

    # %% Change the EA
    # TODO: Change the ea function
    from src.EA.CMAES_sol import CMAES, CMAES_opts
    CMAES_opts["min"] = -4
    CMAES_opts["max"] = 4
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 0.3
    results_dir_cmaes = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
    ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir_cmaes)

    run_EA(ea, myworld)

    fitnesses_es = np.load(os.path.join(results_dir, 'full_f.npy'))
    fitnesses_cmaes = np.load(os.path.join(results_dir_cmaes, 'full_f.npy'))

    mean_f_es = np.mean(fitnesses_es, axis=1)
    mean_f_cmaes = np.mean(fitnesses_cmaes, axis=1)

    std_f_es = np.std(fitnesses_es, axis=1)
    std_f_cmaes = np.std(fitnesses_cmaes, axis=1)

    gens = np.arange(0, 100, 1)
    plt.plot(gens, mean_f_es, color='k', label='ES')
    plt.plot(gens, mean_f_cmaes, color='r', label='CMAES')
    plt.fill_between(gens, mean_f_es - std_f_es, mean_f_es + std_f_es, color='k', alpha=0.5)
    plt.fill_between(gens, mean_f_cmaes - std_f_cmaes, mean_f_cmaes + std_f_cmaes, color='r', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig('my_world_f.pdf')


if __name__ == '__main__':
    main()
