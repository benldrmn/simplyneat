import pickle
import time

from simplyneat import *

import gym

from simplyneat.agent.agent import Agent, _NumpyAgent
from simplyneat.config.config import Config, LoggingLevel
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genome import Genome

from simplyneat.neat import Neat
from simplyneat.population.population import StatisticsTypes



def fitness(agent):
    done = False
    env = gym.make('SpaceInvaders-ram-v0')
    total_reward = 0
    observation = env.reset()

    with agent as a:
        while not done:
            next_move = a.next_move(observation)
            observation, reward, done, info = env.step(next_move)
            total_reward += reward

        env.close()
        return total_reward

if __name__ == '__main__':
    """Testing Neat on space invaders"""
    env = gym.make('SpaceInvaders-ram-v0')
    total_reward = 0
    observation = env.reset()
    actions = env.action_space

    config = Config({'fitness_function': fitness, 'number_of_input_nodes': len(observation), 'number_of_output_nodes': actions.n,
                     'population_size':500, 'logging_level': LoggingLevel.DEBUG,
                     'elite_group_size': 3})

    neat = Neat(config)
    start = time.time()
    statistics = neat.run(50)
    print("RUNTIME: " + str(time.time() - start))
    print(statistics[StatisticsTypes.MAX_FITNESS])
    print(statistics[StatisticsTypes.BEST_GENOME])
    # done = False
    # while not done:
    #     with Agent(config, statistics[StatisticsTypes.BEST_GENOME][-1]) as a:
    #         env.render()
    #         next_move = a.next_move(observation)
    #         observation, reward, done, info = env.step(next_move)
    statistics[StatisticsTypes.BEST_GENOME][-1].save_agent()

    # with open("C:\\Users\\benld\PycharmProjects\\simplyneat\\simplyneat\\tests\\temp.p", "rb") as f:
    #     n = _NumpyAgent(config, Genome(config, pickle.load(f)))
    # done = False
    # env = gym.make('SpaceInvaders-ram-v0')
    # total_reward = 0
    # r = 0
    # observation = env.reset()
    # while not done:
    #     env.render()
    #     next_move = n.next_move(observation)
    #     observation, reward, done, info = env.step(next_move)
    #     total_reward += reward + 1
    #     r +=reward
    #     if reward != 0:
    #         print(reward)
    # env.close()
    # print(total_reward)
    # print(reward)
