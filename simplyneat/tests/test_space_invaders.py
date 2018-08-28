from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder
from simplyneat.population.population import StatisticsTypes
from simplyneat.neat import Neat
import gym


def fitness(agent):
    done = False
    env = gym.make('SpaceInvaders-ram-v0')
    total_reward = 0
    observation = env.reset()

    while not done:
        next_move = agent.next_move(observation)
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
    env.close()

    config = Config({'fitness_function': fitness, 'number_of_input_nodes': len(observation), 'number_of_output_nodes': actions.n,
                    'compatibility_threshold': 6, 'verbose': True})

    neat = Neat(config)
    statistics = neat.run(20)
    print(statistics[StatisticsTypes.MAX_FITNESS])
    print(statistics[StatisticsTypes.BEST_GENOME])

