from simplyneat.config.config import Config
from simplyneat.neat import Neat
from simplyneat.tests.test2 import foofoo


def fit(x):
    return 1

if __name__ == '__main__':
    neat = Neat(Config({'fitness_function': foofoo, 'number_of_input_nodes': 256, 'number_of_output_nodes': 6,
                        'population_size': 3, 'change_weight_probability': 1, 'processes_in_pool':4}))
    print(neat.run(2))
