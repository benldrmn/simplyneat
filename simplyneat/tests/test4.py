from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder
from simplyneat.population.population import StatisticsTypes
from simplyneat.neat import Neat

if __name__ == '__main__':
    """Testing how one breeding iteration affects species, works fine"""
    config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 5, 'number_of_output_nodes': 3,
                    'compatibility_threshold': 6})
    neat = Neat(config)
    statistics = neat.run(20)
    print(statistics[StatisticsTypes.SPECIES_SIZE_HISTOGRAM])
