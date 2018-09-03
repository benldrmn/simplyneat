from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder
from simplyneat.population.population import StatisticsTypes
from simplyneat.neat import Neat

if __name__ == '__main__':
    """Testing how one breeding iteration affects species, works fine"""
    config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 3, 'number_of_output_nodes': 3,
                    'compatibility_threshold': 6, 'population_size': 20, 'elite_group_size':3})
    neat = Neat(config)
    statistics = neat.run(20)
    print(statistics[StatisticsTypes.SPECIES_SIZE_HISTOGRAM])
    print(statistics[StatisticsTypes.BEST_GENOME][-1].genome_number)
    print(statistics[StatisticsTypes.MAX_FITNESS][-1])


