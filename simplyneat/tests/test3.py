from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder
from simplyneat.population.population import Population

if __name__ == '__main__':
    """Testing how one breeding iteration affects species, works fine"""
    config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 5, 'number_of_output_nodes': 3,
                    'compatibility_threshold': 0.01})
    genome = Genome(genome_number=0, config=config)
    breeder = Breeder(config=config)
    population = Population(config, genomes=[genome])
    new_population = breeder.breed_population(population)

    print(new_population)

    print("Best genome: %s" % new_population.best_genome)
    print("Max fitness: %s" % new_population.max_fitness)
    print("Biggest species: %s" % new_population.biggest_species)