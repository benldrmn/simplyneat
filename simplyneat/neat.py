from simplyneat.breeder.breeder import Breeder
from simplyneat.genome.genome import Genome
import logging

from simplyneat.population.population import Population, StatisticsTypes


class Neat:

    def __init__(self, config):
        logging.info("Initializing NEAT environment:")
        self._config = config
        logging.info("Config set")
        self._breeder = Breeder(config)
        logging.info("Breeder set")
        self._initial_genome = Genome(config)   # first organism, the minimal organism the entire population grows from
        logging.info("Initial genome set")
        self._population = Population(config, genomes=[self._initial_genome])
        logging.info("Initial population set")
        # dictionary of statistics, key is StatisticsType value is a list of statistics
        self._statistics = {statistic: [] for statistic in StatisticsTypes}

        self._best_genome = Genome(self._config)
        self._best_genome_fitness = self._best_genome.fitness

        logging.info("Initialized NEAT environment")

    def run(self, number_of_generations=0):
        logging.info("Running for %s generations" % str(number_of_generations))
        for i in range(number_of_generations):
            logging.info("Generation number %s" % i)
            self._step()

        return self._statistics, self._best_genome

    def _step(self):
        """A single iteration of NEAT's algorithm, test the entire population and get the next generation"""
        self._population = self._breeder.breed_population(self._population)
        self._add_statistics()
        self._log_statistics()
        self._update_best_genome()

    def _add_statistics(self):
        """Appends to each list of statistics according to how the generation performed. 
        Assumes population.get_statistic can handle all Enum values"""
        for statistic in StatisticsTypes:
            self._statistics[statistic].append(self._population.get_statistic(statistic))

    def _log_statistics(self):
        logging.info("Logging population statistics")
        for statistic in StatisticsTypes:
            logging.info("%s: %s" % (statistic, self._population.get_statistic(statistic)))

    def _update_best_genome(self):
        best_genome = self._population.best_genome
        best_fitness = best_genome.fitness
        if best_fitness > self._best_genome_fitness or \
                (best_fitness == self._best_genome_fitness and best_genome.size < self._best_genome.size):
            self._best_genome = best_genome
            self._best_genome_fitness = best_genome.fitness
