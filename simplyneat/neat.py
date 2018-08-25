from simplyneat.config.config import Config
from simplyneat.population.population import Population
from simplyneat.breeder.breeder import Breeder
from simplyneat.genome.genome import Genome
import logging
from enum import Enum


class StatisticsTypes(Enum):
    MAX_FITNESS = 0
    MIN_FITNESS = 1
    AVERAGE_FITNESS = 2
    NUM_SPECIES = 3
    BEST_GENOME = 4
    WORST_GENOME = 5


class Neat:

    def __init__(self, config):
        logging.info("Initializing NEAT environment:")
        self._config = config
        logging.info("✔ Config set")
        self._number_of_generations = config.number_of_generations
        logging.info("✔ Number of generations set")
        self._breeder = Breeder(config)
        logging.info("✔ Breeder set")
        self._initial_genome = Genome(config)   # first organism, the minimal organism the entire population grows from
        logging.info("✔ Initial genome set")
        self._population = Population(config, genomes=[self._initial_genome])
        logging.info("✔ Initial population set")
        # dictionary of statistics, key is StatisticsType value is a list of statistics
        self._statistics = {statistic: [] for statistic in StatisticsTypes}

        logging.info("Initialized NEAT environment, preparing to run")

        self.run()

        logging.info("NEAT finished running, returning statistics dictionary")

        return self._statistics

    def run(self):
        for i in range(self._number_of_generations):
            logging.info("Generation number %s" % (i+1))
            self.__step()

    def __step(self):
        """A single iteration of NEAT's algorithm, test the entire population and get the next generation"""
        logging.info("Gathering statistics from previous generation")
        self.__add_statistics()
        logging.info("Breeding new generation")
        self._population = self._breeder.breed_population(self._population)

    def __add_statistics(self):
        """Appends to each list of statistics according to how the generation performed. 
        Assumes population.get_statistic can handle all Enum values"""
        for statistic in StatisticsTypes:
            self._statistics[statistic].append(self._population.get_statistic(statistic))
