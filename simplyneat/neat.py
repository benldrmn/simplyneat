from simplyneat.breeder.breeder import Breeder
from simplyneat.genome.genome import Genome
import logging

from simplyneat.population.population import Population, StatisticsTypes


class Neat:

    def __init__(self, config):
        #TODO: have it configurable
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

        logging.info("Initialized NEAT environment")

    def run(self, number_of_generations=0):
        # TODO: keep track of best genome of all time (every gen get best genome and compare to current best of all time)
        # TODO: pickle (BEN),
        # TODO: log some stats on the run,

        logging.info("Running for %s generations" % str(number_of_generations))
        for i in range(number_of_generations):
            logging.info("Generation number %s" % (i+1))
            self._step()
        return self._statistics

    def _step(self):
        """A single iteration of NEAT's algorithm, test the entire population and get the next generation"""
        self._population = self._breeder.breed_population(self._population)
        self._add_statistics()

    def _add_statistics(self):
        """Appends to each list of statistics according to how the generation performed. 
        Assumes population.get_statistic can handle all Enum values"""
        for statistic in StatisticsTypes:
            self._statistics[statistic].append(self._population.get_statistic(statistic))
