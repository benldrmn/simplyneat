from simplyneat.breeder.breeder import Breeder
from simplyneat.genome.genome import Genome
import logging

from simplyneat.population.population import Population, StatisticsTypes


class Neat:

    def __init__(self, config):
        if config.verbose:
            logging.getLogger().setLevel(logging.INFO)
        logging.info("Initializing NEAT environment:")
        self._config = config
        logging.info("✔ Config set")
        self._default_number_of_generations = config.default_number_of_generations
        logging.info("✔ Default number of generations set")
        self._breeder = Breeder(config)
        logging.info("✔ Breeder set")
        self._initial_genome = Genome(config)   # first organism, the minimal organism the entire population grows from
        logging.info("✔ Initial genome set")
        self._population = Population(config, genomes=[self._initial_genome])
        logging.info("✔ Initial population set")
        # dictionary of statistics, key is StatisticsType value is a list of statistics
        self._statistics = {statistic: [] for statistic in StatisticsTypes}

        logging.info("Initialized NEAT environment")

    def run(self, number_of_generations=0):
        # TODO: keep track of best genome of all time (every gen get best genome and compare to current best of all time)
        # TODO: pickle (BEN),
        # TODO: log some stats on the run,
        # TODO: optional: re-run for X more gen (if we decide we have more time)
        # TODO: add to config time limit and finish after the first one finishes - either time limit or num_generations
        # TODO: TESTS and EXAMPLES (look at neat-python github for examples)

        if not number_of_generations:
            number_of_generations = self._default_number_of_generations
        logging.info("Preparing to run:")
        for i in range(number_of_generations):
            logging.info("Generation number %s" % (i+1))
            self.__step()
        return self._statistics

    def __step(self):
        """A single iteration of NEAT's algorithm, test the entire population and get the next generation"""
        logging.info("Gathering statistics of current generation")
        self.__add_statistics()
        logging.info("Max fitness: %s" % str(self._statistics[StatisticsTypes.MAX_FITNESS][-1]))
        logging.info("Average fitness: %s" % str(self._statistics[StatisticsTypes.AVERAGE_FITNESS][-1]))
        logging.info("Breeding new generation")
        self._population = self._breeder.breed_population(self._population)


    def __add_statistics(self):
        """Appends to each list of statistics according to how the generation performed. 
        Assumes population.get_statistic can handle all Enum values"""
        for statistic in StatisticsTypes:
            self._statistics[statistic].append(self._population.get_statistic(statistic))
