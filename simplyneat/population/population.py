import logging
import random

from simplyneat.genome.genome import compatibility_distance
from simplyneat.statistics import StatisticsTypes
from simplyneat.species.species import Species
import numpy as np


class Population:

    def __init__(self, config, genomes=None, species=None):
        """Builds the population according to a list of genomes and species. 
        Assign each organism to one of the given species."""
        if species is None:
            self._list_of_species = []
        else:
            self._list_of_species = species
        if genomes is None:
            self._genomes = []
        else:
            self._genomes = genomes
        self._compatibility_threshold = config.compatibility_threshold      # threshold for being in the same species
        self._size = config.population_size                                 # population size
        self._elite_group_size = config.elite_group_size                    # members of population who always pass on
        self._config = config

        # generation number
        # divide the genomes into species
        self.__speciate_population()
        # set new representatives or eliminate extinct species according to new generation genomes
        for species in self._list_of_species:
            if len(species.genomes) != 0:
                species.randomize_representative()          # set new representative after speciating
            else:
                self._list_of_species.remove(species)       # species has no members and is therefore extinct

    @property
    def species(self):
        return self._list_of_species

    @property
    def genomes(self):
        return self._genomes

    @property
    def elite_group(self):
        """Returns a list of the best genomes which we'd like to keep for the next generation"""
        sorted_genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)
        return sorted_genomes[0:self._elite_group_size]

    def __add_genome(self, genome):
        assert genome not in self._genomes
        logging.info("New genome added: " + str(genome))
        self._genomes.append(genome)
        self.__assign_species(genome)

    def __assign_species(self, genome):
        """Assigns a species to a given genome, returning the index of the assigned species"""
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].representative
            # try to assign genome to species with given index
            if compatibility_distance(genome.genome, representative.genome) < self._compatibility_threshold:
                self._list_of_species[index].add_genomes(genome)
                logging.info("Assigned genome to species: " + str(genome) + str(index))
                return index
        # this is a new species!
        self._list_of_species.append(Species(genome))
        return len(self._list_of_species)-1  # the indexes are 0-based while len obviously isn't

    def __speciate_population(self):
        """Assign a species for every genome in the current population"""
        for genome in self._genomes:
            self.__assign_species(genome)

    def get_statistic(self, statistic):
        """Returns a certain statistic which is kept by the population.
        Statistic is an enum of type StatisticsType (defined in neat.py). Make sure to handle all statistics."""
        if statistic not in StatisticsTypes:
            raise Exception("Statistic type unknown to population")
        if statistic == StatisticsTypes.MAX_FITNESS:
            return self.max_fitness
        if statistic == StatisticsTypes.MIN_FITNESS:
            return self.min_fitness
        if statistic == StatisticsTypes.AVERAGE_FITNESS:
            return self.average_fitness
        if statistic == StatisticsTypes.NUM_SPECIES:
            return self.number_of_species
        if statistic == StatisticsTypes.BEST_GENOME:
            return self.best_genome
        if statistic == StatisticsTypes.WORST_GENOME:
            return self.worst_genome
        raise Exception("Statistic type unhandled in population")

    @property
    def max_fitness(self):
        return max([genome.fitness for genome in self.genomes])

    @property
    def min_fitness(self):
        return min([genome.fitness for genome in self.genomes])

    @property
    def average_fitness(self):
        return np.mean([genome.fitness for genome in self.genomes])

    @property
    def number_of_species(self):
        return len(self.species)

    @property
    def best_genome(self):
        return max(self.genomes, key=lambda genome: genome.fitness)

    @property
    def worst_genome(self):
        return min(self.genomes, key=lambda genome: genome.fitness)


