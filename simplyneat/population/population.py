import logging
import random
import time
from enum import Enum

import numpy as np

from simplyneat.genome.genome import compatibility_distance
from simplyneat.species.species import Species


class Population:

    def __init__(self, config, genomes=None, species=None):
        """Builds the population according to a list of genomes and species. 
        Assign each organism to one of the given species."""
        start_time = time.time()
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

        # divide the genomes into species
        self._speciate_population()
        # set new representatives or eliminate extinct species according to new generation genomes
        for species in self._list_of_species:
            if len(species.genomes) != 0:
                species.randomize_representative()          # set new representative after speciating
            else:
                self._list_of_species.remove(species)       # species has no members and is therefore extinct
        logging.debug("Population init took %s sec. #%s species of sizes: %s" %
                      (time.time() - start_time, len(self._list_of_species),
                       [species.size for species in self._list_of_species]))

    @property
    def species(self):
        return self._list_of_species

    @property
    def genomes(self):
        return self._genomes

    @property
    def elite_group(self):
        """Returns a list of the best genomes which we'd like to keep for the next generation"""
        sorted_genomes = sorted(self._genomes, key=lambda genome: genome.fitness, reverse=True)
        return sorted_genomes[0:self._elite_group_size]

    @property
    def size(self):
        return self._size

    def _add_genome(self, genome):
        assert genome not in self._genomes
        self._genomes.append(genome)
        self._assign_species(genome)

    def _assign_species(self, genome):
        """Assigns a species to a given genome, returning the index of the assigned species"""
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].representative
            # try to assign genome to species with given index
            if compatibility_distance(genome, representative) < self._compatibility_threshold:
                self._list_of_species[index].add_genome(genome)
                return index
        # The genome doesn't belong to an existing species. This is a new species, with only genome as a member for now.
        self._list_of_species.append(Species([genome]))
        return len(self._list_of_species) - 1  # the indexes are 0-based while len obviously isn't

    def _speciate_population(self):
        """Assign a species for every genome in the current population"""
        speciation_start_time = time.time()
        for genome in self._genomes:
            self._assign_species(genome)
        logging.debug("Speciation took %s sec" % (time.time() - speciation_start_time))

    def get_statistic(self, statistic):
        """Returns a certain statistic which is kept by the population.
        Statistic is an enum of type StatisticsType. Make sure to handle all statistics."""
        if statistic not in StatisticsTypes:
            raise ValueError("Statistic type unknown to population")
        if statistic == StatisticsTypes.MAX_FITNESS:
            return self.max_fitness
        if statistic == StatisticsTypes.MIN_FITNESS:
            return self.min_fitness
        if statistic == StatisticsTypes.AVERAGE_FITNESS:
            return self.average_fitness
        if statistic == StatisticsTypes.NUM_SPECIES:
            return self.number_of_species

    @property
    def max_fitness(self):
        return max([genome.fitness for genome in self._genomes])

    @property
    def min_fitness(self):
        return min([genome.fitness for genome in self._genomes])

    @property
    def average_fitness(self):
        return np.mean([genome.fitness for genome in self._genomes])

    @property
    def number_of_species(self):
        return len(self.species)

    @property
    def best_genome(self):
        return max(self._genomes, key=lambda genome: genome.fitness)

    def __str__(self):
        return 'Population with %s species and %s genomes: \nSpecies: %s' \
               % (len(self.species), len(self._genomes), self.species)

#TODO: refactor out to stats class
class StatisticsTypes(Enum):
    MAX_FITNESS = 'MAX_FITNESS'
    MIN_FITNESS = 'MIN_FITNESS'
    AVERAGE_FITNESS = 'AVERAGE_FITNESS'
    NUM_SPECIES = 'NUM_SPECIES'
