import random
import logging

from simplyneat.genome.breeders import GenomesBreeder
from simplyneat.genome.genome import Genome, compatibility_distance
from simplyneat.species.species import Species


class Population:

    _current_generation_number = 0      # TODO: static variable

    def __init__(self, config, genomes=None, species=None):
        if species is None:
            self._list_of_species = []
        else:
            self._list_of_species = species
        if genomes is None:
            self._genomes = []
        else:
            self._genomes = genomes
        #TODO: should probably be under the breeder and not in population init
        self._fitness_function = config.fitness_function
        self._distance_threshold = config.distance_threshold        # threshold for being in the same species
        self._size = config.population_size                         # population size
        self._change_weight_probability = config.change_weight_probability
        self._add_node_probability = config.add_node_probability
        self._add_connection_probability = config.add_connection_probability
        self._config = config

        #TODO: have it configurable - if reset innovations list each generation - create a new breeder, else used the one given by the config
        self._breeder = GenomesBreeder(self._config)
        # generation number
        Population._current_generation_number += 1
        self._generation_number = Population._current_generation_number
        # divide the genomes into species
        self.__speciate_population()

    @property
    def species(self):
        return self._list_of_species

    @property
    def genomes(self):
        return self._genomes

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
            if compatibility_distance(genome.genome, representative.genome) < self._distance_threshold:
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
