import random
import logging

from simplyneat.genome.breeders import GenomesBreeder
from simplyneat.genome.genome import Genome, compatibility_distance
from simplyneat.organism.organism import Organism
from simplyneat.species.species import Species
import numpy as np


class Population:

    _current_generation_number = 0      # TODO: static variable

    def __init__(self, config, organisms=None, list_of_species=None):
        if list_of_species is None:
            self._list_of_species = []                              # a list of species
        else:
            self._list_of_species = list_of_species
        if organisms is None:
            self._organisms = []                                    # a list of organisms
        else:
            self._organisms = organisms
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
        # divide the organisms into species
        self.__speciate_population()

    @property
    def species(self):
        return self._list_of_species

    @property
    def organisms(self):
        return self._organisms

    def __add_organism(self, organism):
        assert organism not in self._organisms
        logging.info("New organism added: " + str(organism))
        self._organisms.append(organism)
        self.__assign_species(organism)

    def __assign_species(self, organism):
        """Assigns a species to a given organism, returning the index of the assigned species"""
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].representative
            # try to assign organism to species with given index
            if compatibility_distance(organism.genome, representative.genome) < self._distance_threshold:
                self._list_of_species[index].add_organism(organism)
                logging.info("Assigned organism to species: " + str(organism) + str(index))
                return index
        # this is a new species!
        self._list_of_species.append(Species(organism))
        return len(self._list_of_species)-1  # the indexes are 0-based while len obviously isn't

    def __speciate_population(self):
        """Assign a species for every organism in the current population"""
        for organism in self._organisms:
            self.__assign_species(organism)
