import random

from simplyneat.genome.genome import Genome
import logging

from simplyneat.organism.organism import Organism
from simplyneat.species.species import Species


class Population:
    def __init__(self, fitness_function, distance_threshold):
        self._list_of_species = []                          # a list of species
        self._organisms = []                                # a list of organisms
        self._fitness_function = fitness_function           # TODO: maybe receive fitness function in config?
        self._distance_threshold = distance_threshold       # TODO: receive distance_threshold in config

    def __add_organism(self, organism):
        assert organism not in self._organisms
        logging.info("New organism added: " + str(organism))
        self._organisms.append(organism)
        self.assign_species(organism)

    def assign_species(self, organism):
        logging.info("Assigned organism to species: " + str(organism) + str(species))
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].representative()

            #TODO: distance_threshold should be configurable
            distance_threshold = 3

            if Genome.compatibility_distance(organism.genome, representative.genome) < distance_threshold:
                self._list_of_species[index].add_organism(organism)     #TODO: implement
                return
        # this is a new species!
        self._list_of_species.append(Species(organism))

    def __calculate_adjusted_fitness(self, organism):
        assert isinstance(organism, Organism)
        assert organism in self._organisms
        fitness = self._fitness_function(organism)
        # at least 1 since the sharing of an organism with itself is 1
        sum_of_sharing = sum([Population.__sharing_function(Genome.compatibility_distance(organism.genome(), other_organism.genome()))
                         for other_organism in self._organisms])
        return fitness/sum_of_sharing

    @staticmethod
    def __sharing_function(distance, distance_threshold):
        if distance >= distance_threshold:          # TODO: make distance threshold a static variable
            return 0
        else:
            return 1

