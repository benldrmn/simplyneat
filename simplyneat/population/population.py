from simplyneat.species.species import Species
from simplyneat.genome.genome import Genome
import logging
import random


class Population:
    def __init__(self):
        self._list_of_species = []                          # a list of species
        self._organisms = []                                # a list of organisms

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




