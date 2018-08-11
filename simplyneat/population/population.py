import random
import logging
from simplyneat.genome.genome import Genome
from simplyneat.organism.organism import Organism
from simplyneat.species.species import Species


class Population:
    def __init__(self, config):
        self._list_of_species = []                          # a list of species
        self._organisms = []                                # a list of organisms
        self._fitness_function = config.fitness_function
        self._distance_threshold = config.distance_threshold       # threshold for being in the same species
        self._size = config.population_size                        # population size
        self.adjusted_fitness_matrix = self.__calculate_adjusted_fitness_matrix()

    def __add_organism(self, organism):
        assert organism not in self._organisms
        logging.info("New organism added: " + str(organism))
        self._organisms.append(organism)
        self.speciate(organism)

    def speciate(self, organism):
        """Assigns a species to a given organism, returning the index of the assigned species"""
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].representative()

            #TODO: distance_threshold should be configurable
            distance_threshold = 3

            if Genome.compatibility_distance(organism.genome, representative.genome) < distance_threshold:
                self._list_of_species[index].add_organism(organism)
                logging.info("Assigned organism to species: " + str(organism) + str(index))
                return index
        # this is a new species!
        self._list_of_species.append(Species(organism))
        return len(self._list_of_species)-1

    def __calculate_number_of_offsprings(self):
        """returns a list, entry [i] is the number of offsprings for species i in the following generation"""
        species_adjusted_fitness = [sum(species_fitness_vector) for species_fitness_vector in self.adjusted_fitness_matrix]
        total_fitness = sum(species_adjusted_fitness)
        return self._size * species_adjusted_fitness / total_fitness             # offspring number proportionate to relative fitness
        # TODO: the sum of the value returned above can be not equal to self._size due to rounding, fix this latertm

    def __calculate_adjusted_fitness_matrix(self):
        """Returns a list of lists, entry [i,j] contains the regular adjusted of organism j in species i"""
        fitness_matrix = []
        for i in range(len(self._list_of_species)):
            fitness_matrix[i] = []
            for j in range(len(self._list_of_species[i])):
                fitness_matrix[i][j] = self.__calculate_adjusted_fitness(self._list_of_species[i].organisms[j])
        return fitness_matrix

    def __calculate_adjusted_fitness(self, organism):
        """Calculates the adjusted fitness of a single organism"""
        assert isinstance(organism, Organism)
        assert organism in self._organisms
        fitness = self._fitness_function(organism)
        # at least 1 since the sharing of an organism with itself is 1
        sum_of_sharing = sum([Population.__sharing_function(Genome.compatibility_distance(organism.genome(),
                                                            other_organism.genome()), self._distance_threshold)
                                                            for other_organism in self._organisms])
        return fitness/sum_of_sharing

    @staticmethod
    def __sharing_function(distance, distance_threshold):
        if distance >= distance_threshold:          # TODO: make distance threshold a static variable
            return 0
        else:
            return 1

