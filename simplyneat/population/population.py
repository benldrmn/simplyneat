import random
import logging
from simplyneat.genome.genome import Genome
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
        self._fitness_function = config.fitness_function
        self._distance_threshold = config.distance_threshold        # threshold for being in the same species
        self._size = config.population_size                         # population size
        self._change_weight_probability = config.change_weight_probability
        self._add_node_probability = config.add_node_probability
        self._add_connection_probability = config.add_connection_probability
        self._config = config
        # generation number
        Population._current_generation_number += 1
        self._generation_number = Population._current_generation_number
        # divide the organisms into species
        for organism in self._organisms:
            self.speciate(organism)
        self.fitness_matrix = self.__calculate_fitness_matrix()
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
            # try to assign organism to species with given index
            if Genome.compatibility_distance(organism.genome, representative.genome) < self._distance_threshold:
                self._list_of_species[index].add_organism(organism)
                logging.info("Assigned organism to species: " + str(organism) + str(index))
                return index
        # this is a new species!
        self._list_of_species.append(Species(organism))
        return len(self._list_of_species)-1

    def next_population(self):
        """Breed each species breed two of its members according to the new distribution,
        mutate the new offspring, then assign it a new species once again"""
        new_species_distribution = self.__calculate_number_of_offsprings()
        new_organisms = []
        # create new organisms from existing ones
        for species_index in range(len(self._list_of_species)):
            # repeat once for each organism in the new species' distribution
            for _ in new_species_distribution[species_index]:
                # choose parents    # TODO: for now we choose randomly, maybe choose by fitness?
                index1 = random.choice(len(self._list_of_species[species_index].organisms))
                index2 = random.choice(len(self._list_of_species[species_index].organisms))
                genome1 = self._list_of_species[species_index].organisms[index1].genome
                genome2 = self._list_of_species[species_index].organisms[index2].genome
                # crossover
                new_organism = Genome.breed(genome1, genome2,
                                            self.adjusted_fitness_matrix[species_index][index1], self.adjusted_fitness_matrix[species_index][index2])
                # TODO: instead of adjusted fitness matrix maybe use regular fitness
                # perform the three mutations
                # TODO: put the list of mutations in a class and run that instead of one-by-one
                if np.random.binomial(1, self._change_weight_probability):
                    new_organism.mutate_connection_weight()
                if np.random.binomial(1, self._add_connection_probability):
                    new_organism.mutate_add_connection()
                if np.random.binomial(1, self._add_node_probability):
                    new_organism.mutate_add_node()
                # TODO: note that the order was arbitrary, maybe there should be a specific order
                new_organisms.append(new_organism)

        # clear the list of species from previous generation
        for species in self._list_of_species:
            species.reset_organisms()

        # return a new population based on the new organisms, the previous species (without organisms but with the old reps.)
        return Population(self._config, new_organisms, self._list_of_species)

    def __calculate_number_of_offsprings(self):
        """returns a list, entry [i] is the number of offsprings for species i in the following generation"""
        species_adjusted_fitness = [sum(species_fitness_vector) for species_fitness_vector in self.adjusted_fitness_matrix]
        # species_adjusted_fitness[i] is the adjusted fitness of species i
        total_fitness = sum(species_adjusted_fitness)           # total adjusted fitness of entire population
        new_species_distribution = self._size * species_adjusted_fitness / total_fitness
        assert sum(new_species_distribution) == self._size                           # offspring number proportionate to relative fitness
        # TODO: the sum of the value returned above can be not equal to self._size due to rounding, fix this later
        # or just hope this never happens (which it probably will)
        return new_species_distribution

    def __calculate_fitness_matrix(self):
        """Returns a list of lists, entry [i,j] contains the regular fitness of organism j in species i"""
        fitness_matrix = []
        for i in range(len(self._list_of_species)):
            fitness_matrix[i] = []
            for j in range(len(self._list_of_species[i].organisms)):
                fitness_matrix[i][j] = self._fitness_function(self._list_of_species[i].organisms[j])
        return fitness_matrix[i][j]

    def __calculate_adjusted_fitness_matrix(self):
        """Returns a list of lists, entry [i,j] contains the adjusted fitness of organism j in species i"""
        fitness_matrix = []
        for i in range(len(self._list_of_species)):
            fitness_matrix[i] = []
            for j in range(len(self._list_of_species[i].organisms)):
                fitness_matrix[i][j] = self.__calculate_adjusted_fitness(i, j)
        return fitness_matrix

    def __calculate_adjusted_fitness(self, species_index, organism_index):
        """Calculates the adjusted fitness of a single organism"""
        assert species_index in range(len(self._list_of_species))
        assert organism_index in range(len(self._list_of_species[species_index].organisms))
        fitness = self.fitness_matrix[species_index][organism_index]
        # at least 1 since the sharing of an organism with itself is 1
        sum_of_sharing = sum([self.__sharing_function(
            Genome.compatibility_distance(self._list_of_species[species_index].organisms[organism_index].genome(),
                                          other_organism.genome())) for other_organism in self._organisms])
        return fitness/sum_of_sharing

    # this was initially a static method but the threshold was ugly
    def __sharing_function(self, distance):
        if distance >= self._distance_threshold:
            return 0
        else:
            return 1

