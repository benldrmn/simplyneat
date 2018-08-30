import random
import copy
import logging
import itertools

from simplyneat.genome.genes.node_gene import NodeType, encode_node
from simplyneat.genome.genome import calculate_mismatching_genes, Genome, compatibility_distance
from simplyneat.population.population import Population
import numpy as np


class Breeder:
    """In charge of breeding populations, while applying mutations to new genomes. Keeps track of all innovations."""
    def __init__(self, config):
        self._compatibility_threshold = config.compatibility_threshold
        self._population_size = config.population_size
        self._connection_weight_mutation_distribution = config.connection_weight_mutation_distribution
        self._weight_mutation_distribution = config.weight_mutation_distribution
        self._reset_breeder = config.reset_breeder
        self._max_tournament_size = config.max_tournament_size
        self._elite_group_size = config.elite_group_size
        self._genome_number = 1
        self._innovation_number = 1
        self._config = config
        self._mutation_probability_dictionary = {self.__mutate_add_connection: config.add_connection_probability,
                                                 self.__mutate_add_node: config.add_node_probability,
                                                 self.__mutate_connection_weight: config.change_weight_probability,
                                                 self.__mutate_reenable_connection: config.reenable_connection_probability,
                                                 }
        self._innovations_dictionary = {}

    def breed_population(self, population):
        """Breeds and mutates population, returning the next generation"""
        new_population_genomes = population.elite_group
        # breed pairs
        pairs_of_parents_to_breed = self.__generate_parents_pairs_to_breed(population)
        for genome1, genome2 in pairs_of_parents_to_breed:
            new_population_genomes.append(self.__breed_parents(genome1, genome2))
        assert len(new_population_genomes) == self._population_size
        # mutate offsprings
        for genome in new_population_genomes:
            for mutation, probability in self._mutation_probability_dictionary.items():
                if np.random.binomial(1, probability):
                    mutation(genome)
        # reset all genomes while keeping old representatives
        for species in population.species:
            species.reset_genomes()
        if self._reset_breeder:         # reset innovations list rather than creating a whole new breeder each time
            self._innovations_dictionary = {}
        assert len(new_population_genomes) == self._population_size
        return Population(self._config, species_number=population.species_number,
                          genomes=new_population_genomes, species=population.species)

    def __generate_parents_pairs_to_breed(self, population):
        species_list = population.species
        new_species_distribution = self.__calculate_offspring_per_species(species_list, population)
        pairs_of_parents_to_breed = []
        # create new genomes from existing ones
        for species_index in range(len(species_list)):
            # repeat once for each genomes in the new species' distribution
            for _ in range(new_species_distribution[species_index]):
                # choose parents in k-tournament
                genomes = species_list[species_index].genomes
                assert len(genomes) > 0
                k = min(self._max_tournament_size, len(genomes))
                genome1 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                genome2 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                pairs_of_parents_to_breed.append((genome1, genome2))
        assert len(pairs_of_parents_to_breed) + min(1, min(population.size, self._elite_group_size)), self== self._population_size
        return pairs_of_parents_to_breed

    def __breed_parents(self, parent_genome1, parent_genome2):
        """Returns a genome containing the crossover of connection-genes from both genomes"""
        # Matching genes are inherited randomly, excess and disjoint genes are inherited from the better parent
        # if parents have same fitness, the better parent is the one with the smaller genome
        fitness1 = parent_genome1.fitness
        fitness2 = parent_genome2.fitness
        if fitness2 > fitness1 or (fitness1 == fitness2 and parent_genome2.size < parent_genome1.size):
            parent_genome1, parent_genome2 = parent_genome2, parent_genome1
            # Makes sure that genome1 is the genome of the fitter parent

        connection_genes1 = parent_genome1.connection_genes
        connection_genes2 = parent_genome2.connection_genes

        disjoint, excess = calculate_mismatching_genes(connection_genes1, connection_genes2)
        mismatching = disjoint + excess
        matching = set(connection_genes1.keys()).intersection(set(connection_genes2.keys()))
        #TODO: deep copy nodes and connections for the new genome instead of doing this
        offspring_connection_genes = {}
        # matching genes - inherit one from a random parent
        for innovation_number in matching:
            max_split_number = max(connection_genes1[innovation_number].split_number,
                                   connection_genes2[innovation_number].split_number)
            if random.choice([True, False]):        # take parent1's weight
                connection_gene = copy.copy(connection_genes1[innovation_number])
                connection_gene.split_number = max_split_number
                offspring_connection_genes[innovation_number] = connection_gene
            else:                                   # take parent2's weight
                connection_gene = copy.copy(connection_genes2[innovation_number])
                connection_gene.split_number = max_split_number
                offspring_connection_genes[innovation_number] = connection_gene
        # mismatched genes - inherit all from the fitter parent
        for innovation_number in mismatching:
            if innovation_number in connection_genes1.keys():
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])

        new_genome = Genome(self._config, genome_number=self._genome_number, connection_genes=offspring_connection_genes)
        self._genome_number += 1
        return new_genome

    def __mutate_add_connection(self, genome):
        assert isinstance(genome, Genome)
        possible_sources = genome.node_genes.keys()
        # INPUT\BIAS neurons can't be a destination.
        possible_destinations = [node_index for node_index in genome.node_genes.keys()
                                 if genome.node_genes[node_index].node_type not in [NodeType.BIAS, NodeType.INPUT]]
        # Two edges with the same source and destination are not possible.
        possible_edges = list(set(itertools.product(possible_sources, possible_destinations)) -\
                              set(map(lambda connection_gene: (connection_gene.source_node.node_index,
                                                               connection_gene.destination_node.node_index),
                                      genome.connection_genes.values())))
        if not possible_edges:
            logging.debug("No possible edges. Possible sources: %s, possible destinations: %s, current edges: %s",
                          str(possible_sources), str(possible_destinations), str(genome.connection_genes.keys()))
        else:
            source_index, dest_index = random.choice(possible_edges)  # randomly choose one edge from the possible edges
            source, dest = genome.node_genes[source_index], genome.node_genes[dest_index]
            new_innovation = None         # default value, sets new innovation by static innovation counter
            if (source, dest) in self._innovations_dictionary.keys():
                new_innovation = self._innovations_dictionary[(source, dest)]       # innovation from previous mutations
            else:
                new_innovation = self._innovation_number
                self._innovation_number += 1
            new_innovation = genome.add_connection_gene(source, dest, self._connection_weight_mutation_distribution(),
                                                        0, new_innovation, True)

            self._innovations_dictionary[(source, dest)] = new_innovation

    def __mutate_add_node(self, genome):
        """Takes an existing edge and splits it in the middle with a new node"""
        assert isinstance(genome, Genome)
        if not genome.connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            #
            active_connections = list(genome.connection_genes.values())
            active_connections = [connection for connection in active_connections
                                      if connection.is_enabled()]
            if not active_connections:
                logging.debug("add_note mutation failed: all connections are disabled")
                return
            old_connection = random.choice(active_connections)
            # add old connection to not split it again
            old_source = old_connection.source_node
            old_dest = old_connection.destination_node
            new_node_index = encode_node(old_source.node_index, old_dest.node_index, old_connection.split_number)

            old_connection.split_number += 1
            old_connection.disable()

            new_node = genome.add_node_gene(NodeType.HIDDEN, new_node_index)
            # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
            genome.add_connection_gene(old_source, new_node, weight=1,
                                       innovation=self._innovation_number, split_number=0, enabled=True)
            self._innovation_number += 1
            # the new connection leading out of the new node from to the old dest has
            # the old connection's weight according to the NEAT paper
            genome.add_connection_gene(new_node, old_dest, weight=old_connection.weight,
                                       innovation=self._innovation_number, split_number=0, enabled=True)
            self._innovation_number += 1

    def __mutate_connection_weight(self, genome):
        """Alters the weight of a connection"""
        if not genome.connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            # apparently 'dict_values' object does not support indexing
            connection_gene = random.choice(list(genome.connection_genes.values()))
            connection_gene.weight += self._weight_mutation_distribution()
            # TODO: read 4.1 better to understand how this works

    def __mutate_reenable_connection(self, genome):
        connection_genes = list(genome.connection_genes.values())
        disabled_connection_genes = [connection_gene for connection_gene in connection_genes
                                     if not connection_gene.is_enabled()]
        if not disabled_connection_genes:
            logging.debug("reenable_connection mutation failed: no connection genes to reenable")
            return
        (random.choice(disabled_connection_genes)).enable()

    def __calculate_adjusted_fitness(self, genome, population):
        """Calculates the adjusted fitness of a single genome"""
        # at least 1 since the sharing of an genome with itself is 1
        sum_of_sharing = sum([self.__sharing_function(compatibility_distance(genome, other_genome))
                              for other_genome in population.genomes])
        return genome.fitness / sum_of_sharing

    def __sharing_function(self, distance):
        if distance >= self._compatibility_threshold:
            return 0
        else:
            return 1

    def __calculate_offspring_per_species(self, list_of_species, population):
        """returns a list, entry [i] is the number of offsprings for species i in the following generation"""
        species_total_adjusted_fitness = [0] * len(list_of_species)
        for species_index in range(len(list_of_species)):
            for genome in list_of_species[species_index].genomes:
                species_total_adjusted_fitness[species_index] += self.__calculate_adjusted_fitness(genome, population)

        # species_adjusted_fitness[i] is the adjusted fitness of species i
        population_total_adjusted_fitness = sum(species_total_adjusted_fitness)  # entire population's adjusted fitness
        population_total_adjusted_fitness = max(population_total_adjusted_fitness, 1)
        # offspring number proportionate to relative fitness
        delta = max(0, self._elite_group_size - len(population.genomes))        # in case elite_size > population_size
        breeding_size = self._population_size - self._elite_group_size + delta  # don't need to breed elites
        new_species_distribution = [species_adjusted_fitness * breeding_size / population_total_adjusted_fitness
                                    for species_adjusted_fitness in species_total_adjusted_fitness]
        # the number of genomes in a species is an integer
        new_species_distribution = [int(x) for x in new_species_distribution]
        size_delta = breeding_size - sum(new_species_distribution)
        assert size_delta >= 0
        new_species_distribution[0] += size_delta
        assert sum(new_species_distribution) == breeding_size
        return new_species_distribution
