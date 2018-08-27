import random
import copy
import logging
import itertools

from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene, encode_node, NodeType
from simplyneat.genome.genes.node_gene import encode_node
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
        self._config = config
        self._mutation_probability_dictionary = {self.__mutate_add_connection: config.add_connection_probability,
                                                 self.mutate_add_node: config.add_node_probability,
                                                 self.__mutate_connection_weight: config.change_weight_probability}
        self._innovations_dictionary = {}

    def breed_population(self, population):
        """Breeds and mutates population, returning the next generation"""
        new_population_genomes = population.elite_group
        # breed pairs
        pairs_of_parents_to_breed = self.__generate_parents_pairs_to_breed(population)
        for genome1, genome2 in pairs_of_parents_to_breed:
            new_population_genomes.append(self.__breed_parents(genome1, genome2))
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
        return Population(self._config, genomes=new_population_genomes, species=population.species)

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
                k = min(self._max_tournament_size, len(genomes))
                genome1 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                genome2 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                pairs_of_parents_to_breed.append((genome1, genome2))
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

        offspring_connection_genes = {}
        # matching genes - inherit one from a random parent
        for innovation_number in matching:
            if random.choice([True, False]):        # take parent1's weight
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])
            else:                                   # take parent2's weight
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes2[innovation_number])

        # mismatched genes - inherit all from the fitter parent
        for innovation_number in mismatching:
            if innovation_number in connection_genes1.keys():
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])

        return Genome(self._config, offspring_connection_genes)

    def __mutate_add_connection(self, genome):
        assert isinstance(genome, Genome)
        possible_sources = genome.node_genes.keys()
        # INPUT\BIAS neurons can't be a destination.
        possible_destinations = [node_index for node_index in genome.node_genes.keys()
                                 if genome.node_genes[node_index].node_type not in [NodeType.BIAS, NodeType.INPUT]]
        # Two edges with the same source and destination are not possible.
        # TODO: The implementation of to_edge_tuple changed for some reason and now it returns a tuple of nodes instead of a tuple of node_indexes, but the code below wasn't changed
        possible_edges = list(set(itertools.product(possible_sources, possible_destinations)) -\
            set(map(lambda connection_gene: connection_gene.to_edge_tuple(), genome.connection_genes.values())))
        if not possible_edges:
            logging.debug("No possible edges. Possible sources: %s, possible destinations: %s, current edges: %s",
                          str(possible_sources), str(possible_destinations), str(genome.connection_genes.keys()))
        else:
            source_index, dest_index = random.choice(possible_edges)  # randomly choose one edge from the possible edges
            source, dest = genome.node_genes[source_index], genome.node_genes[dest_index]
            new_innovation = None         # default value, sets new innovation by static innovation counter
            if (source, dest) in self._innovations_dictionary.keys():
                new_innovation = self._innovations_dictionary[(source, dest)]       # innovation from previous mutations
            new_innovation = genome.add_connection_gene(source, dest, self._connection_weight_mutation_distribution(),
                                                        True, new_innovation)

            self._innovations_dictionary[(source, dest)] = new_innovation

    def mutate_add_node(self, genome):      # TODO: made this public for the test
        """Takes an existing edge and splits it in the middle with a new node"""
        assert isinstance(genome, Genome)
        if not genome.connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            enabled_connections = list(genome.connection_genes.values())
            enabled_connections = [connection for connection in enabled_connections
                                   if connection.is_enabled()]
            old_connection = random.choice(enabled_connections)
            old_source, old_dest = old_connection.to_edge_tuple()
            new_node_index = encode_node(old_source.node_index, old_dest.node_index)

            old_connection.disable()

            new_node = genome.add_node_gene(NodeType.HIDDEN, new_node_index)
            if not isinstance(new_node, NodeGene):
                print("mutate_add_node found that new_node ain't a NodeGene")
            # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
            genome.add_connection_gene(old_source, new_node, 1, True)
            # the new connection leading out of the new node from to the old dest has
            # the old connection's weight according to the NEAT paper
            genome.add_connection_gene(new_node, old_dest, old_connection.weight, True)

    def __mutate_connection_weight(self, genome):
        """Alters the weight of a connection"""
        if not genome.connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            # apparently 'dict_values' object does not support indexing
            connection_gene = random.choice(list(genome.connection_genes.values()))
            connection_gene.weight += self._weight_mutation_distribution()
            # TODO: read 4.1 better to understand how this works

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
