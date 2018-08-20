import random
import copy
import logging
import itertools

from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene, encode_node
from simplyneat.genome.genome import calculate_mismatching_genes, Genome, compatibility_distance
from simplyneat.population.population import Population


class GenomesBreeder:
    def __init__(self, config):
        self._distance_threshold = config.distance_threshold
        self._population_size = config.population_size
        self._connection_weight_mutation_distribution = config.connection_weight_mutation_distribution
        self._weight_mutation_distribution = config.weight_mutation_distribution
        self._config = config

    def breed_population(self, population):
        new_population_organisms = []
        pairs_of_parents_to_breed = self.__generate_parents_pairs_to_breed(population)
        for genome1, genome2 in pairs_of_parents_to_breed:
            new_population_organisms.append(self.__breed_parents(genome1, genome2))
        #TODO: we pass population.speceis to the new population without assigning new representatives! that's not good
        #TODO: don't forget to apply mutations and check innovations somewhere in here
        return Population(self._config, new_population_organisms, population.species)

    def __generate_parents_pairs_to_breed(self, population):
        species_list = population.species
        new_species_distribution = self.__calculate_offspring_per_species(species_list)
        pairs_of_parents_to_breed = []
        # create new organisms from existing ones
        for species_index in range(len(species_list)):
            # repeat once for each organism in the new species' distribution
            for _ in range(new_species_distribution[species_index]):
                # choose parents    # TODO: choose using k-tournament
                index1 = random.choice(len(species_list[species_index].organisms))
                index2 = random.choice(len(species_list[species_index].organisms))
                genome1 = species_list[species_index].organisms[index1].genome
                genome2 = species_list[species_index].organisms[index2].genome
                pairs_of_parents_to_breed.append((genome1, genome2))
        return pairs_of_parents_to_breed

    def __breed_parents(self, parent_genome1, parent_genome2):
        """Fitness1, fitness2 are the regular fitnesses of genome1, genome2.
        Returns a genome containing the crossover of connection-genes from both genomes"""
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
        #TODO: disable genes with some probability if parent has it disabled......
        # matching genes - inherit one from a random parent
        for innovation_number in matching:
            if random.choice([True, False]):
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])     # copied because mutations can change values
            else:
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes2[innovation_number])

        # mismatched genes - inherit all from the fitter parent
        for innovation_number in mismatching:
            if innovation_number in connection_genes1.keys():
                offspring_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])

        #TODO: mutate before returning
        return Genome(self._config, offspring_connection_genes)

    #TODO: refactor out to Mutator
    #TODO: not good! check innovation before creating a new connection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __mutate_add_connection(self, genome):
        assert isinstance(genome, Genome)
        possible_sources = genome.node_genes
        # INPUT\BIAS neurons can't be a destination.
        possible_destinations = [node_index for node_index in genome.node_genes
                                 if genome.node_genes[node_index].type not in ['BIAS', 'INPUT']]
        # Two edges with the same source and destination are not possible.
        possible_edges = set(itertools.product(possible_sources, possible_destinations)) -\
                         set(map(lambda connection_gene: connection_gene.to_edge_tuple(), genome.connection_genes.values()))
        if not possible_edges:
            logging.debug("No possible edges. Possible sources: %s, possible destinations: %s, current edges: %s",
                          str(possible_sources), str(possible_destinations), str(genome.connection_genes.keys()))
        else:
            source, dest = random.choice(possible_edges)  # randomly choose one edge from the possible edges
            # TODO: I assume that the new connection gene is always enabled after the mutation
            # TODO: Check innovations list
            #TODO: actually add the new connection somewhere....
            new_connection = ConnectionGene(source, dest, self._connection_weight_mutation_distribution, True)

    def __mutate_add_node(self, genome):
        """Takes an existing edge and splits it in the middle with a new node"""
        assert isinstance(genome, Genome)
        if not genome.connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            old_connection = random.choice(list(genome.connection_genes.values()))
            old_source, old_dest = old_connection.to_edge_tuple()
            #TODO: CHECK IF THIS INNOVATION ALREADY EXISTS IN POPULATION!!! (LIKE IN ADD CONNECTION)
            new_node_index = encode_node(old_source, old_dest)
            # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
            new_connection_to_node = ConnectionGene(old_source, new_node_index, 1, True)
            # the new connection leading out of the new node from to the old dest has
            # the old connection's weight according to the NEAT paper
            new_connection_from_node = ConnectionGene(new_node_index, old_dest, old_connection.weight, True)
            old_connection.disable()
            #TODO: mutate the actual genome (didnt do shit with the created genes)

    def __mutate_connection_weight(self):
        """Alters the weight of a connection"""
        connection_gene = random.choice(self._connection_genes.values())
        connection_gene.weight += self._weight_mutation_distribution()
        # TODO: read 4.1 better to understand how this works


    def __calculate_adjusted_fitness(self, organism, population):
        """Calculates the adjusted fitness of a single organism"""
        # at least 1 since the sharing of an organism with itself is 1
        #TODO: Fix the syntax. we should have list of spieces hold genomes - organism class should be deleted by now
        sum_of_sharing = sum([self.__sharing_function(compatibility_distance(organism.genome, other_organism.genome))
                              for other_organism in population.organisms])
        return organism.fitness/sum_of_sharing

    # todo: this was initially a static method but the threshold was ugly
    def __sharing_function(self, distance):
        if distance >= self._distance_threshold:
            return 0
        else:
            return 1

    def __calculate_offspring_per_species(self, list_of_species):
        #TODO: see if documentation of this function still holds after changes
        """returns a list, entry [i] is the number of offsprings for species i in the following generation"""
        #TODO: clean this ugly solution:
        species_total_adjusted_fitness = [0] * len(list_of_species)
        for species_index in range(len(list_of_species)):
            for organism in list_of_species:
                species_total_adjusted_fitness[species_index] += self.__calculate_adjusted_fitness(organism)

        # species_adjusted_fitness[i] is the adjusted fitness of species i
        population_total_adjusted_fitness = sum(species_total_adjusted_fitness)  # total adjusted fitness of entire population
        # offspring number proportionate to relative fitness
        new_species_distribution = [self._population_size * species_fitness / population_total_adjusted_fitness
                                    for species_fitness in species_total_adjusted_fitness]
        # the number of organisms in a species is an integer
        #TODO: maybe normalize instead of round?
        new_species_distribution = [int(round(x)) for x in new_species_distribution]
        #TODO: the population size changes from the initial config. probably ok. just document
        self._population_size = sum(new_species_distribution)
        return new_species_distribution