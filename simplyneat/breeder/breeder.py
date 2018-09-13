try:
    #TODO: doesn't help, forget about it (only use mp.pool)
    from pathos.pools import ProcessPool as Pool
except ImportError:
    # fall back on the multiprocessing pool implementation
    from multiprocessing.pool import Pool

import random
import copy
import functools

from simplyneat.breeder import mutations
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genome import calculate_mismatching_genes, Genome, compatibility_distance
from simplyneat.population.population import Population


class Breeder:
    """In charge of breeding populations, while applying mutations to new genomes. Keeps track of all innovations."""
    def __init__(self, config):
        #TODO: re-add reset innovations each generation to config
        self._compatibility_threshold = config.compatibility_threshold
        self._population_size = config.population_size
        self._max_tournament_size = config.max_tournament_size
        self._elite_group_size = config.elite_group_size
        self._processes_in_pool = config.processes_in_pool
        self._config = config

        if self._processes_in_pool > 1:
            self._pool = Pool(self._processes_in_pool)
        else:
            self._pool = None

        self._innovation_counter = 0
        self._innovations_dictionary = {}

    def breed_population(self, population):
        """Breeds and mutates population, returning the next generation"""
        new_population_genomes = population.elite_group
        pairs_of_parents_to_breed = self._generate_parents_pairs_to_breed(population)
        # the functools.partial is a workaround to pass the config since pool.map doesn't accept lambda functions
        #TODO: if config holds a lambda function (i.e. fitness) it fails. don't allow lambdas in config
        if self._pool is None:
            offsprings_structural_innovations_pairs = list(map(functools.partial(_produce_offspring, config=self._config),
                                                         pairs_of_parents_to_breed))
        else:
            offsprings_structural_innovations_pairs = self._pool.map(functools.partial(_produce_offspring, config=self._config),
                                                                    pairs_of_parents_to_breed)

        new_structural_innovations = []
        new_offsprings = []
        for offspring_and_innovations_pair in offsprings_structural_innovations_pairs:
            new_offsprings.append(offspring_and_innovations_pair[0])
            new_structural_innovations += offspring_and_innovations_pair[1]
        # assign the proper innovation numbers to the structural changes (for example, new connection) from the last
        # breeding session.
        self._assign_innovations(new_structural_innovations)

        # add the new offsprings (with the correct innovation number for all of the genes)
        # to the new population's genomes
        new_population_genomes += new_offsprings
        # reuse the old population's species - keep the representatives but clear all of the genomes in each species
        # so the new population created would speciate it's genomes itself
        for species in population.species:
            species.reset_genomes()

        assert len(new_population_genomes) == self._population_size
        return Population(self._config, genomes=new_population_genomes, species=population.species)

    def _generate_parents_pairs_to_breed(self, population):
        species_list = population.species
        new_species_distribution = self._calculate_offspring_per_species(species_list, population)
        pairs_of_parents_to_breed = []
        # create new genomes from existing ones
        for species_index in range(len(species_list)):
            # repeat once for each genomes in the new species' distribution
            for _ in range(new_species_distribution[species_index]):
                # choose parents in a k-tournament manner - sample k genomes randomly and take the best out of those k
                genomes = species_list[species_index].genomes
                assert len(genomes) > 0
                k = min(self._max_tournament_size, len(genomes))
                genome1 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                genome2 = max(random.sample(genomes, k), key=lambda genome: genome.fitness)
                pairs_of_parents_to_breed.append((genome1, genome2))
        return pairs_of_parents_to_breed

    def _calculate_adjusted_fitness(self, genome, population):
        """Calculates the adjusted fitness of a single genome"""
        # at least 1 since the sharing of an genome with itself is 1
        sum_of_sharing = sum([self._sharing_function(compatibility_distance(genome, other_genome))
                              for other_genome in population.genomes])
        return genome.fitness / sum_of_sharing

    def _sharing_function(self, distance):
        if distance >= self._compatibility_threshold:
            return 0
        else:
            return 1

    def _calculate_offspring_per_species(self, list_of_species, population):
        """returns a list, entry [i] is the number of offsprings for species i in the following generation"""
        species_total_adjusted_fitness = [0] * len(list_of_species)
        for species_index in range(len(list_of_species)):
            for genome in list_of_species[species_index].genomes:
                species_total_adjusted_fitness[species_index] += self._calculate_adjusted_fitness(genome, population)

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

    def _assign_innovations(self, genes):
        # we only assign innovation numbers to connection genes
        connection_genes = [gene for gene in genes if isinstance(gene, ConnectionGene)]
        for gene in connection_genes:
            if gene.index in self._innovations_dictionary:
                gene.innovation = self._innovations_dictionary[gene.index]
            else:
                gene.innovation = self._innovation_counter
                self._innovations_dictionary[gene.index] = gene.innovation
                self._innovation_counter += 1


def _produce_offspring(parents_pair, config):
    offspring = _breed_parents(parents_pair[0], parents_pair[1], config)
    structural_innovations = _mutate_offspring(offspring, config)
    return offspring, structural_innovations


def _breed_parents(parent_genome1, parent_genome2, config):
    """Returns a genome containing the crossover of connection-genes from both genomes"""
    # Matching genes are inherited randomly, excess and disjoint genes are inherited from the better parent
    # if parents have same fitness, the better parent is the one with the smaller genome
    fitness1 = parent_genome1.fitness
    fitness2 = parent_genome2.fitness
    # Make sure that genome1 is the genome of the fitter parent
    if fitness2 > fitness1 or (fitness1 == fitness2 and parent_genome2.size < parent_genome1.size):
        parent_genome1, parent_genome2 = parent_genome2, parent_genome1

    innovation_to_connections1 = {connection.innovation: connection for connection in parent_genome1.connection_genes.values()}
    innovation_to_connections2 = {connection.innovation: connection for connection in parent_genome2.connection_genes.values()}

    disjoint, excess = calculate_mismatching_genes(innovation_to_connections1, innovation_to_connections2)
    mismatching = disjoint + excess
    matching = set(innovation_to_connections1.keys()).intersection(set(innovation_to_connections2.keys()))
    #TODO: deep copy nodes and connections for the new genome instead of doing this
    offspring_connection_genes = {}
    # matching genes - inherit one from a random parent
    for innovation_number in matching:
        max_split_number = max(innovation_to_connections1[innovation_number].split_number,
                               innovation_to_connections2[innovation_number].split_number)
        if random.choice([True, False]):        # take parent1's weight
            connection_gene = copy.copy(innovation_to_connections1[innovation_number])
            connection_gene.split_number = max_split_number
            offspring_connection_genes[innovation_number] = connection_gene
        else:                                   # take parent2's weight
            connection_gene = copy.copy(innovation_to_connections2[innovation_number])
            connection_gene.split_number = max_split_number
            offspring_connection_genes[innovation_number] = connection_gene
    # mismatched genes - inherit all from the fitter parent
    for innovation_number in mismatching:
        if innovation_number in innovation_to_connections1.keys():
            offspring_connection_genes[innovation_number] = copy.copy(innovation_to_connections1[innovation_number])

    new_genome = Genome(config, connection_genes=offspring_connection_genes)
    return new_genome


def _mutate_offspring(offspring, config):
    """returns a list of the new structural changes' genes added to the offspring so the breeder can assign them
    the appropriate innovation number"""
    structural_mutations_genes = []
    if random.random() < config.add_connection_probability:
        structural_mutations_genes += mutations.mutate_add_connection(genome=offspring,
                                                                      connection_weight_mutation_distribution=
                                                                      config.connection_weight_mutation_distribution)
    if random.random() < config.add_node_probability:
        structural_mutations_genes += mutations.mutate_add_node(genome=offspring)
    if random.random() < config.change_weight_probability:
        mutations.mutate_connection_weight(genome=offspring,
                                           weight_mutation_distribution=
                                           config.change_weight_mutation_distribution)
    if random.random() < config.reenable_connection_probability:
        mutations.mutate_reenable_connection(genome=offspring)

    return structural_mutations_genes
