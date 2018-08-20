import itertools
import logging
import random
import numpy as np
import copy
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene


class Genome:

    #todo: consider adding node_genes to the ctor so we can define each nodes activation function
    #todo: that way when we mutate a genome, we can get the nodes list and alter one nodes activation
    def __init__(self, config, connection_genes=None):
        # Constants
        self._number_of_input_nodes = config.number_of_input_nodes
        self._number_of_output_nodes = config.number_of_output_nodes
        self._c1, self._c2, self._c3 = config.c1, config.c2, config.c3
        self._weight_mutation_distribution = config.weight_mutation_distribution            # for mutation of changing weights
        self._connection_weight_mutation_distribution = config.connection_weight_mutation_distribution      # for mutation of connection creation

        if self._number_of_input_nodes <= 0:
            raise ValueError('number of input nodes must be greater than 0')
        if self._number_of_output_nodes <= 0:
            raise ValueError('number of output nodes must be greater than 0')

        # Gene sequences. The Key is innovation number.
        if connection_genes is None:
            self._connection_genes = {}
        else:
            self._connection_genes = copy.copy(connection_genes)    # shallow copy the given connection_genes dict
        self._node_genes = {}                                       # Key is node_index (as in encode_node @ node_gene)

        self.__init_node_genes()

        #todo: heavy calculations ahead (no todo here I think, just be aware)
        self._neural_net = self.__create_neural_network()
        self._fitness = config.fitness_function(self._neural_net) #TODO: assuming for now that's the fitness_function's required api

    def __create_neural_network(self):
        raise NotImplementedError

    @property
    def node_genes(self):
        """Returns a dictionary of node-genes, the keys are indexes and values are node-genes"""
        return self._node_genes

    @property
    def connection_genes(self):
        """Returns a dict of connection genes, where the key is the innovation number of the connection gene value"""
        return self._connection_genes

    @property
    def size(self):
        """Returns the length of the entire genome, connection genes and node genes combined"""
        return len(self._connection_genes) + len(self._node_genes)

    def __init_node_genes(self):
        """Initializes node for the entire genome, i.e. adds SENSOR, OUTPUT, BIAS nodes which are present in all
        genomes, and adds necessary nodes for a given dictionary of connection_genes"""
        assert self._number_of_output_nodes > 0
        assert self._number_of_input_nodes > 0

        for i in range(self._number_of_input_nodes):
            self.__add_node_gene('SENSOR', i)  # SENSOR is an input node
        for i in range(self._number_of_output_nodes):
            self.__add_node_gene('OUTPUT', self._number_of_input_nodes + i)
        self.__add_node_gene('BIAS', -1)        # node with index -1 is the bias
        logging.debug("Added SENSOR, OUTPUT and BIAS node genes")

        # if we have connections we need to add their corresponding nodes to the genome
        self.__init_connection_genes_nodes()

    def __init_connection_genes_nodes(self):
        """Adds, for each connection gene in self.connection_gens, the corresponding nodes needed so the connection gene
        is well-defined within the genome. For example, if we have the connection (1,3) we add the nodes (if not already
        exists) 1 and 3.
        Only used in the genome's initialization phase. Otherwise use __add_connection_gene or __add_node_gene."""
        assert isinstance(self._connection_genes, dict)

        for connection_gene in self._connection_genes.values():
            source_index, dest_index = connection_gene.to_edge_tuple()
            if source_index not in self._node_genes.keys():
                self.__add_node_gene('HIDDEN', source_index)      # add source index
                logging.debug("Added source node gene with index: " + str(source_index))
            if dest_index not in self._node_genes.keys():
                self.__add_node_gene('HIDDEN', dest_index)        # add dest index
                logging.debug("Added dest node gene with index: " + str(dest_index))
            source_node = self._node_genes[source_index]
            dest_node = self._node_genes[dest_index]
            # add dest to source's neighbors
            source_node.add_connection_to(dest_node)
            # add source to dest's neighbors (edge in the opposite direction of course)
            dest_node.add_connection_from(source_node)

    def __add_node_gene(self, node_type, node_index):
        """Adds a single node gene to the genome"""
        assert node_index not in self._node_genes.keys()
        new_node_gene = NodeGene(node_type, node_index)
        self._node_genes[node_index] = new_node_gene
        logging.info("New node gene added: " + str(new_node_gene))
        return node_index

    def __delete_node_gene(self, node_index):
        assert node_index in self._node_genes
        # we only delete nodes if they became isolated after a delete_connection mutation.
        # note that all nodes start as isolated nodes after the __add_node mutation.
        assert self._node_genes[node_index].is_isolated()

        logging.info("Node gene deleted: " + str(self._node_genes[node_index]))
        del self._node_genes[node_index]

    def __add_connection_gene(self, source, dest, weight, enabled = True):
        # TODO: Liron: this may be redundant, I thought the constructor should receive the entire connection-gene-list rather than just adding a single connection
        # maybe this is useful for mutations?
        assert source in self._node_genes
        assert dest in self._node_genes

        new_connection_gene = ConnectionGene(source, dest, weight, True)
        self._connection_genes[new_connection_gene.innovation] = new_connection_gene
        self._node_genes[source].add_connection_to(dest)
        self._node_genes[dest].add_connection_from(source)
        logging.info("New connection gene added: " + str(new_connection_gene))

    def __delete_connection_gene(self, innovation_number):
        # TODO: Liron: maybe deleting genes isn't necessary, for now we'll leave it be
        if innovation_number in self._connection_genes:
            logging.info("Connection gene deleted: " + str(self._connection_genes[innovation_number]))
            source, dest = self._connection_genes[innovation_number].to_edge_tuple()
            del self._connection_genes[innovation_number]

            source_gene = self._node_genes[source]
            source_gene.delete_connection_to(dest)
            dest_gene = self._node_genes[dest]
            dest_gene.delete_connection_from(source)

            if dest_gene.is_isolated():
                logging.info("Removing isolated node gene: %s", str(dest_gene))
                self.__delete_node_gene(dest)
            if source_gene.is_isolated():
                logging.info("Removing isolated node gene: %s", str(source_gene))
                self.__delete_node_gene(source)
        else:
            logging.debug("Can't delete connection gene %s - not found", str(innovation_number))

    def __str__(self):
        return 'A genome. Node genes: %s, Connection genes: %s' % (self._node_genes, self._connection_genes)

    #TODO: is it a good practice? (avoid sneaky bugs)
    __repr__ = __str__


def compatibility_distance(genome1, genome2):
    """Returns the compatibility distance, a measure of how closely related two genomes are"""
    if genome1.size == 0 and genome2.size == 0:
        logging.info("genome1: %s AND genome2: %s both have 0 genes and hence have compitability distance of 0"
                     % (str(genome1), str(genome2)))
        return 0.0
    # create a new dict with all of the genomes' genes
    connection_genes1 = genome1.connection_genes()
    connection_genes2 = genome2.connection_genes()
    # N is as defined in the NEAT paper (number of genes in the larger genome
    N = max(len(connection_genes1), len(connection_genes2))

    disjoint, excess = calculate_mismatching_genes(connection_genes1, connection_genes2)
    matching_connection_genes_innovations = set(genome1.connection_genes().keys()).intersection(set(genome2.connection_genes().keys()))

    weight_differences = [abs(connection_genes1[innovation_num].weight - connection_genes2[innovation_num].weight)
                          for innovation_num in matching_connection_genes_innovations]

    average_weight_difference = np.mean(weight_differences)

    # TODO: maybe find prettier solution for c1,c2,c3
    return genome1.c1*len(excess)/N + genome1.c2*len(disjoint)/N + genome1.c3*average_weight_difference


def calculate_mismatching_genes(connection_genes1, connection_genes2):
    """Returns a pair of lists containing innovation numbers of disjoint and excess connection genes"""
    # Max innovation is of connection genes. Node genes don't hold an innovation number.
    max_innovation_genome1 = max(connection_genes1.keys())
    max_innovation_genome2 = max(connection_genes2.keys())

    n = min(max_innovation_genome1, max_innovation_genome2)
    m = max(max_innovation_genome1, max_innovation_genome2)

    # Innovation numbers of genes corresponding to exactly one connection_genes1 or connection_genes2
    non_matching_connection_genes = set(connection_genes1.keys()).symmetric_difference(set(connection_genes2.keys()))
    # refer to the NEAT paper for an accurate definition of excess and disjoint genes
    excess = []
    disjoint = []
    for innovation_num in range(m + 1):
        if innovation_num in non_matching_connection_genes:
            if innovation_num <= n:
                disjoint.append(innovation_num)
            else:
                excess.append(innovation_num)
    return disjoint, excess
