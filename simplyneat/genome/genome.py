from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene
from itertools import product
import logging
import random
import numpy as np


class Genome:
    def __init__(self, number_of_input_nodes, number_of_output_nodes):
        if number_of_input_nodes <= 0:
            raise ValueError('number of input nodes must be greater than 0')
        if number_of_output_nodes <= 0:
            raise ValueError('number of output nodes must be greater than 0')
        self._max_used_node_index = 0  # the current maximal index used for a node. Note that this node might have been removed already
        self._node_genes = {}  # key: node_index
        self._connection_genes = {}  # key: (source_node, dest_node). This type of pair is called an edge.
        self.__init_node_genes(number_of_input_nodes, number_of_output_nodes)

    #TODO: the 2 below are possibly redundant
    # returns a dict of node genes, where the key is the innovation number of the node gene value
    @property
    def node_genes(self):
        return {value.innovation: value for value in self._node_genes.values()}

    # returns a dict of connection genes, where the key is the innovation number of the connection gene value
    @property
    def connection_genes(self):
        return {value.innovation: value for value in self._connection_genes.values()}

    @property
    def genes(self):
        """Returns a dictionary, keys are innovation numbers and values are corresponding genes"""
        return {value.innovation: value for value in self._node_genes.values() + self._connection_genes.values()}

    @staticmethod
    def __calculate_mismatching_genes(genome1_genes, genome2_genes):
        """Returns a pair of lists containing innovation numbers of disjoint and excess genes"""
        max_innovation_genome1 = max(genome1_genes.keys())      # max innovation is of connection genes
        max_innovation_genome2 = max(genome2_genes.keys())

        n = min(max_innovation_genome1, max_innovation_genome2)
        m = max(max_innovation_genome1, max_innovation_genome2)

        excess = []
        disjoint = []
        for i in range(m + 1):
            if i in set(genome1_genes.keys()).symmetric_difference(set(genome2_genes.keys())):
                if i <= n:
                    disjoint.append(i)
                else:
                    excess.append(i)
        return disjoint, excess

    @staticmethod
    def compatibility_distance(genome1, genome2):
        """Returns the compatibility distance, a measure of how closely related two genomes are"""
        # create a new dict with all of the genomes' genes
        genome1_genes = genome1.genes()
        genome2_genes = genome2.genes()

        #TODO: temp values - should be configurablr
        c1 = 1
        c2 = 1
        c3 = 1
        N = max(len(genome1_genes), len(genome2_genes))
        #TODO: refactor to functions

        disjoint, excess = Genome.__calculate_mismatching_genes(genome1_genes, genome2_genes)

        intersecting_connection_innovations = \
            set(genome1.connection_genes().keys()).intersection(set(genome2.connection_genes().keys()))
        weight_differences = [abs(genome1_genes[i].weight() - genome2_genes[i].weight()) for i in intersecting_connection_innovations]

        average_weight_difference = np.mean(weight_differences)

        return c1*len(excess)/N + c2*len(disjoint)/N + c3*average_weight_difference

    @staticmethod
    def crossover(genome1, genome2):
        """Returns a genome containing the crossover of both genomes"""
        # Matching genes are inherited randomly, excess and disjoint genes are inherited from the better parent
        # if parents have same fitness, the better parent is the one with the smaller genome
        genome1_genes = genome1.genes()
        genome2_genes = genome2.genes()
        disjoint, excess = Genome.__calculate_mismatching_genes(genome1_genes, genome2_genes)
        matching = set(genome1_genes.keys()).intersection(set(genome2_genes.keys()))

    def __init_node_genes(self, number_of_input_nodes, number_of_output_nodes):
        # TODO: Slight code duplication below
        for _ in range(number_of_input_nodes):
            self.__add_node_gene('SENSOR')  # SENSOR is an input node
        for _ in range(number_of_output_nodes):
            self.__add_node_gene('OUTPUT')

        self.__add_node_gene('BIAS')

    def __add_node_gene(self, node_type):
        # assign the next available node index after current max - once a node is removed (if became isolated), it's index isn't re-assigned
        self._max_used_node_index += 1
        new_node_index = self._max_used_node_index
        new_node_gene = NodeGene(node_type, new_node_index)
        self._node_genes[new_node_index] = new_node_gene
        logging.info("New node gene added: " + str(new_node_gene))
        return new_node_index

    def __delete_node_gene(self, node_index):
        assert node_index in self._node_genes
        # we only delete nodes if they became isolated after a delete_connection mutation.
        # note that all nodes start as isolated nodes after the __add_node mutation.
        assert self._node_genes[node_index].is_isolated()

        logging.info("Node gene deleted: " + str(self._node_genes[node_index]))
        del self._node_genes[node_index]

    def __add_connection_gene(self, source, dest, weight, enabled = True):
        assert source in self._node_genes
        assert dest in self._node_genes

        new_connection_gene = ConnectionGene(source, dest, weight, True)
        self._connection_genes[(source, dest)] = new_connection_gene
        self._node_genes[source].add_connection_to(dest)
        self._node_genes[dest].add_connection_from(source)
        # TODO: implement str(connection) and log it instead
        logging.info("New connection gene added: " + str(new_connection_gene))

    def __delete_connection_gene(self, source, dest):
        assert source in self._node_genes
        assert dest in self._node_genes

        if (source, dest) in self._connection_genes:
            logging.info("Connection gene deleted: " + str(self._connection_genes[(source, dest)]))
            del self._connection_genes[(source, dest)]

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
            logging.debug("Can't delete connection gene %s - not found", str((source, dest)))

    #TODO: not good! check innovation before creating a new connection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __mutate_add_connection(self):
        # OUTPUT neurons can't be a source.
        # TODO: maybe allow output as source.
        possible_sources = [node_index for node_index in self._node_genes.keys() if self._node_genes[node_index].type != 'OUTPUT']
        # INPUT\BIAS neurons can't be a destination.
        possible_destinations = [node_index for node_index in self._node_genes.keys()
                                 if self._node_genes[node_index].type not in ['BIAS', 'INPUT']]
        # Two edges with the same source and destination are not possible.
        possible_edges = set(product(possible_sources, possible_destinations)) - set(self._connection_genes.keys())
        if not possible_edges:
            logging.debug("No possible edges. Possible sources: %s, possible destinations: %s, current edges: %s",
                          str(possible_sources), str(possible_destinations), str(self._connection_genes.keys()))
        else:
            source, dest = random.choice(possible_edges)  # randomly choose one edge from the possible edges
            # TODO: normal is just a place-holder distribution. have it configurable
            # TODO: I assume that the new connection gene is always enabled after the mutation
            self.__add_connection_gene(source, dest, np.random.normal(), True)

    def __mutate_delete_connection(self, source, dest):
        #TODO: don't have source, dest as input but rather choose a connection randomly. Assert instead ifelse in __delete_connection_gene
        return self.__delete_connection_gene(source, dest)

    def __mutate_add_node(self):
        if not self._connection_genes:
            logging.debug("add_note mutation failed: no connection genes to split")
        else:
            old_connection = random.choice(list(self._connection_genes.values()))
            old_source, old_dest = old_connection.to_edge_tuple()
            #TODO: CHECK IF THIS INNOVATION ALREADY EXISTS IN POPULATION!!! (LIKE IN ADD CONNECTION)
            new_node_index = self.__add_node_gene('HIDDEN')
            # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
            self.__add_connection_gene(old_source, new_node_index, 1, True)
            # the new connection leading out of the new node from to the old dest has
            # the old connection's weight according to the NEAT paper
            self.__add_connection_gene(new_node_index, old_dest, old_connection.weight, True)
            old_connection.disable()

    def __mutate_connection_weight(self):
        connection_gene = random.choice(self._connection_genes)
        connection_gene.weight += random.normalvariate(0, 1)         # TODO: assignment to weight should work with property.setter
        # TODO: standard normal distribution was arbitrary, better have config file








