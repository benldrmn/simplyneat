from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene
from itertools import product
import logging
import random
import numpy as np
import copy

class Genome:
    def __init__(self, number_of_input_nodes, number_of_output_nodes, connection_genes={}):
        # TODO: number of input and output nodes should be in config
        if number_of_input_nodes <= 0:
            raise ValueError('number of input nodes must be greater than 0')
        if number_of_output_nodes <= 0:
            raise ValueError('number of output nodes must be greater than 0')
        self.number_of_input_nodes = number_of_input_nodes
        self.number_of_output_nodes = number_of_output_nodes
        self._node_genes = {}                       # key: node_index (as in __encode_node)
        self._connection_genes = connection_genes   # key: innovation number
        self.__init_node_genes(number_of_input_nodes, number_of_output_nodes, connection_genes)

    def __init_node_genes(self, number_of_input_nodes, number_of_output_nodes, connection_genes):
        """Initializes node for the entire genome, i.e. adds SENSOR, OUTPUT, BIAS nodes which are present in all
        genomes, and adds necessary nodes for a given dictionary of connection_genes"""
        # TODO: number of input and output nodes should be in config
        for i in range(number_of_input_nodes):
            self.__add_node_gene('SENSOR', i)  # SENSOR is an input node
        for i in range(number_of_output_nodes):
            self.__add_node_gene('OUTPUT', number_of_input_nodes + i)
        self.__add_node_gene('BIAS', -1)        # node with index -1 is the bias
        logging.info("Added SENSOR, OUTPUT and BIAS node genes")

        # if we have connections we need to add their corresponding nodes to the genome
        for connection_gene in connection_genes.values():
            source_index, dest_index = connection_gene.to_edge_tuple()
            if source_index not in self._node_genes.keys():
                self.__add_node_gene('HIDDEN', source_index)      # add source index
                logging.info("Added source node gene with index: "+str(source_index))
            if dest_index not in self._node_genes.keys():
                self.__add_node_gene('HIDDEN', dest_index)        # add dest index
                logging.info("Added dest node gene with index: "+str(dest_index))
            source_node = self._node_genes[source_index]
            dest_node = self._node_genes[dest_index]
            source_node.add_connection_to(dest_node)        # add dest to source's neighbors
            dest_node.add_connection_from(source_node)      # add source to dest's neighbors (edge in the opposite direction of course)

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
        # Liron: this may be redundant, I thought the constructor should receive the entire connection-gene-list rather than just adding a single connection
        # maybe this is useful for mutations?
        assert source in self._node_genes
        assert dest in self._node_genes

        new_connection_gene = ConnectionGene(source, dest, weight, True)
        self._connection_genes[new_connection_gene.innovation] = new_connection_gene
        self._node_genes[source].add_connection_to(dest)
        self._node_genes[dest].add_connection_from(source)
        logging.info("New connection gene added: " + str(new_connection_gene))

    def __delete_connection_gene(self, innovation_number):
        # Liron: maybe deleting genes isn't necessary, for now we'll leave it be
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

    # returns a dictionary of node-genes, the keys are indexes and values are node-genes
    @property
    def node_genes(self):
        return self._node_genes

    # returns a dict of connection genes, where the key is the innovation number of the connection gene value
    @property
    def connection_genes(self):
        return self._connection_genes

    @property
    def size(self):
        """Returns the length of the entire genome, connection genes and node genes combined"""
        return len(self._connection_genes) + len(self._node_genes)

    @staticmethod
    def __calculate_mismatching_genes(connection_genes1, connection_genes2):
        """Returns a pair of lists containing innovation numbers of disjoint and excess connection genes"""
        max_innovation_genome1 = max(connection_genes1.keys())      # max innovation is of connection genes
        max_innovation_genome2 = max(connection_genes2.keys())

        n = min(max_innovation_genome1, max_innovation_genome2)
        m = max(max_innovation_genome1, max_innovation_genome2)

        excess = []
        disjoint = []
        for i in range(m + 1):
            if i in set(connection_genes1.keys()).symmetric_difference(set(connection_genes2.keys())):
                if i <= n:
                    disjoint.append(i)
                else:
                    excess.append(i)
        return disjoint, excess

    @staticmethod
    def compatibility_distance(genome1, genome2):
        """Returns the compatibility distance, a measure of how closely related two genomes are"""
        # create a new dict with all of the genomes' genes
        connection_genes1 = genome1.connection_genes()
        connection_genes2 = genome2.connection_genes()

        #TODO: temp values - should be configurablr
        c1 = 1
        c2 = 1
        c3 = 1
        N = max(len(connection_genes1), len(connection_genes2))
        #TODO: refactor to functions

        disjoint, excess = Genome.__calculate_mismatching_genes(connection_genes1, connection_genes2)

        intersecting_connection_innovations = \
            set(genome1.connection_genes().keys()).intersection(set(genome2.connection_genes().keys()))
        weight_differences = [abs(connection_genes1[i].weight() - connection_genes2[i].weight()) for i in intersecting_connection_innovations]

        average_weight_difference = np.mean(weight_differences)

        return c1*len(excess)/N + c2*len(disjoint)/N + c3*average_weight_difference

    # TODO: change fitness1 and fitness2, we already have an adjusted fitness matrix maybe get it from there instead of computing once again
    # TODO: maybe adjusted fitness instead of fitness?
    @staticmethod
    def crossover(genome1, genome2, fitness1, fitness2):
        """Returns a dictionary containing the crossover of connection-genes from both genomes"""
        # Matching genes are inherited randomly, excess and disjoint genes are inherited from the better parent
        # if parents have same fitness, the better parent is the one with the smaller genome
        if fitness2 > fitness1 or (fitness1 == fitness2 and genome2.size() < genome1.size()):
            genome1, genome2 = genome2, genome1
            fitness1, fitness2 = fitness2, fitness1
            # Makes sure that genome1 is the genome of the fitter parent

        connection_genes1 = genome1.connection_genes
        connection_genes2 = genome2.connection_genes
        disjoint, excess = Genome.__calculate_mismatching_genes(connection_genes1, connection_genes2)
        mismatching = disjoint + excess
        matching = set(connection_genes1.keys()).intersection(set(connection_genes2.keys()))
        result_connection_genes = {}

        # matching genes
        for innovation_number in matching:
            if random.choice([True, False]):
                result_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])     # copied because mutations can change values
            else:
                result_connection_genes[innovation_number] = copy.copy(connection_genes2[innovation_number])

        # mismatched genes
        for innovation_number in mismatching:
            if innovation_number in connection_genes1.keys():
                result_connection_genes[innovation_number] = copy.copy(connection_genes1[innovation_number])

        # TODO: return new genome instead of dictionary. connection-genes should be a mapping from edges to connections.
        return Genome(genome1.number_of_input_nodes, genome1.number_of_output_nodes, result_connection_genes)

    #TODO: not good! check innovation before creating a new connection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __mutate_add_connection(self):
        # OUTPUT neurons can't be a source.
        # TODO: maybe allow output as source.
        possible_sources = [node_index for node_index in self._node_genes.keys() if self._node_genes[node_index].type != 'OUTPUT']
        # INPUT\BIAS neurons can't be a destination.
        possible_destinations = [node_index for node_index in self._node_genes.keys()
                                 if self._node_genes[node_index].type not in ['BIAS', 'INPUT']]
        # Two edges with the same source and destination are not possible.
        possible_edges = set(product(possible_sources, possible_destinations)) -\
                         set(map(lambda connection_gene: connection_gene.to_edge_tuple(), self._connection_genes.values()))
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
            new_node_index = self.__add_node_gene('HIDDEN', Genome.__encode_node(old_source, old_dest))     # Split the connection
            # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
            self.__add_connection_gene(old_source, new_node_index, 1, True)
            # the new connection leading out of the new node from to the old dest has
            # the old connection's weight according to the NEAT paper
            self.__add_connection_gene(new_node_index, old_dest, old_connection.weight, True)
            old_connection.disable()

    @staticmethod
    def __encode_node(prev_source, prev_dest):
        """Returns new node index based on the edge the node is splitting"""
        return prev_source, prev_dest           # Liron : TODO: I removed node_index because I think inputs are already indexes

    def __mutate_connection_weight(self):
        connection_gene = random.choice(self._connection_genes.values())
        connection_gene.weight += random.normalvariate(0, 1)         # TODO: assignment to weight should work with property.setter
        # TODO: standard normal distribution was arbitrary, better have config file

    def __str__(self):
        return 'A genome. Node genes: %s, Connection genes: %s' % (self._node_genes, self._connection_genes)

    __repr__ = __str__          # TODO: ruins the pourpse of repr, but is useful for testing. after done testing should remove this line

