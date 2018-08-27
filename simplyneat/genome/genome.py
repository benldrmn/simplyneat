import itertools
import logging
import random
import numpy as np
import copy
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene, NodeType
# from simplyneat.agent.neuralnet import TheanoAgent        # TODO: remove after testing


class Genome:

    _current_genome_number = 0

    #todo: consider adding node_genes to the ctor so we can define each nodes activation function
    #todo: that way when we mutate a genome, we can get the nodes list and alter one nodes activation
    #TODO: Inherit also the nodes and thus save creating new nodes all the time - i.e. inheriting connection (a,b) also inherits nodes a and b
    def __init__(self, config, connection_genes=None):
        # Constants
        self._number_of_input_nodes = config.number_of_input_nodes
        self._number_of_output_nodes = config.number_of_output_nodes
        self.excess_coefficient, self.disjoint_coefficient, self.weight_difference_coefficient = \
            config.excess_coefficient, config.disjoint_coefficient, config.weight_difference_coefficient
        self._weight_mutation_distribution = config.weight_mutation_distribution            # for mutation of changing weights
        self._connection_weight_mutation_distribution = config.connection_weight_mutation_distribution      # for mutation of connection creation
        self.config = config                            # TODO: (property?)left config public on pourpse, for crossover TODO: what?

        self._genome_number = Genome._current_genome_number     # Liron: added for debugging, might wanna keep it anyway
        Genome._current_genome_number += 1

        if self._number_of_input_nodes <= 0:
            raise ValueError('number of input nodes must be greater than 0')
        if self._number_of_output_nodes <= 0:
            raise ValueError('number of output nodes must be greater than 0')

        self._node_genes = {}  # Key is node_index (as in encode_node @ node_gene)

        # Gene sequences. The Key is innovation number.
        if connection_genes is None:
            self._connection_genes = {}
        else:
            # shallow copy the given connection_genes dict. The dict's content will be altered
            # during __init_node_genes() - replaced with new connection objects.
            self._connection_genes = copy.copy(connection_genes)

        self.__init_node_genes()

        # self._neural_net = self.__create_neural_network()     # TODO: remove comment after done debugging
        # self._fitness = config.fitness_function(self._neural_net) #TODO: assuming for now that's the fitness_function's required api
        self._fitness = 1                                       # TODO: remove comment after done debugging

    # def __create_neural_network(self):                        # TODO: remove comment after done debugging
    #     return TheanoAgent(self.config, self)                 # TODO: remove comment after done debugging

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

    @property
    def fitness(self):
        return self._fitness

    @property
    def genome_number(self):
        return self._genome_number

    def __init_node_genes(self):
        """Initializes node for the entire genome, i.e. adds SENSOR, OUTPUT, BIAS nodes which are present in all
        genomes, and adds necessary nodes for a given dictionary of connection_genes"""
        assert self._number_of_output_nodes > 0
        assert self._number_of_input_nodes > 0

        for i in range(self._number_of_input_nodes):
            self.add_node_gene(NodeType.SENSOR, i)  # SENSOR is an input node
        for i in range(self._number_of_output_nodes):
            self.add_node_gene(NodeType.OUTPUT, self._number_of_input_nodes + i)
        self.add_node_gene(NodeType.BIAS, -1)        # node with index -1 is the bias
        logging.debug("Added SENSOR, OUTPUT and BIAS node genes")

        # if we have connections we need to add their corresponding nodes to the genome
        self.__init_connection_genes_nodes()

    def __init_connection_genes_nodes(self):
        """Adds, for each connection gene in self.connection_gens, the corresponding nodes needed so the connection gene
        is well-defined within the genome. For example, if we have the connection (1,3) we add the nodes (if not already
        exists) 1 and 3. We also create a new connection which references the newly created nodes instead of the old
        connection which references genes which are not a part of the current genome.
        Only used in the genome's initialization phase. Otherwise use __add_connection_gene or __add_node_gene."""
        assert isinstance(self._connection_genes, dict)
        for connection_gene in self._connection_genes.values():
            #TODO: The implementation of to_edge_tuple changed for some reason and now it returns a tuple of nodes instead of a tuple of node_indexes, but the code below wasn't changed
            source_index, dest_index = connection_gene.to_edge_tuple()
            # if node doesn't exist, it has to be a hidden node since all of the other types were created and
            # add to the self._node_genes dict in init_node_genes
            if source_index not in self._node_genes.keys():
                self.add_node_gene(NodeType.HIDDEN, source_index)      # add source index
                logging.debug("Added source node gene with index: " + str(source_index))
            if dest_index not in self._node_genes.keys():
                self.add_node_gene(NodeType.HIDDEN, dest_index)        # add dest index
                logging.debug("Added dest node gene with index: " + str(dest_index))

            source_node = self._node_genes[source_index]
            dest_node = self._node_genes[dest_index]

            # create a new node from the newly created source the the newly created dest with the same attributes as connection_gene
            new_connection = ConnectionGene(source_node, dest_node, connection_gene.weight, connection_gene.is_enabled(),
                                            connection_gene.innovation)
            assert self._connection_genes[new_connection.innovation] == connection_gene
            self._connection_genes[new_connection.innovation] = new_connection

            source_node.add_outgoing_connection(new_connection)
            dest_node.add_incoming_connection(new_connection)

    def add_node_gene(self, node_type, node_index):
        """Adds a single node gene to the genome"""
        #TODO: node index is sometimes tuple and sometimes int (when input\output\bias)
        if node_index in self._node_genes.keys():
            raise ValueError("Node index already in genome")
        new_node_gene = NodeGene(node_type, node_index)
        self._node_genes[node_index] = new_node_gene
        logging.info("New node gene added: " + str(new_node_gene))
        return new_node_gene

    def delete_node_gene(self, node_index):
        if node_index not in self._node_genes.keys():
            raise ValueError("Node index not in genome")
        # we only delete nodes if they became isolated after a delete_connection mutation.
        # note that all nodes start as isolated nodes after the __add_node mutation.
        assert self._node_genes[node_index].is_isolated()

        logging.info("Node gene deleted: " + str(self._node_genes[node_index]))
        del self._node_genes[node_index]

    def add_connection_gene(self, source, dest, weight, enabled=True, innovation=None):
        """Adds a connection gene.
        By default innovation is None, which means we set the innovation for the new gene by looking at the static
        innovation count, otherwise the new gene's innovation number is innovation. 
        Return the new innovation number."""
        print([dest])
        if not isinstance(dest, NodeGene):
            print("add_connection_gene found that dest ain't a NodeGene")
        if source not in self._node_genes.values():
            raise ValueError("Source node not defined for the genome!")
        if dest not in self._node_genes.values():           # TODO: error - thinks dest is a tuple
            raise ValueError("Destination node not defined for the genome!")
        new_connection_gene = ConnectionGene(source, dest, weight, enabled, innovation)

        self._connection_genes[new_connection_gene.innovation] = new_connection_gene
        source.add_outgoing_connection(new_connection_gene)
        dest.add_incoming_connection(new_connection_gene)

        logging.info("New connection gene added: " + str(new_connection_gene))
        return new_connection_gene.innovation

    def __str__(self):
        return '[Genome number: %s \nNode genes: %s \nConnection genes: %s]' \
               % (self._genome_number, self._node_genes, self._connection_genes)

    def __repr__(self):
        return 'Genome number: %s' % self._genome_number


def compatibility_distance(genome1, genome2):
    """Returns the compatibility distance, a measure of how closely related two genomes are"""
    if genome1.size == 0 and genome2.size == 0:
        logging.info("genome1: %s AND genome2: %s both have 0 genes and hence have compitability distance of 0"
                     % (str(genome1), str(genome2)))
        return 0.0
    # create a new dict with all of the genomes' genes
    connection_genes1 = genome1.connection_genes
    connection_genes2 = genome2.connection_genes
    # N is as defined in the NEAT paper (number of genes in the larger genome)
    N = max(genome1.size, genome2.size)

    disjoint, excess = calculate_mismatching_genes(connection_genes1, connection_genes2)
    matching_connection_genes_innovations = set(genome1.connection_genes.keys()).intersection(
        set(genome2.connection_genes.keys()))

    if matching_connection_genes_innovations:
        weight_differences = [abs(connection_genes1[innovation_num].weight - connection_genes2[innovation_num].weight)
                              for innovation_num in matching_connection_genes_innovations]
        average_weight_difference = np.mean(weight_differences)
    else:
        average_weight_difference = 0

    # TODO: maybe find prettier solution for coefficients
    return genome1.excess_coefficient*len(excess)/N + \
           genome1.disjoint_coefficient*len(disjoint)/N + \
           genome1.weight_difference_coefficient*average_weight_difference


def calculate_mismatching_genes(connection_genes1, connection_genes2):
    """Returns a pair of lists containing innovation numbers of disjoint and excess connection genes"""
    if not connection_genes1.keys() or not connection_genes2.keys():
        # if one of the dictionaries is empty then everything in the other is excess!
        return [], list(set(connection_genes1.keys()).symmetric_difference(set(connection_genes2.keys())))

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

