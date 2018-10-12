import copy
import logging
import random
import time

import numpy as np

from simplyneat.agent.agent import Agent
from simplyneat.config.config import init_logger
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeGene, NodeType


class Genome:

    #todo: consider adding node_genes to the ctor so we can define each nodes activation function
    #todo: that way when we mutate a genome, we can get the nodes list and alter one nodes activation
    #TODO: Inherit also the nodes and thus save creating new nodes all the time - i.e. inheriting connection (a,b) also inherits nodes a and b
    def __init__(self, config, connection_genes=None):
        # Constants
        self._number_of_input_nodes = config.number_of_input_nodes
        self._number_of_output_nodes = config.number_of_output_nodes

        self.excess_coefficient, self.disjoint_coefficient, self.weight_difference_coefficient = \
            config.excess_coefficient, config.disjoint_coefficient, config.weight_difference_coefficient

        self._config = config
        #TODO: hacky
        init_logger(self._config.logging_level)

        #todo: create a helper logger init func for all loggers

        if self._number_of_input_nodes <= 0:
            raise ValueError('number of input nodes must be greater than 0')
        if self._number_of_output_nodes <= 0:
            raise ValueError('number of output nodes must be greater than 0')

        self._node_genes = {}  # Key is node_index (as in encode_node @ node_gene)

        # Gene sequences. The Key is the connection index.
        if connection_genes is None:
            self._connection_genes = {}
        else:
            #TODO: find a more elegant solution (deep copy connection genes?)
            # shallow copy the given connection_genes dict. The dict's content will be altered
            # during _init_connection_genes_nodes() - replaced with new connection objects.
            self._connection_genes = copy.copy(connection_genes)

        self._init_node_genes()

        #TODO: can't save in self agent cause tensorflow is not pickleable (doesn't matter -not pickleable becuas config)
        #self._agent = Agent(self._config, self)

        fitness_init_time = time.time()
        self._fitness = config.fitness_function(Agent(self._config, self))
        # We only allow non-negative fitness functions
        assert self._fitness >= 0
        logging.debug("Fitness calculation (%s) took %s sec" % (str(self._fitness), str(time.time() - fitness_init_time)))

    #TODO: rethink
    def save_agent(self):
        return Agent(self._config, self).save()

    @property
    def node_genes(self):
        """Returns a dictionary of node-genes, the keys are indexes and values are node-genes"""
        return self._node_genes

    @property
    def connection_genes(self):
        """Returns a dict of connection genes, where the key is the connection index of the connection gene value"""
        return self._connection_genes

    @property
    def enabled_connection_genes(self):
        return {index: connection for index, connection in self._connection_genes.items()
                if connection.is_enabled()}

    @property
    def size(self):
        #TODO: rethink if needed (confusing defiinition - only connection or also nodes?)
        """Returns the length of the entire genome, connection genes and node genes combined"""
        return len(self._connection_genes) + len(self._node_genes)

    @property
    def fitness(self):
        return self._fitness

    def _init_node_genes(self):
        """Initializes node for the entire genome, i.e. adds INPUT, OUTPUT, BIAS nodes which are present in all
        genomes, and adds necessary nodes for a given dictionary of connection_genes"""

        for i in range(self._number_of_input_nodes):
            self.add_node_gene(NodeType.INPUT, i)
        for i in range(self._number_of_output_nodes):
            self.add_node_gene(NodeType.OUTPUT, self._number_of_input_nodes + i)
        self.add_node_gene(NodeType.BIAS, -1)        # node with index -1 is the bias

        # if we have connections we need to add their corresponding nodes to the genome
        self._init_connection_genes_nodes()

    def _init_connection_genes_nodes(self):
        """Adds, for each connection gene in self.connection_gens, the corresponding nodes needed so the connection gene
        is well-defined within the genome. For example, if we have the connection (1,3) we add the nodes (if not already
        exists) 1 and 3. We also create a new connection which references the newly created nodes instead of the old
        connection which references genes which are not a part of the current genome.
        Only used in the genome's initialization phase. Otherwise use __add_connection_gene or __add_node_gene."""
        assert isinstance(self._connection_genes, dict)
        for connection_gene in self._connection_genes.values():
            source_index = connection_gene.source_node.index
            dest_index = connection_gene.destination_node.index
            # if node doesn't exist, it has to be a hidden node since all of the other types were created and
            # add to the self._node_genes dict in init_node_genes
            if source_index not in self._node_genes.keys():
                self.add_node_gene(NodeType.HIDDEN, source_index)
            if dest_index not in self._node_genes.keys():
                self.add_node_gene(NodeType.HIDDEN, dest_index)

            source_node = self._node_genes[source_index]
            dest_node = self._node_genes[dest_index]

            # create a new node from the newly created source the the newly created dest with the same attributes
            # as connection_gene
            new_connection = ConnectionGene(source_node, dest_node, weight=connection_gene.weight,
                                            split_number=connection_gene.split_number,
                                            innovation=connection_gene.innovation,
                                            enabled_flag=connection_gene.is_enabled())
            assert self._connection_genes[new_connection.index] == connection_gene
            self._connection_genes[new_connection.index] = new_connection

            source_node.add_outgoing_connection(new_connection)
            dest_node.add_incoming_connection(new_connection)

    def add_node_gene(self, node_type, node_index):
        """Adds a single node gene to the genome"""
        #TODO: node index is sometimes tuple and sometimes int (when input\output\bias)
        if node_index in self._node_genes.keys():
            raise ValueError("Node index %s already in genome" % node_index)
        new_node_gene = NodeGene(node_type, node_index)
        self._node_genes[node_index] = new_node_gene
        return new_node_gene

    def add_connection_gene(self, source, dest, weight, split_number, innovation=None, enabled=True):
        """Adds a connection gene. Defaults to None, which implies that the innovation number is not yet known and should
        be updated afterwards when it is known (for example, by the breeder which keeps track of innovations in the
        population it breeds.
        Returns the new connection gene created."""
        if source not in self._node_genes.values():
            raise ValueError("Source node not defined for the genome!")
        if dest not in self._node_genes.values():           # TODO: error - thinks dest is a tuple
            raise ValueError("Destination node not defined for the genome!")

        new_connection_gene = ConnectionGene(source, dest, weight=weight, split_number=split_number,
                                             innovation=innovation, enabled_flag=enabled)
        self._connection_genes[new_connection_gene.index] = new_connection_gene
        source.add_outgoing_connection(new_connection_gene)
        dest.add_incoming_connection(new_connection_gene)

        return new_connection_gene

    def __str__(self):
        return 'Genome Node genes: %s \nGenome Connection genes: %s' %\
               ([str(node) for node in self._node_genes.values()],
                [str(connection) for connection in self._connection_genes.values()])


def compatibility_distance(genome1, genome2):
    """Returns the compatibility distance, a measure of how closely related two genomes are"""
    #TODO: diverging from paper? only count connection genes
    if len(genome1.connection_genes) == 0 and len(genome2.connection_genes) == 0:
        return 0.0

    # create a new dict with all of the genomes' genes with innovation as key for easy processing later on
    innovation_to_connections1 = {connection.innovation: connection for connection in genome1.connection_genes.values()}
    innovation_to_connections2 = {connection.innovation: connection for connection in genome2.connection_genes.values()}
    # N is as defined in the NEAT paper (number of genes in the larger genome. We only consider the connection genes)
    N = max(len(genome1.connection_genes), len(genome2.connection_genes), 1)

    disjoint, excess = calculate_mismatching_genes(innovation_to_connections1, innovation_to_connections2)
    matching_connection_genes_innovations = set(innovation_to_connections1.keys()).intersection(set(innovation_to_connections2.keys()))

    if matching_connection_genes_innovations:
        weight_differences = [abs(innovation_to_connections1[innovation_num].weight - innovation_to_connections2[innovation_num].weight)
                              for innovation_num in matching_connection_genes_innovations]
        average_weight_difference = np.mean(weight_differences)
    else:
        average_weight_difference = 0.0

    # TODO: maybe find prettier solution for coefficients
    distance = genome1.excess_coefficient*len(excess)/N + genome1.disjoint_coefficient*len(disjoint)/N +\
           genome1.weight_difference_coefficient*average_weight_difference
    #TODO: remove below
    if random.random() < 0.00001:
        a = [conn.index for conn in genome1.connection_genes.values()]
        b = [conn.index for conn in genome2.connection_genes.values()]
        print("%s * %s / %s   +   %s * %s / %s    +   %s * %s = %s.\n conn1: %s\n conn2: %s\n intersection: %s\n diff: %s \n\n" %
              (genome1.excess_coefficient, len(excess), N, genome1.disjoint_coefficient, len(disjoint), N,
               genome1.weight_difference_coefficient, average_weight_difference, distance,
               a, b, set(a).intersection(set(b)), set(a).symmetric_difference(set(b))))
    return distance

#TODO: change to return indices instead of innovation? (and change using funcs accordingly)
def calculate_mismatching_genes(innovation_to_connections1, innovation_to_connections2):
    """Returns a pair of lists containing innovation numbers of disjoint and excess connection genes"""
    if not innovation_to_connections1 or not innovation_to_connections2:
        # if one of the dictionaries is empty then everything in the other is excess!
        return [], list(set(innovation_to_connections1.keys()).symmetric_difference(set(innovation_to_connections2.keys())))

    # Max innovation is of connection genes. Node genes don't hold an innovation number.
    max_innovation_genome1 = max(innovation_to_connections1.keys())
    max_innovation_genome2 = max(innovation_to_connections2.keys())

    n = min(max_innovation_genome1, max_innovation_genome2)
    m = max(max_innovation_genome1, max_innovation_genome2)

    # Innovation numbers of genes corresponding to exactly one connection_genes1 or connection_genes2
    non_matching_connection_genes = set(innovation_to_connections1.keys()).symmetric_difference(set(innovation_to_connections2.keys()))
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

