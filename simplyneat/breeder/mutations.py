import itertools
import logging
import random

from simplyneat.genome.genes.node_gene import NodeType, encode_node
from simplyneat.genome.genome import Genome


"""each mutation receives a genome (and possibly additional parameters such a distribution function). The mutation
mutates the genome inplace.
Returns a list of the changed gene(s) of the given genome.
"""


def mutate_reenable_connection(genome):
    connection_genes = list(genome.connection_genes.values())
    disabled_connection_genes = [connection_gene for connection_gene in connection_genes
                                 if not connection_gene.is_enabled()]
    if not disabled_connection_genes:
        logging.debug("reenable_connection mutation failed: no connection genes to reenable")
        return []
    connection_to_enable = random.choice(disabled_connection_genes)
    connection_to_enable.enable()
    return [connection_to_enable]


def mutate_add_connection(genome, connection_weight_mutation_distribution):
    assert isinstance(genome, Genome)
    # the keys of the genome.node_genes dict are the corresponding nodes' indexes
    possible_sources = genome.node_genes.keys()
    # INPUT\BIAS neurons can't be a destination.
    possible_destinations = [node_index for node_index in genome.node_genes.keys()
                             if genome.node_genes[node_index].node_type not in [NodeType.BIAS, NodeType.INPUT]]
    # Two edges with the same source and destination are not possible.
    possible_edges = list(set(itertools.product(possible_sources, possible_destinations)) -
                          set(map(lambda connection_gene: (connection_gene.source_node.index,
                                                           connection_gene.destination_node.index),
                                  genome.connection_genes.values())))
    if not possible_edges:
        logging.debug("No possible edges. Possible sources: %s, possible destinations: %s, current edges: %s",
                      str(possible_sources), str(possible_destinations), str(genome.connection_genes.values()))
        return []
    else:
        source_index, dest_index = random.choice(possible_edges)  # randomly choose one edge from the possible edges
        source, dest = genome.node_genes[source_index], genome.node_genes[dest_index]

        new_connection = genome.add_connection_gene(source, dest, weight=connection_weight_mutation_distribution(),
                                                    split_number=0, enabled=True)
        return [new_connection]


def mutate_add_node(genome):
    """Takes an existing edge and splits it in the middle with a new node"""
    assert isinstance(genome, Genome)
    if not genome.connection_genes:
        logging.debug("add_note mutation failed: no connection genes to split")
        return []
    else:
        enabled_connections = list(genome.enabled_connection_genes.values())
        if not enabled_connections:
            logging.debug("add_note mutation failed: all connections are disabled")
            return []
        old_connection = random.choice(enabled_connections)
        old_source = old_connection.source_node
        old_dest = old_connection.destination_node
        # A node index is defined by the indexes of the old source and destination of the connection it has split
        # and also the split number of the aforementioned connection so that the index is well-defined and unique
        # the split number makes sure that splitting the same connection more than once wouldn't result with nodes
        # with the same indices. The definition is recursive (we rely on previous node indexes - the old_source and
        # the old_dest, but that's ok because the input and output nodes' indexes are defined independently.
        new_node_index = encode_node(old_source.index, old_dest.index, old_connection.split_number)

        old_connection.split_number += 1
        old_connection.disable()

        new_node = genome.add_node_gene(NodeType.HIDDEN, new_node_index)
        # the new connection leading into the new node from the old source has weight 1 according to the NEAT paper
        new_to_connection = genome.add_connection_gene(old_source, new_node, weight=1, split_number=0, enabled=True)
        # the new connection leading out of the new node from to the old dest has
        # the old connection's weight according to the NEAT paper
        new_from_connection = genome.add_connection_gene(new_node, old_dest, weight=old_connection.weight,
                                                         split_number=0, enabled=True)
        return [new_from_connection, new_to_connection, new_node]


def mutate_connection_weight(genome, weight_mutation_distribution):
    """Alters the weight of a connection"""
    if not genome.connection_genes:
        logging.debug("add_note mutation failed: no connection genes to split")
        return []
    else:
        connection_gene = random.choice(list(genome.connection_genes.values()))
        connection_gene.weight += weight_mutation_distribution()
        return [connection_gene]
        # TODO: read 4.1 better to understand how this works
