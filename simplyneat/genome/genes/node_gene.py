import logging


class NodeGene:

    def __init__(self, node_type, node_index):
        # SENSOR, OUTPUT, HIDDEN or BIAS
        self.type = node_type  # todo: consider enum or something, don't leave as a string. strings suck
        # a number of internal book-keeping, as in the NEAT paper's illustrations
        self._node_index = node_index

        # nodes that connect to this node as destination
        self._has_connections_from = set()
        # nodes that this node connects to as source
        self._has_connections_to = set()

    @property
    def node_index(self):
        return self._node_index

    @property
    def neighbors_to(self):
        return self._has_connections_to

    @property
    def neighbors_from(self):
        return self._has_connections_from

    def add_connection_to(self, destination_node):
        if destination_node in self._has_connections_to:
            raise ValueError("An edge already exists between %s and node number %d", str(self), destination_node)
        self._has_connections_to.add(destination_node)

    def delete_connection_to(self, destination_node):
        if destination_node not in self._has_connections_to:
            raise ValueError("No edge exists between %s and node number %d", str(self), destination_node)
        self._has_connections_to.remove(destination_node)

    def add_connection_from(self, source_node):
        if source_node in self._has_connections_from:
            raise ValueError("An edge already exists between node number %d and %s", source_node, str(self))
        self._has_connections_from.add(source_node)

    def delete_connection_from(self, source_node):
        if source_node not in self._has_connections_from:
            raise ValueError("No edge exists between node number %d and %s", source_node, str(self))
        self._has_connections_from.remove(source_node)

    def is_isolated(self):
        if not self._has_connections_from and not self._has_connections_to:
            return True
        return False

    def __str__(self):
        return "(Node type: %s, index: %s)" % (self.type, self._node_index)
        # return "Node type: %s, index: %d," \
        #        " has connections from: %s," \
        #        " has connections to: %s" %\
        #        (self.type, self._node_index, str(self._has_connections_from), str(self._has_connections_to))

    __repr__ = __str__


def encode_node(prev_source_index, prev_dest_index):
    """Returns new node index based on the edge the node is splitting"""
    return prev_source_index, prev_dest_index



