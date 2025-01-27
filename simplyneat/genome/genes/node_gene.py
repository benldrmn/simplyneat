from enum import Enum

class NodeGene:

    def __init__(self, node_type, node_index):
        if node_type not in NodeType:
            raise ValueError("Node type not recognized")
        # INPUT, OUTPUT, HIDDEN or BIAS
        self._type = node_type
        # a number of internal book-keeping, as in the NEAT paper's illustrations
        self._index = node_index
        self._incoming_connections = []
        self._outgoing_connections = []

    @property
    def node_type(self):
        return self._type

    @property
    def index(self):
        return self._index

    @property
    def incoming_connections(self):
        return self._incoming_connections

    @property
    def enabled_incoming_connections(self):
        return list(filter(lambda connection: connection.is_enabled(), self._incoming_connections))

    @property
    def outgoing_connections(self):
        return self._outgoing_connections

    @property
    def enabled_outgoing_connections(self):
        return list(filter(lambda connection: connection.is_enabled(), self._outgoing_connections))

    def add_incoming_connection(self, incoming_connection):
        self._incoming_connections.append(incoming_connection)

    # we have 2 different delete function instead of a unified function as we may have a node may have both connection
    # a->b and b->a and thus there might be an ambiguity in regard to which connection we should delete
    def delete_incoming_connection(self, incoming_connection):
        if incoming_connection not in self._incoming_connections:
            raise ValueError("Node %s has no incoming connections %s", str(self), str(incoming_connection))
        else:
            return self._incoming_connections.remove(incoming_connection)

    def add_outgoing_connection(self, outgoing_connection):
        self._outgoing_connections.append(outgoing_connection)

    def delete_outgoing_connection(self, outgoing_connection):
        if outgoing_connection not in self._outgoing_connections:
            raise ValueError("Node %s has no outgoing connection %s", str(self), str(outgoing_connection))
        else:
            return self._outgoing_connections.remove(outgoing_connection)

    def is_isolated(self):
        if not self._incoming_connections and not self._outgoing_connections:
            return True
        return False

    #TODO: consider more verbose str
    def __str__(self):
        return "Node Gene: %s,%s" % (self._index, self._type)

    def __key(self):
        return self._type, self._index

    def __eq__(self, y):
        return isinstance(y, self.__class__) and self.__key() == y.__key()

    # def __hash__(self):
    #     return hash(self.__key())

#TODO: maybe argument is connection so we can use it to define encoding even if different node activations (internal DS that remmbers how many splits for each activation)
def encode_node(prev_source_index, prev_dest_index, split_number):
    """Returns new node index based on the edge the node is splitting"""
    return prev_source_index, prev_dest_index, split_number


class NodeType(Enum):
    BIAS = 'BIAS'
    INPUT = 'INPUT'
    HIDDEN = 'HIDDEN'
    OUTPUT = 'OUTPUT'
