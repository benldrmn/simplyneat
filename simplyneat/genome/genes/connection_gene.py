import logging


class ConnectionGene:

    #TODO: https://stackoverflow.com/questions/2080660/python-multiprocessing-and-a-shared-counter, https://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    _current_innovation_number = 0

    def __init__(self, source_node, destination_node, weight, enabled_flag=True, innovation=None):
        self._source_node = source_node
        self._dest_node = destination_node
        self._weight = weight
        self._enabled = enabled_flag
        if innovation is None:
            self._innovation = ConnectionGene._current_innovation_number
            ConnectionGene._current_innovation_number += 1
        else:
            if not isinstance(innovation, int):
                raise ValueError("innovation must be of type int (or None for the next possible innovation)")
            self._innovation = innovation

    @property
    def source_node(self):
        return self._source_node

    @property
    def destination_node(self):
        return self._dest_node

    @property
    def innovation(self):
        return self._innovation

    @innovation.setter
    def innovation(self, innovation):
        self._innovation = innovation

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def to_edge_tuple(self):
        return self._source_node.node_index, self._dest_node.node_index

    def enable(self):
        return self.__change_enabled_flag(True)

    def disable(self):
        return self.__change_enabled_flag(False)

    def is_enabled(self):
        return self._enabled is True

    def __change_enabled_flag(self, new_flag):
        """Changes the enabled flag to new flag.
        Returns true if the flag was actually changed (i.e. wasn't already == new_flag). Else, returns false."""
        prev_flag = self._enabled
        self._enabled = new_flag
        return prev_flag != new_flag

    def __str__(self):
        return "(Connection source: %s, destination: %s, weight: %d, enabled: %s, innovation: %s)" \
               % (self._source_node, self._dest_node, self.weight, str(self._enabled), str(self._innovation))
        # return "Connection source: %s, destination: %s, weight: %d, innovation number: %d, enabled: %s" %\
        #         (self._source_node, self._dest_node, self.weight, self._innovation, str(self._enabled))

    def __key(self):
        return (self._source_node, self._dest_node, self._weight, self._enabled, self._innovation)

    def __eq__(self, y):
        return self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    __repr__ = __str__

