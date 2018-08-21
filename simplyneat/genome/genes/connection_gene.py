import logging


class ConnectionGene:

    _current_innovation_number = 0

    def __init__(self, source_node, destination_node, weight, enabled_flag=True):
        self._source_node = source_node             # the index of the source node
        self._dest_node = destination_node          # the index of the dest node
        self._weight = weight
        self._enabled = enabled_flag
        self._innovation = ConnectionGene._current_innovation_number
        ConnectionGene._current_innovation_number += 1

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
        return self._source_node, self._dest_node

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
        return "(Connection source: %s, destination: %s, weight: %d, enabled: %s)" % (self._source_node, self._dest_node, self.weight, str(self._enabled))
        # return "Connection source: %s, destination: %s, weight: %d, innovation number: %d, enabled: %s" %\
        #         (self._source_node, self._dest_node, self.weight, self._innovation, str(self._enabled))

    __repr__ = __str__

