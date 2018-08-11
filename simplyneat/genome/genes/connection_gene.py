import logging


class ConnectionGene():
    # TODO: not thread safe
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

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def to_edge_tuple(self):
        return self._source_node, self._dest_node

    def enable(self):
        self.__change_enable_state(True)

    def disable(self):
        self.__change_enable_state(False)

    def is_enabled(self):
        return self._enabled is True

    def __change_enable_state(self, change_to_enabled):
        if self._enabled:
            prev_state = 'ENABLED'
        else:
            prev_state = 'DISABLED'
        if change_to_enabled:
            new_state = 'ENABLED'
            self._enabled = True
        else:
            new_state = 'DISABLED'
            self._enabled = False
        logging.debug("Connection gene: %s changed from %s to %s", str(self), prev_state, new_state)

    def __str__(self):
        return "(Connection source: %s, destination: %s, weight: %d, enabled: %s)" % (self._source_node, self._dest_node, self.weight, str(self._enabled))
        # return "Connection source: %s, destination: %s, weight: %d, innovation number: %d, enabled: %s" %\
        #         (self._source_node, self._dest_node, self.weight, self._innovation, str(self._enabled))

    __repr__ = __str__          # TODO: ruins the pourpse of repr, but is useful for testing. after done testing should remove this line

