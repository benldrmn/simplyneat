from simplyneat.genome.genes.gene import Gene
import logging


class ConnectionGene(Gene):
    def __init__(self, source_node, destination_node, weight, enabled_flag):
        super().__init__()
        self._source_node = source_node
        self._dest_node = destination_node
        #TODO: consider changing weight to _weight and have it as a @property
        self._weight = weight
        self._enabled = enabled_flag

    @property
    def weight(self):
        return self._weight

    def to_edge_tuple(self):
        return self._source_node, self._dest_node

    def enable(self):
        self.__change_enable_state(True)

    def disable(self):
        self.__change_enable_state(False)

    def is_enabled(self):
        return self._enabled is True

    def __change_enable_state(self, change_to_enabled):
        prev_state = 'ENABLED' if self._enabled else prev_state = 'DISABLED'
        if change_to_enabled:
            new_state = 'ENABLED'
            self._enabled = True
        else:
            new_state = 'DISABLED'
            self._enabled = False
        logging.debug("Connection gene: %s changed from %s to %s", str(self), prev_state, new_state)


    def __str__(self):
        return "Connection source: %d, destination: %d, weight: %d, innovation number: %d, enabled: %s" %\
                (self._source_node, self._dest_node, self.weight, self._innovation, str(self._enabled))