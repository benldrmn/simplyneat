import logging


class ConnectionGene:

    def __init__(self, source_node, destination_node, weight, split_number, innovation, enabled_flag=True):
        self._source_node = source_node
        self._dest_node = destination_node
        self._weight = weight
        self._split_number = split_number
        self._enabled = enabled_flag
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

    @property
    def split_number(self):
        return self._split_number

    @split_number.setter
    def split_number(self, new_split_number):
        self._split_number = new_split_number

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
        return "[%s,%s,%s,%s,%s]" \
               % (self._source_node.node_index, self._dest_node.node_index, self._weight, self._enabled, str(self._innovation))

    def __key(self):
        return self._source_node, self._dest_node, self._weight, self._enabled, self._innovation

    def __eq__(self, y):
        return self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    __repr__ = __str__

