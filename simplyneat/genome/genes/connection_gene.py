import numbers


class ConnectionGene:

    def __init__(self, source_node, destination_node, weight, split_number, innovation, enabled_flag=True):
        if not isinstance(weight, numbers.Number):
            raise ValueError("Given weight is not a number")
        if not isinstance(split_number, int):
            raise ValueError("Split number is not an integer")
        self._source_node = source_node
        self._dest_node = destination_node
        self._weight = weight
        self._split_number = split_number
        self._enabled = enabled_flag
        self._innovation = innovation
        self._index = (self._source_node.index, self._dest_node.index)

    @property
    def index(self):
        return self._index

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
        return self._change_enabled_flag(True)

    def disable(self):
        return self._change_enabled_flag(False)

    def is_enabled(self):
        return self._enabled is True

    def _change_enabled_flag(self, new_flag):
        """Changes the enabled flag to new flag.
        Returns true if the flag was actually changed (i.e. wasn't already == new_flag). Else, returns false."""
        prev_flag = self._enabled
        self._enabled = new_flag
        return prev_flag != new_flag

    def __str__(self):
        return "Connection Gene: %s,%s,%s,%s,%s" \
               % (self._index, self._weight, self._split_number, self._enabled, str(self._innovation))

    def __key(self):
        return self._source_node, self._dest_node, self._weight, self._split_number, self._enabled, self._innovation

    def __eq__(self, y):
        return isinstance(y, self.__class__) and self.__key() == y.__key()

    # def __hash__(self):
    #     return hash(self.__key())
