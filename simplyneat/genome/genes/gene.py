class Gene:
    _current_innovation_number = 0  # TODO: currently not thread safe

    def __init__(self):
        self._innovation = Gene._current_innovation_number
        Gene._current_innovation_number += 1

    @property
    def innovation(self):
        return self._innovation
