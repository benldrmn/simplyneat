from simplyneat.genome.genome import Genome


class Organism:
    def __init__(self, number_of_input_nodes, number_of_output_nodes, organism_index):
        self._genome = Genome(number_of_input_nodes, number_of_output_nodes)
        self._organism_index = organism_index

    @property
    def genome(self):
        return self._genome





