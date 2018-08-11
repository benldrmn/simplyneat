from simplyneat.genome.genome import Genome


class Organism:
    def __init__(self, number_of_input_nodes, number_of_output_nodes, organism_index, fitness_function):
        self._genome = Genome(number_of_input_nodes, number_of_output_nodes)
        self._organism_index = organism_index
        # TODO: for now, fitness_function is passed to the constructor

    @property
    def genome(self):
        return self._genome

