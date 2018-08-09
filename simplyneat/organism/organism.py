from simplyneat.genome.genome import Genome


class Organism:
    def __init__(self, number_of_input_nodes, number_of_output_nodes, number):
        self._genome = Genome(number_of_input_nodes, number_of_output_nodes)
        self.number = number  # number of organism

    def get_genome(self):
        return self._genome

    def __str__(self):
        return "Organism number: %d" % self.number





