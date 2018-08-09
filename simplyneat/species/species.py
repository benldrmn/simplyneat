import random
import organism

#TODO: delete\rewrite (not needed now, garbage)
class Species:
    def __init__(self, organisms, number):
        self._organisms = organisms  # initial organisms in the species
        # new structural innovations of this generation of the species (refer to section 3.2 in the NEAT paper)
        self._structural_innovations_of_generation = [] #todo: should be global for population, not only species!
        self.number = number        # species serial number to hold throughout generations
        assert(len(organisms) != 0)
        self._representative = self._organisms[0]        # representative for comparisons

    def get_representative(self):
        return self._representative
#TODO: move below to genome.py
    def __mutate(self, phenotype):
        return 0

    def __remove_connection(self, connections, nodes):
        if not connections:
            return

        random_connection_number = random.choice(connections.keys())
        connection_in_node = connections[random_connection_number].in_node
        connection_out_node = connections[random_connection_number].out_node
        connections.pop(random_connection_number)
        nodes[connection_in_node].number_of_connections -= 1
        nodes[connection_out_node].number_of_connections -= 1

    def __str__(self):
        return "Species number: %d, Organisms: %s" %(self.number, self._organisms)