import random
#TODO: delete\rewrite (not needed now, garbage)
class Specie:
    def __init__(self, organisms):
        self._organisms = organisms  # initial organisms in the specie
        # new structural innovations of this generation of the specie (refer to section 3.2 in the NEAT paper)
        self._structural_innovations_of_generation = [] #todo: should be global for population, not only specie!

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


