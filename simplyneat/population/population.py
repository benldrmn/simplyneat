import random
import species


class Population:
    def __init__(self, size, distance_threshold, constants):
        self.size = size                                    # N total population size
        self._list_of_species = []                          # a list of species
        self._organisms = []                                # a list of organisms
        self._distance_threshold = distance_threshold       # delta_t
        self._constants = constants                         # c_1, c_2, c_3 for distance computation
        self._organism_to_species = {}                      # mapping to check an organism's species
        self._species_to_size = {}                          # mapping from species to its size

    def __compute_distance(self, organism1, organism2):
        (node_genes1, connection_genes1) = organism1.get_genome().get_genes()
        (node_genes2, connection_genes2) = organism2.get_genome().get_genes()
        max_innovation1 = organism1.get_genome().get_max_innovation()
        max_innovation2 = organism2.get_genome().get_max_innovation()
        max_innovation = min(max_innovation1, max_innovation2)      # gene is excess iff innovation > max_innovation

        mismatching_genes = [gene for gene in node_genes1 if gene not in node_genes2] + \
                            [gene for gene in node_genes2 if gene not in node_genes1] + \
                            [gene for gene in connection_genes1 if gene not in connection_genes2] + \
                            [gene for gene in connection_genes2 if gene not in connection_genes1]
        # TODO: node_genes and connection_genes are dictionaries, so for and in don't work like in lists. fix this.
        excess_genes = [gene for gene in mismatching_genes if gene.get_innovation() >= max_innovation]
        # TODO: not sure if greater or equal, or just greater than
        disjoint_genes = [gene for gene in mismatching_genes if gene.get_innovation() < max_innovation]
        weight_difference = 0 #TODO: fill this in

        return self._constants[0] * len(excess_genes) / self.size + \
               self._constants[1] * len(disjoint_genes) / self.size + \
               self._constants[2] * weight_difference / self.size

    def __add_organism(self, organism):
        assert(organism not in self._organisms)
        logging.info("New organism added: "+str(organism))
        self._organisms.append(organism)
        self.assign_species(organism)

    def assign_species(self, organism):
        """randomly goes over list of species checking if within distance threshold, 
        returns 0 if assigned to existing species otherwise returns new species number"""
        logging.info("Assigned organism to species: " + str(organism) + str(species))
        indexes = list(range(len(self._list_of_species)))
        random.shuffle(indexes)     # random permutation of indexes
        for index in indexes:
            representative = self._list_of_species[index].get_representative()
            distance = self.__compute_distance(representative, organism)
            if distance < self._distance_threshold:     # found to be in species
                self._organism_to_species[organism] = self._list_of_species[index]
                #self.list_of_species[index].add_organism(organism)     #TODO: implement
                return 0

        # this is a new species!
        self.__add_species(organism)
        return len(self._list_of_species)       # new species number

    def __add_species(self, organism):
        new_species = Species([organism], len(self._list_of_species)+1)
        logging.info("New species added: %s" % new_species)
        self._organism_to_species[organism] = new_species
        self._list_of_species.append(new_species)

    def new_species_sizes(self):
        #TODO: implement this, maybe do it in generation?

