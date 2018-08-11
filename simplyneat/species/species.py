from random import choice

from simplyneat.organism.organism import Organism


class Species:
    def __init__(self, organisms):
        self._organisms = list(organisms)  # initial organisms in the species
        if not self._organisms:
            raise ValueError("A species must be initialized with at least one organism")
        self._representative = choice(self._organisms)
        # new structural innovations of this generation of the species (refer to section 3.2 in the NEAT paper)
        self._structural_innovations_of_generation = [] #todo: should be global for population, not only species!

    @property
    def organisms(self):
        return self._organisms

    @property
    def representative(self):
        return self._representative         # TODO: maybe 1-center?

    def add_organism(self, organism):
        if not isinstance(organism, Organism):
            raise ValueError("add_organism argument should be an instance of %s, not %s", Organism.__class__, organism.__class__)
        self._organisms.append(organism)
