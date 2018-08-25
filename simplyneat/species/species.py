import random

from simplyneat.genome.genome import Genome


class Species:
    def __init__(self, genomes):
        self._genomes = list(genomes)  # initial genomes in the species
        if not self._genomes:
            raise ValueError("A species must be initialized with at least one genome")
            self._representative = random.choice(self._genomes)
            # new structural innovations of this generation of the species (refer to section 3.2 in the NEAT paper)

    @property
    def genomes(self):
        return self._genomes

    @property
    def representative(self):
        return self._representative

    def randomize_representative(self):
        """Sets a random representative. 
        Useful for speciating a new generation of organisms according to old generation representatives, which were chosen at random"""
        self._representative = random.choice(self._genomes)

    def add_genomes(self, genome):
        if not isinstance(genome, Genome):
            raise ValueError("add_genome argument should be an instance of %s, not %s", Genome.__class__, genome.__class__)
        self._genomes.append(genome)

    def reset_genomes(self):
        """Removes all genomes from species while maintaining the previous representative"""
        self._genomes = []
