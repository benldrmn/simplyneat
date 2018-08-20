import random

from simplyneat.genome.genome import Genome


class Species:
    def __init__(self, genomes):
        self._genomes = list(genomes)  # initial genomes in the species
        if not self._genomes:
            raise ValueError("A species must be initialized with at least one genome")
        #TODO: very important - currently new represntative is not assigned anywhere after construction of species!
        self._representative = random.choice(self._genomes)
        # new structural innovations of this generation of the species (refer to section 3.2 in the NEAT paper)
        self._structural_innovations_of_generation = []     # todo: should be global for population, not only species!

    @property
    def genomes(self):
        return self._genomes

    @property
    def representative(self):
        return self._representative
    # TODO: if species is created with just 1 genome (which it most always will be) that genome will necessarily be the representative
    # TODO: maybe we should have the representative be chosen each generation instead? maybe choose a representative by 1-center?

    def add_genomes(self, genome):
        if not isinstance(genome, Genome):
            raise ValueError("add_genome argument should be an instance of %s, not %s", Genome.__class__, genome.__class__)
        self._genomes.append(genome)

    def reset_genomes(self):
        """Removes all genomes from species while maintaining the original representative"""
        self._genomes = []
