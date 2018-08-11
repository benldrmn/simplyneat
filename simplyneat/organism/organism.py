from simplyneat.genome.genome import Genome


class Organism:
    def __init__(self, config):
        self._genome = Genome(config)

    @property
    def genome(self):
        return self._genome

