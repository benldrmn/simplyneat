from simplyneat.genome.genome import *

connections1 = {1:ConnectionGene(0, 5, -5), 7:ConnectionGene(1, 6, 7), 8:ConnectionGene(2, 6, 8), 9:ConnectionGene(3, 6, 9)}
connections2 = {1:ConnectionGene(0, 5, 0), 6:ConnectionGene(0, 6, 6), 10:ConnectionGene(4, 6, 10)}

genome1 = Genome(5, 2, connections1)
print(genome1)
genome2 = Genome(5, 2, connections2)
print(genome2)

genome3 = Genome.crossover(genome1, genome2, 1, 0)
# genome3 should contain connections [ConnectionGene(1, 6, 7), ConnectionGene(2, 6, 8), ConnectionGene(3, 6, 9)] and one of [ConnectionGene(0, 5, 0), ConnectionGene(0, 5, -6)]
print(genome3)

genome3.mutate_add_node()
print(genome3)


