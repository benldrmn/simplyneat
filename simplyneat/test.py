from simplyneat.genome.genome import Genome

genome = Genome(1, 1)
connection_genes = genome.connection_genes
print(connection_genes)

genome.add_connection_gene(0, 4, 1)
connection_genes = genome.connection_genes
print(connection_genes)
