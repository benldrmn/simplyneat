from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder

def foofoo(x):
    return 1

if __name__ == '__main__':
    """Testing the basic node and connection addition, works fine"""
    config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 5, 'number_of_output_nodes': 3})
    genome = Genome(config=config)
    breeder = Breeder(config=config)
    node_genes = genome.node_genes
    genome.add_connection_gene(node_genes[0], node_genes[5], 5)
    breeder.__mutate_add_node(genome)             # should split the 0->5 edge
    genome.add_connection_gene(node_genes[(0, 5)], node_genes[7], 12)
    genome.add_connection_gene(node_genes[0], node_genes[6], 6)
    genome.add_connection_gene(node_genes[1], node_genes[6], 7)
    genome.add_connection_gene(node_genes[-1], node_genes[5], 4)        # connect bias
    genome.add_connection_gene(node_genes[4], node_genes[4], 8)         # self loop
    print(genome)




