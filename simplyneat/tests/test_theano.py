from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.breeder.breeder import Breeder
from simplyneat.agent.neuralnet import TheanoAgent
import gym
from simplyneat.neat import Neat

if __name__ == '__main__':
    """Testing the basic node and connection addition, works fine"""
    config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 3, 'number_of_output_nodes': 1})
    genome = Genome(config=config)
    breeder = Breeder(config=config)
    node_genes = genome.node_genes
    genome.add_connection_gene(node_genes[0], node_genes[3], 10)

    theano_agent = TheanoAgent(config=config, genome=genome)
    print(theano_agent._activations)


