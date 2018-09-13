from simplyneat.agent.tensorflow_agent import TensorflowAgent
from simplyneat.agent.theano_agent import TheanoAgent
from simplyneat.config.config import Config
from simplyneat.genome.genome import Genome
from simplyneat.neat import Neat


def foo(x):
    return 1


if __name__ == '__main__':
    config = Config({'fitness_function': foo, 'number_of_input_nodes': 1, 'number_of_output_nodes': 1,
                     'population_size': 3, 'processes_in_pool': 4})

    genome = Genome(config)
    genome.add_connection_gene(genome.node_genes[0], genome.node_genes[1], 1, 0, 1)
    tf = TensorflowAgent(config, genome)
    #theano = TheanoAgent(config, genome)

    print(tf._activations)
    #print(theano._activations)
    print("KABOOM")
    tf._forward_pass([1])
    #theano._forward_pass([1])
    print(tf._activations)
    #print(theano._activations)
