from simplyneat.agent.agent import _TensorflowAgent
from simplyneat.config.config import Config, LoggingLevel
from simplyneat.genome.genome import Genome


def foo(x):
    return 1


if __name__ == '__main__':
    config = Config({'fitness_function': foo, 'number_of_input_nodes': 1, 'number_of_output_nodes': 1,
                     'population_size': 3, 'processes_in_pool': 4, 'logging_level': LoggingLevel.DEBUG})

    genome = Genome(config)
    genome.add_connection_gene(genome.node_genes[0], genome.node_genes[1], 1, 0, 1)
    tf = _TensorflowAgent(config, genome)

    print(tf._activations)
    print("KABOOM")
    tf._forward_pass([1])
    print(tf._activations)

    tf.close()
