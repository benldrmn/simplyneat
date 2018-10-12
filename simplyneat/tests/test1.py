import sys
sys.path.append('../')

from simplyneat.config.config import Config
from simplyneat.genome.genes.connection_gene import ConnectionGene
from simplyneat.genome.genes.node_gene import NodeType, NodeGene
from simplyneat.genome.genome import Genome
from simplyneat.agent.neuralnet import TheanoAgent
import gym
from simplyneat.neat import Neat

def fitness(agent):
    done = False
    env = gym.make('SpaceInvaders-ram-v0')
    total_reward = 0
    observation = env.reset()

    while not done:
        next_move = agent.next_move(observation)
        observation, reward, done, info = env.step(next_move)
        total_reward += reward

    env.close()
    return total_reward

if __name__ == '__main__':
    neat = Neat(Config({'fitness_function': fitness, 'number_of_input_nodes': 256, 'number_of_output_nodes': 6}))
    print(neat.run())



#  n0 = NodeGene(NodeType.INPUT, 0)
   #  n1 = NodeGene(NodeType.OUTPUT, 1)
   #  n2 = NodeGene(NodeType.HIDDEN, 2)
   #  n4 = NodeGene(NodeType.HIDDEN, 4)
   #  n3tag = NodeGene(NodeType.OUTPUT, 3)
   #
   #  k0 = NodeGene(NodeType.INPUT, 0)
   #  k1 = NodeGene(NodeType.OUTPUT, 1)
   #  k2 = NodeGene(NodeType.OUTPUT, 2)
   #  k3 = NodeGene(NodeType.OUTPUT, 3)
   #
   #  connections = {0: ConnectionGene(k0, k1, -10), 1: ConnectionGene(k0, k2, 0), 2:ConnectionGene(k0, k3, 10)}
   #
   #  connections4 = {0:ConnectionGene(n0,n2,10), 1:ConnectionGene(n2,n1,20), 2:ConnectionGene(n1,n2,30),
   #                  3:ConnectionGene(n0,n1,40), 4:ConnectionGene(n2,n4,60), 5:ConnectionGene(n0,n4,50)}
   #  config = Config({'number_of_input_nodes': 1, 'number_of_output_nodes': 3, 'fitness_function': lambda x: 1})
   #  genome = Genome(config, connections)
   #  t=TheanoAgent(config, genome)
   #  print(t)
   #  print(t.next_move([1]))
   #  print(t.next_move([10]))
   # # print(t.eval([100]))
   #  # genome2 = Genome(config, connections2)
   #  # print(genome2)
   #
   #  #NeuralNet(config, genome1)(
   #
   #  # genome3 = Genome.breed(genome1, genome2, 1, 0)
   #  # # genome3 should contain connections [ConnectionGene(1, 6, 7), ConnectionGene(2, 6, 8), ConnectionGene(3, 6, 9)] and one of [ConnectionGene(0, 5, 0), ConnectionGene(0, 5, -6)]
   #  # print(genome3)
   #  #
   #  # genome3.mutate_add_node()
   #  # print(genome3)
   #
   #
