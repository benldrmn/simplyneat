from simplyneat.neuron.neuron import Neuron
from simplyneat.neuron.neuron_types import type
import theano.tensor as T

class BiasNeuron(Neuron):


    def __init__(self, innovation_number):
        super().__init__(innovation_number)

    def type(self):
        return 'BIAS'  # todo: for now return a string, check how it affects performace

    def forward(self):
        return T.scalar()