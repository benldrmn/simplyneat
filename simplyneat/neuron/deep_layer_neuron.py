from simplyneat.neuron.neuron import Neuron

# Neurons deeper than first layer (hidden layer neurons and output neuron)
class DeepLayerNeuron(Neuron):
    #todo: is shared necessary in the SIGMOID_STEEPNESS?
    #SIGMOID_STEEPNESS = theano.shared(4.9, borrow=True, dtype=theano.config.floatX)  # as defined in the NEAT paper, page 15 (112)
    SIGMOID_STEEPNESS = 4.9

    input = T.vector('input')
    weights = T.vector('weights')
    bias = T.scalar('bias')
    linearity = theano.function([input, weights, bias], T.dot(input, weights) + bias)

    x = T.scalar('x')
    non_linearity = theano.function([x], T.nnet.sigmoid(SIGMOID_STEEPNESS * x))
