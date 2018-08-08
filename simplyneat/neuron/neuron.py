import theano
import theano.tensor as T
import numpy as np
import theano.typed_list


class Neuron:

    #current_neuron_serial = theano.shared(0, borrow=True)

    def __init__(self, innovation_number):
        self.innovation = theano.shared(np.int32(innovation_number))
        #self.incoming_links = theano.typed_list.TypedLis
        #self.serial_number = Neuron.current_neuron_serial.get_value()
        #Neuron.current_neuron_serial += 1

    def forward(self, input_vector, weights_vector, bias_scalar):
        return Neuron.non_linearity(Neuron.linearity(input_vector, weights_vector, bias_scalar))

    def type(self):
        pass







