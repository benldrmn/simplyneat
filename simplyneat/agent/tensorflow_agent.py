from simplyneat.agent.agent import NeuralNetworkAgent
import tensorflow as tf

from simplyneat.genome.genes.node_gene import NodeType


class TensorflowAgent(NeuralNetworkAgent):

    def __init__(self, config, genome):
        self._sorted_outputs_list = self._define_order_on_outputs(genome)
        self._layer_to_node_genes_dict = self._divide_nodes_to_layers(genome.node_genes.values())
        self._activations = {node: 0 for node in genome.node_genes.values()}
        self._activation_functions = {}
        self.node_to_tf_scalar = {node: tf.placeholder(dtype=tf.float32, shape=()) for node
                                  in genome.node_genes.values()}
        self._create_activation_functions(genome)

    def _create_activation_functions(self, genome):
        # shape = () indicates a scalar (a floating point number)
        connection_weights = {connection: tf.constant(value=connection.weight, dtype=tf.float32, shape=())
                              for connection in genome.enabled_connection_genes.values()}

        for node in self.node_to_tf_scalar.keys():
            function_inputs = []
            if node.node_type == NodeType.INPUT or not node.enabled_incoming_connections:
                input_node_tf_scalar = self.node_to_tf_scalar[node]
                function_inputs.append(input_node_tf_scalar)
                node_activation_func = input_node_tf_scalar  # identity function for inputs
            else:
                node_activation_func = 0
                for connection in node.enabled_incoming_connections:
                    source_node_tf_scalar = self.node_to_tf_scalar[connection.source_node]
                    function_inputs.append(source_node_tf_scalar)
                    node_activation_func += connection_weights[connection] * source_node_tf_scalar
                node_activation_func = tf.sigmoid(node_activation_func) #TODO: change (configurable)
            #TODO: utilize outputs to have one function per layer instead of one per node
            self._activation_functions[node] = node_activation_func

    def _forward_pass(self, inputs):
        with tf.Session() as session:
            # from layer 0 (inputs) onwards
            sorted_layer_numbers = sorted(self._layer_to_node_genes_dict.keys())
            for layer_num in sorted_layer_numbers:
                current_layer_new_activations = {}
                for node in self._layer_to_node_genes_dict[layer_num]:
                    activation_function_inputs = {}
                    if node.node_type == NodeType.INPUT:
                        # TODO: relying on node index to be a 0 to #inputs -1 integer - not good
                        node_activation = inputs[node.index]
                        self._activations[node] = node_activation
                        activation_function_inputs[self.node_to_tf_scalar[node]] = node_activation
                    elif node.node_type == NodeType.BIAS:  # BIAS node activation is always 1.0
                        node_activation = 1.0
                        self._activations[node] = 1.0
                        activation_function_inputs[self.node_to_tf_scalar[node]] = node_activation
                    else:
                        for connection in node.enabled_incoming_connections:
                            source_node = connection.source_node
                            activation_function_inputs[self.node_to_tf_scalar[source_node]] = self._activations[source_node]
                        node_activation = session.run(self._activation_functions[node], feed_dict=activation_function_inputs)
                    current_layer_new_activations[node] = node_activation
                # update activations for the next layer to use
                # we don't update during the previous loop since recurrent loop of the same layer should use old activations
                for node, activation in current_layer_new_activations.items():
                    self._activations[node] = activation

            outputs_activations = []
            for output_node in self._sorted_outputs_list:
                outputs_activations.append(self._activations[output_node])
            return outputs_activations
