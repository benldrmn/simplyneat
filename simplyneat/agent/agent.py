import pickle
from scipy.special import expit

import numpy as np
import tensorflow as tf

from simplyneat.config.config import LoggingLevel
from simplyneat.genome.genes.node_gene import NodeType


class Agent:

    def __init__(self, config, genome):
        self._config = config
        self._genome = genome

    #TODO: pickle genome instead (it's inside agent anyway)
    def save(self):
        with open('temp.p', 'wb') as f:
            pickle.dump(self._genome.connection_genes, f)

    def __enter__(self):
        #TODO: inconsistent naming below
        self._tf_agent = _NumpyAgent(self._config, self._genome)
        return self._tf_agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tf_agent.close()


class _TensorflowAgent:

    def __init__(self, config, genome):
        self._sorted_outputs_list = self._define_order_on_outputs(genome)
        self._layer_to_node_genes_dict = self._divide_nodes_to_layers(genome.node_genes.values())
        self._activations = {node: 0 for node in genome.node_genes.values()}
        self._activation_functions = {}
        self.node_to_tf_scalar = {node: tf.placeholder(dtype=tf.float32, shape=()) for node
                                  in genome.node_genes.values()}
        self._create_activation_functions(genome)
        self._session = self._init_session(config)
        # no need to run tf.global_variables_initializer with session since we don't use variables in the network

    def next_move(self, inputs):
        outputs = self._forward_pass(inputs)
        return np.argmax(outputs)

    def close(self):
        return self._session.close()

    def _init_session(self, config):
        if config.logging_level <= LoggingLevel.DEBUG:
            session_config = tf.ConfigProto(log_device_placement=True)
        else:
            session_config = tf.ConfigProto()

        session_config.gpu_options.allow_growth = True
        self._session = tf.Session(config=session_config)
        return self._session

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
                    self._activations[node] = node_activation
                    activation_function_inputs[self.node_to_tf_scalar[node]] = node_activation
                else:
                    for connection in node.enabled_incoming_connections:
                        source_node = connection.source_node
                        activation_function_inputs[self.node_to_tf_scalar[source_node]] = self._activations[source_node]
                    node_activation = self._session.run(self._activation_functions[node], feed_dict=activation_function_inputs)
                current_layer_new_activations[node] = node_activation
            # update activations for the next layer to use
            # we don't update during the previous loop since recurrent loop of the same layer should use old activations
            for node, activation in current_layer_new_activations.items():
                self._activations[node] = activation

        outputs_activations = []
        for output_node in self._sorted_outputs_list:
            outputs_activations.append(self._activations[output_node])
        return outputs_activations

    @staticmethod
    def _define_order_on_outputs(genome):
        # from output node 0 to output node #outputs - 1
        sorted_outputs_list = []
        for node_gene in genome.node_genes.values():
            if node_gene.node_type == NodeType.OUTPUT:
                sorted_outputs_list.append(node_gene)
        list.sort(sorted_outputs_list, key=lambda node: node.index)
        return sorted_outputs_list

    # Layer 0 contains all and only input nodes. Layer number might not be contagious (i.e. layers: 0, 1, 3...)
    def _divide_nodes_to_layers(self, node_genes):
        node_to_longest_acyclic_path_len = {}  # key: node, value: longest white path to node TODO: explain

        color = {node: 'not visited' for node in node_genes}
        for node in node_genes:
            if node.node_type == NodeType.INPUT:
                self._get_longest_acyclic_paths(node, 0, color, node_to_longest_acyclic_path_len)

        layer_to_node_genes = {}     # key: layer number, value: nodes list
        for node, longest_acyclic_path in node_to_longest_acyclic_path_len.items():
            if longest_acyclic_path not in layer_to_node_genes:
                layer_to_node_genes[longest_acyclic_path] = []
            layer_to_node_genes[longest_acyclic_path].append(node)

        return layer_to_node_genes

    def _get_longest_acyclic_paths(self, node, path_len, color, node_to_longest_acyclic_path_len):
        if color[node] == 'visited':
            return

        color[node] = 'visited'

        if node not in node_to_longest_acyclic_path_len or node_to_longest_acyclic_path_len[node] < path_len:
            node_to_longest_acyclic_path_len[node] = path_len

            for connection in node.enabled_outgoing_connections:
                dest = connection.destination_node
                if color[dest] != 'visited':  # if doesn't create a cycle
                    self._get_longest_acyclic_paths(dest, path_len + 1, color, node_to_longest_acyclic_path_len)

        color[node] = 'not visited'


class _NumpyAgent:

    def __init__(self, config, genome):
        self._sorted_outputs_list = self._define_order_on_outputs(genome)
        self._layer_to_node_indices_dict = self._divide_nodes_to_layers(genome.node_genes.values())
        self._activations = {node.index: 0 for node in genome.node_genes.values()}
        self._node_index_to_gene = genome.node_genes

    def next_move(self, inputs):
        outputs = self._forward_pass(inputs)
        return np.argmax(outputs)

    def close(self):
        pass

    def _forward_pass(self, inputs):
        # from layer 0 (inputs) onwards
        sorted_layer_numbers = sorted(self._layer_to_node_indices_dict.keys())
        for layer_num in sorted_layer_numbers:
            current_layer_new_activations = {}
            for node_index in self._layer_to_node_indices_dict[layer_num]:
                node = self._node_index_to_gene[node_index]
                if node.node_type == NodeType.INPUT:
                    # TODO: relying on node index to be a 0 to #inputs -1 integer - not good
                    #TODO: REMOVE self._activ[node] = in input\bias nodes
                    node_activation = inputs[node.index]
                    self._activations[node.index] = node_activation
                elif node.node_type == NodeType.BIAS:  # BIAS node activation is always 1.0
                    node_activation = 1.0
                    self._activations[node.index] = node_activation
                else:
                    node_activation = 0.0
                    for connection in node.enabled_incoming_connections:
                        source_node = connection.source_node
                        node_activation += connection.weight * self._activations[source_node.index]
                    node_activation = sigmoid(node_activation)

                current_layer_new_activations[node.index] = node_activation
            # update activations for the next layer to use
            # we don't update during the previous loop since recurrent loop of the same layer should use old activations
            for node_index, activation in current_layer_new_activations.items():
                self._activations[node_index] = activation

        outputs_activations = []
        for output_node in self._sorted_outputs_list:
            outputs_activations.append(self._activations[output_node.index])
        return outputs_activations

    @staticmethod
    def _define_order_on_outputs(genome):
        # from output node 0 to output node #outputs - 1
        sorted_outputs_list = []
        for node_gene in genome.node_genes.values():
            if node_gene.node_type == NodeType.OUTPUT:
                sorted_outputs_list.append(node_gene)
        list.sort(sorted_outputs_list, key=lambda node: node.index)
        return sorted_outputs_list

    # Layer 0 contains all and only input nodes. Layer number might not be contagious (i.e. layers: 0, 1, 3...)
    def _divide_nodes_to_layers(self, node_genes):
        node_to_longest_acyclic_path_len = {}  # key: node index, value: longest white path to node TODO: explain

        color = {node.index: 'not visited' for node in node_genes}
        for node in node_genes:
            if node.node_type == NodeType.INPUT:
                self._get_longest_acyclic_paths(node, 0, color, node_to_longest_acyclic_path_len)

        layer_to_node_genes = {}     # key: layer number, value: nodes list
        for node_index, longest_acyclic_path in node_to_longest_acyclic_path_len.items():
            if longest_acyclic_path not in layer_to_node_genes:
                layer_to_node_genes[longest_acyclic_path] = []
            layer_to_node_genes[longest_acyclic_path].append(node_index)

        return layer_to_node_genes

    def _get_longest_acyclic_paths(self, node, path_len, color, node_to_longest_acyclic_path_len):
        if color[node.index] == 'visited':
            return

        color[node.index] = 'visited'

        if node.index not in node_to_longest_acyclic_path_len or node_to_longest_acyclic_path_len[node.index] < path_len:
            node_to_longest_acyclic_path_len[node.index] = path_len

            for connection in node.enabled_outgoing_connections:
                dest = connection.destination_node
                if color[dest.index] != 'visited':  # if doesn't create a cycle
                    self._get_longest_acyclic_paths(dest, path_len + 1, color, node_to_longest_acyclic_path_len)

        color[node.index] = 'not visited'

def sigmoid(x):
    return expit(x)
