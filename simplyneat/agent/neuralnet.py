import numpy as np
import theano
from theano import tensor as T
from simplyneat.genome.genes.node_gene import NodeType


class TheanoAgent:
    def __init__(self, config, genome):
        #TODO make sure there are no isolated nodes ? what about inputs? we expect num_inputs
        #TODO: change name of the node_genes property to something that makes it cleat it's a dict
        #TODO: easily broken loop below
        self._sorted_outputs_list = []  # from output node 0 to outputs node #outputs - 1
        #TODO: disgusting code
        for node_gene in genome.node_genes.values():
            if node_gene.node_type == NodeType.OUTPUT:
                self._sorted_outputs_list.append(node_gene)
        list.sort(self._sorted_outputs_list, key=lambda node_gene: node_gene.node_index)

        self._layer_to_node_genes_dict = self.__divide_nodes_to_layers(genome.node_genes.values())
        self._activations = {node: 0 for node in genome.node_genes.values()}

        connection_weights = {connection: theano.shared(value=connection.weight, name=str(connection))
                                    for connection in genome.enabled_connection_genes.values()}
        node_to_theano_scalar = {node: T.scalar(name=str(node), dtype=theano.config.floatX) for node in genome.node_genes.values()}

        # We take as possible inputs all of the nodes in the genome (even though some activations may not depend
        # on all of the nodes) for the sake of simplicity.
        # we sort the inputs by the node's name so we can call the created function in the same well defined manner -
        # i.e. inputs sorted by the node's name string.
        activation_function_inputs = [var for node, var in sorted(node_to_theano_scalar.items(), key=lambda item: str(item[0]))]

        self._activation_functions = {}
        for node in node_to_theano_scalar.keys():
            if node.node_type == NodeType.INPUT or not node.enabled_incoming_connections:
                function_output = node_to_theano_scalar[node]
            else:
                function_output = 0
                for connection in node.enabled_incoming_connections:
                    function_output += connection_weights[connection] * node_to_theano_scalar[connection.source_node]
                function_output = T.nnet.sigmoid(function_output) #TODO: change (configurable)
            #TODO: utilize outputs to have one function per layer instead of one per node
            node_activation_func = theano.function(activation_function_inputs, function_output,
                                                   on_unused_input='ignore', allow_input_downcast=True)
            self._activation_functions[node] = node_activation_func

    def next_move(self, inputs):
        outputs = self.__forward_pass(inputs)
        return np.argmax(outputs)

    def __forward_pass(self, inputs):
        # from layer 0 (inputs) onwards
        #TODO: it's a waste to sort the layer_to_node_genes_dict keys every iteration
        for layer_num in sorted(self._layer_to_node_genes_dict.keys()):
            current_layer_new_activations = {}
            for node_gene in self._layer_to_node_genes_dict[layer_num]:
                if node_gene.node_type == NodeType.INPUT:
                    #TODO: relying on node index to be a 0 to #inputs -1 integer - not good
                    node_activation = inputs[node_gene.node_index]
                    self._activations[node_gene] = node_activation
                elif node_gene.node_type == NodeType.BIAS:  # BIAS node activation is always 1.0
                    node_activation = 1.0
                    self._activations[node_gene] = 1.0
                else:
                    #TODO: inefficient sorting each iteration below
                    node_activation = self._activation_functions[node_gene](*[activation for node, activation in sorted(self._activations.items(), key=lambda item: str(item[0]))])
                current_layer_new_activations[node_gene] = node_activation
            # update activations for the next layer to use
            # we don't update during the previous loop since recurrent loop of the same layer should use old activations
            for node_gene, activation in current_layer_new_activations.items():
                self._activations[node_gene] = activation
        outputs_activations = []
        for output_node in self._sorted_outputs_list:
            outputs_activations.append(self._activations[output_node])
        return outputs_activations

    # Layer 0 contains all and only input nodes. Layer number might not be contagious (i.e. layers: 0, 1, 3...)
    def __divide_nodes_to_layers(self, node_genes):
        node_to_longest_acyclic_path_len = {}  # key: node, value: longest white path to node TODO: explain

        color = {node: 'not visited' for node in node_genes}
        for node in node_genes:
            if node.node_type == NodeType.SENSOR:
                self.__get_longest_acyclic_paths(node, 0, color, node_to_longest_acyclic_path_len)

        layer_to_node_genes = {}     # key: layer number, value: nodes list
        for node, longest_acyclic_path in node_to_longest_acyclic_path_len.items():
            if longest_acyclic_path not in layer_to_node_genes:
                layer_to_node_genes[longest_acyclic_path] = []
            layer_to_node_genes[longest_acyclic_path].append(node)

        return layer_to_node_genes

    def __get_longest_acyclic_paths(self, node, path_len, color, node_to_longest_acyclic_path_len):
        if color[node] == 'visited':
            return

        color[node] = 'visited'

        if node not in node_to_longest_acyclic_path_len or node_to_longest_acyclic_path_len[node] < path_len:
            node_to_longest_acyclic_path_len[node] = path_len

            for connection in node.enabled_outgoing_connections:
                dest = connection.destination_node
                if color[dest] != 'visited':  # if doesn't create a cycle
                    self.__get_longest_acyclic_paths(dest, path_len + 1, color, node_to_longest_acyclic_path_len)

        color[node] = 'not visited'
