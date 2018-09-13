from abc import ABC, abstractmethod
import numpy as np

from simplyneat.genome.genes.node_gene import NodeType


class NeuralNetworkAgent(ABC):

    def next_move(self, inputs):
        outputs = self._forward_pass(inputs)
        return np.argmax(outputs)

    @abstractmethod
    def _forward_pass(self, inputs):
        pass

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

