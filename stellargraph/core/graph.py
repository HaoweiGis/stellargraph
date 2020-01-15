# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The StellarGraph class that encapsulates information required for
a machine-learning ready graph used by models.

"""
__all__ = ["StellarGraph", "StellarDiGraph", "GraphSchema", "NeighbourWithWeight"]

from typing import Iterable, Any, Mapping, List, Optional, Set
from collections import namedtuple
import pandas as pd

from .. import globalvar
from .schema import GraphSchema
from .pandas_elements import SingleTypeData, NodeData, EdgeData


NeighbourWithWeight = namedtuple("NeighbourWithWeight", ["node", "weight"])

_WeightConfig = namedtuple("_WeightConfig", ["default", "label"])


def _convert_single(type_name, data, *, shared_cols, weight_config, parent_name):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{parent_name}[{type_name!r}]: expected pandas DataFrame, found {type(data)}"
        )

    if weight_config is not None:
        if weight_config.label not in data.columns:
            data = data.assign(**{globalvar.WEIGHT: weight_config.default})
        else:
            data = data.rename(columns={weight_config.label: globalvar.WEIGHT})

    shared = data[shared_cols]
    features = data.drop(columns=shared_cols).to_numpy()
    return SingleTypeData(shared, features)


def _convert_elements(elements, *, default_type, shared_cols, weight_config, name):
    if isinstance(elements, pd.DataFrame):
        elements = {default_type: elements}

    if not isinstance(elements, dict):
        raise TypeError(f"{name}: expected dict, found {type(elements)}")

    return {
        type_name: _convert_single(
            type_name,
            data,
            weight_config=weight_config,
            shared_cols=shared_cols,
            parent_name=name,
        )
        for type_name, data in elements.items()
    }


class StellarGraph:
    """
    StellarGraph class for directed or undirected graph ML models. It stores both
    graph structure and features for machine learning.

    To create a StellarGraph object ready for machine learning, at a
    minimum pass the graph structure to the StellarGraph as a NetworkX
    graph:

    For undirected models::

        Gs = StellarGraph(nx_graph)


    For directed models::

        Gs = StellarDiGraph(nx_graph)


    To create a StellarGraph object with node features, supply the features
    as a numeric feature vector for each node.

    To take the feature vectors from a node attribute in the original NetworkX
    graph, supply the attribute name to the ``node_features`` argument::

        Gs = StellarGraph(nx_graph, node_features="feature")


    where the nx_graph contains nodes that have a "feature" attribute containing
    the feature vector for the node. All nodes of the same type must have
    the same size feature vectors.

    Alternatively, supply the node features as Pandas DataFrame objects with
    the of the DataFrame set to the node IDs. For graphs with a single node
    type, you can supply the DataFrame object directly to StellarGraph::

        node_data = pd.DataFrame(
            [feature_vector_1, feature_vector_2, ..],
            index=[node_id_1, node_id_2, ...])
        Gs = StellarGraph(nx_graph, node_features=node_data)

    For graphs with multiple node types, provide the node features as Pandas
    DataFrames for each type separately, as a dictionary by node type.
    This allows node features to have different sizes for each node type::

        node_data = {
            node_type_1: pd.DataFrame(...),
            node_type_2: pd.DataFrame(...),
        }
        Gs = StellarGraph(nx_graph, node_features=node_data)


    You can also supply the node feature vectors as an iterator of `node_id`
    and feature vector pairs, for graphs with single and multiple node types::

        node_data = zip([node_id_1, node_id_2, ...],
            [feature_vector_1, feature_vector_2, ..])
        Gs = StellarGraph(nx_graph, node_features=node_data)


    Args:
        graph: The NetworkX graph instance.
        node_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the node types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the nodes of the graph to determine
            their type.

        node_type_default: str, optional (default=globals.NODE_TYPE_DEFAULT)
            This is the default node type to use for nodes that do not have
            an explicit type.

        edge_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the edge types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the edges of the graph to determine
            their type.

        edge_type_default: str, optional (default=globals.EDGE_TYPE_DEFAULT)
            This is the default edge type to use for edges that do not have
            an explicit type.

        node_features: str, dict, list or DataFrame optional (default=None)
            This tells StellarGraph where to find the node feature information
            required by some graph models. These are expected to be
            a numeric feature vector for each node in the graph.

    """

    def __init__(
        self,
        graph=None,
        is_directed=False,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
        nodes=None,
        edges=None,
    ):
        if graph is not None or (nodes is None and edges is None):
            # While migrating, maintain complete compatibility by deferring to the networkx
            # implementation

            # Avoid a circular import
            from .graph_networkx import NetworkXStellarGraph

            self._graph = NetworkXStellarGraph(
                graph,
                is_directed,
                edge_weight_label,
                node_type_name,
                edge_type_name,
                node_type_default,
                edge_type_default,
                feature_name,
                target_name,
                node_features,
                dtype,
            )
        else:
            self._graph = None

            if nodes is None:
                nodes = {}
            if edges is None:
                edges = {}

            self._is_directed = is_directed

            self._nodes = NodeData(
                _convert_elements(
                    nodes,
                    default_type=node_type_default,
                    shared_cols=[],
                    weight_config=None,
                    name="nodes",
                )
            )
            self._edges = EdgeData(
                _convert_elements(
                    edges,
                    default_type=edge_type_default,
                    shared_cols=[globalvar.SOURCE, globalvar.TARGET, globalvar.WEIGHT],
                    weight_config=_WeightConfig(float("nan"), edge_weight_label),
                    name="edges",
                )
            )

    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        if self._graph is not None:
            return self._graph.is_directed()

        return self._is_directed

    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        if self._graph is not None:
            return self._graph.number_of_nodes()

        return len(self._nodes)

    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        if self._graph is not None:
            return self._graph.number_of_edges()

        return len(self._edges)

    def nodes(self) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The graph nodes.
        """
        if self._graph is not None:
            return self._graph.nodes()

        return self._nodes.ids()

    def edges(self, triple=False) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Args:
            triple (bool): A flag that indicates whether to return edge triples
            of format (node 1, node 2, edge type) or edge pairs of format (node 1, node 2).

        Returns:
            The graph edges.
        """
        if self._graph is not None:
            return self._graph.edges(triple)

        return self._edges.all(triple).itertuples(index=False)

    def has_node(self, node: Any) -> bool:
        """
        Indicates whether or not the graph contains the specified node.

        Args:
            node (any): The node.

        Returns:
             bool: A value of True (cf False) if the node is
             (cf is not) in the graph.
        """
        if self._graph is not None:
            return self._graph.has_node(node)

        return node in self._nodes

    def neighbors(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring nodes.
        """
        if self._graph is not None:
            return self._graph.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        return self.in_nodes(node, include_edge_weight, edge_types) | self.out_nodes(
            node, include_edge_weight, edge_types
        )

    def _transform_edges(self, edges, include_edge_weight, edge_types, other_end_col):
        if edge_types is not None:
            correct_type = edges[globalvar.TYPE_ATTR_NAME].isin(set(edge_types))
            edges = edges[correct_type]

        if include_edge_weight:
            return {
                NeighbourWithWeight(node, weight)
                for node, weight in edges[[other_end_col, globalvar.WEIGHT]].itertuples(
                    index=False
                )
            }

        return set(edges[globalvar.SOURCE])

    def in_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring in-nodes.
        """
        if self._graph is not None:
            return self._graph.in_nodes(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        edges = self._edges.ins(node)
        return self._transform_edges(
            edges, include_edge_weight, edge_types, globalvar.SOURCE
        )

    def out_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring out-nodes.
        """
        if self._graph is not None:
            return self._graph.out_nodes(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        edges = self._edges.outs(node)
        return self._transform_edges(
            edges, include_edge_weight, edge_types, globalvar.TARGET
        )

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type (hashable, optional): a type of nodes that exist in the graph

        Returns:
            A list of node IDs with type node_type
        """
        if self._graph is not None:
            return self._graph.nodes_of_type(node_type)

        raise NotImplementedError("FIXME")

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        if self._graph is not None:
            return self._graph.node_type(node)

        return self._nodes.type(node)

    @property
    def node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
        if self._graph is not None:
            return self._graph.node_types

        return self._nodes.types()

    def node_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types (list, optional): A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        if self._graph is not None:
            return self._graph.node_feature_sizes(node_types)

        all_sizes = self._nodes.feature_sizes()
        if node_types is None:
            return all_sizes

        node_types = set(node_types)
        return {
            type_name: size for type_name, size in all_sizes if type_name in node_types
        }

    def check_graph_for_ml(self, features=True):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        if self._graph is not None:
            return self._graph.check_graph_for_ml(features)

        if all(size == 0 for _, size in self.node_feature_sizes().items()):
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # TODO: check the schema

        # TODO: check the feature node_ids against the graph node ids?

    def node_features(self, nodes, node_type=None):
        """
        Get the numeric feature vectors for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            nodes (list or hashable): Node ID or list of node IDs
            node_type (hashable): the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        if self._graph is not None:
            return self._graph.node_features(nodes, node_type)

        if node_type is None:
            types = self._nodes.type(nodes).unique()
            if len(types) == 0:
                raise ValueError(
                    "must have at least one node for inference, if `node_type` is not specified"
                )
            if len(types) > 1:
                raise ValueError("all nodes must have the same type")

            node_type = types[0]

        return self._nodes.features(node_type, nodes)

    ##################################################################
    # Computationally intensive methods:

    def info(self, show_attributes=True, sample=None):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes (bool, default True): If True, include attributes information
            sample (int): To speed up the graph analysis, use only a random sample of
                          this many nodes and edges.

        Returns:
            An information string.
        """
        if self._graph is not None:
            return self._graph.info(show_attributes, sample)

        raise NotImplementedError("FIXME")

    def create_graph_schema(self, create_type_maps=True, nodes=None):
        """
        Create graph schema in dict of dict format from current graph.

        Note the assumption we make that there is only one
        edge of a particular edge type per node pair.

        This means that specifying an edge by node0, node1 and edge type
        is unique.

        Arguments:
            create_type_maps (bool): If True quick lookup of node/edge types is
                created in the schema. This can be slow.

            nodes (list): A list of node IDs to use to build schema. This must
                represent all node types and all edge types in the graph.
                If specified, `create_type_maps` must be False.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema object.
        """
        if self._graph is not None:
            return self._graph.create_graph_schema(create_type_maps, nodes)

        raise NotImplementedError("FIXME")

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        if self._graph is not None:
            return self._graph.node_degrees()

        degrees = self._edges.degrees()
        if len(degrees) < self.number_of_nodes():
            # some isolated nodes, that don't appear in the edges directly
            zeros = pd.Series(0, index=self.nodes())
            degrees = degrees.add(zeros, fill_value=0)

        return degrees.items()

    def to_adjacency_matrix(self, nodes: Optional[Iterable] = None):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        Args:
            nodes (iterable): The optional collection of nodes
                comprising the subgraph. If specified, then the
                adjacency matrix is computed for the subgraph;
                otherwise, it is computed for the full graph.

        Returns:
             The weighted adjacency matrix.
        """
        if self._graph is not None:
            return self._graph.to_adjacency_matrix(nodes)

        raise NotImplementedError("FIXME")

    # FIXME: Experimental/special-case methods that need to be considered more
    def get_index_for_nodes(self, nodes, node_type=None):
        """
        Get the indices for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            n: (list or hashable) Node ID or list of node IDs
            node_type: (hashable) the type of the nodes.

        Returns:
            Numpy array containing the indices for the requested nodes.
        """
        if self._graph is not None:
            return self._graph.get_index_for_nodes(nodes, node_type)

        raise NotImplementedError("FIXME")

    def adjacency_types(self, graph_schema: GraphSchema):
        """
        Obtains the edges in the form of the typed mapping:

            {edge_type_triple: {source_node: [target_node, ...]}}

        Args:
            graph_schema: The graph schema.
        Returns:
             The edge types mapping.
        """
        if self._graph is not None:
            return self._graph.adjacency_types(graph_schema)

        raise NotImplementedError("FIXME")

    def edge_weights(self, source_node: Any, target_node: Any) -> List[Any]:
        """
        Obtains the weights of edges between the given pair of nodes.

        Args:
            source_node (any): The source node.
            target_node (any): The target node.

        Returns:
            list: The edge weights.
        """
        if self._graph is not None:
            return self._graph.edge_weights(source_node, target_node)

        raise NotImplementedError("FIXME")

    def node_attributes(self, node: Any) -> Set[Any]:
        """
        Obtains the names of any (non-standard) node attributes that are
        available in the user data.

        Args:
            node (any): The node of interest.

        Returns:
            set: The collection of node attributes.
        """
        if self._graph is not None:
            return self._graph.node_attributes(node)

        raise NotImplementedError("FIXME")


# A convenience class that merely specifies that edges have direction.
class StellarDiGraph(StellarGraph):
    def __init__(
        self,
        graph=None,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
    ):
        super().__init__(
            graph=graph,
            is_directed=True,
            edge_weight_label=edge_weight_label,
            node_type_name=node_type_name,
            edge_type_name=edge_type_name,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            feature_name=feature_name,
            target_name=target_name,
            node_features=node_features,
            dtype=dtype,
        )
