import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
from community import community_louvain
from collections import defaultdict
from networkx.algorithms.community.quality import modularity
from cdlib import evaluation, algorithms
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import community.community_louvain as cl
import numpy as np



def create_graph(nodes_df, edges_df, directed=False):

    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes to the graph
    for _, row in nodes_df.iterrows():
        G.add_node(row['ID'], **row.to_dict())

    # Add edges to the graph
    for _, row in edges_df.iterrows():
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], **row.to_dict())

    return G


def customize_node_attributes(graph, node_size, class_label_column, color_map):

    # Set node size
    nx.set_node_attributes(graph, node_size, name='node_size')

    # Set node color based on class label
    class_labels = nx.get_node_attributes(graph, class_label_column)
    node_colors = {node: color_map[label] for node, label in class_labels.items() if label in color_map}
    nx.set_node_attributes(graph, node_colors, name='node_color')

    return graph


def customize_edge_attributes(graph, edge_color):

    nx.set_edge_attributes(graph, edge_color, name='edge_color')

    return graph


def compute_graph_metrics(graph):
    metrics = {}
    metrics['Nodes'] = len(graph.nodes)
    metrics['Edges'] = len(graph.edges)

    # Check if the graph is directed or undirected
    if nx.is_directed(graph):
        try:
            # Attempt to compute the average shortest path length
            metrics['Average Clustering Coefficient'] = nx.average_clustering(graph)
            metrics['Average Path Length'] = nx.average_shortest_path_length(graph)
        except nx.NetworkXError as e:
            # If the graph is not strongly connected, handle the error gracefully
            if "Graph is not strongly connected" in str(e):
                metrics['Average Path Length'] = "Graph is not strongly connected"
            else:
                raise e  # Re-raise other NetworkX errors
        # Add other metrics for directed graphs as needed
    else:
        # Compute metrics specific to undirected graphs
        metrics['Average Clustering Coefficient'] = nx.average_clustering(graph)
        metrics['Average Path Length'] = nx.average_shortest_path_length(graph)
        # Compute degree distribution
        degree_hist = nx.degree_histogram(graph)
        degrees = range(len(degree_hist))
        metrics['Degree Distribution'] = dict(zip(degrees, degree_hist))
        metrics['Mean Degree'] = np.mean(list(dict(graph.degree()).values()))
        metrics['Median Degree'] = np.median(list(dict(graph.degree()).values()))

    return metrics


def filter_by_centrality(graph, centrality_measure, threshold):

    if centrality_measure == 'Degree Centrality':
        centrality_scores = nx.degree_centrality(graph)
    elif centrality_measure == 'Betweenness Centrality':
        centrality_scores = nx.betweenness_centrality(graph)
    elif centrality_measure == 'Closeness Centrality':
        centrality_scores = nx.closeness_centrality(graph)
    else:
        raise ValueError(
            "Invalid centrality measure. Supported measures: 'Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality'")

    # Filter nodes based on the centrality threshold
    filtered_nodes = [node for node, score in centrality_scores.items() if score >= float(threshold)]

    # Create a subgraph containing only the filtered nodes and their edges
    filtered_graph = graph.subgraph(filtered_nodes)

    return filtered_graph


def filter_by_community_membership(graph):

    # Implement a community detection algorithm (e.g., Louvain method)
    # Call the community detection function here
    communities = Louvain_Algorithm(graph)

    # Choose a specific community or range of communities to filter
    # Implement a mechanism to select communities here
    #selected_community = select_community(communities)

    # Filter nodes based on community membership
    #filtered_nodes = [node for node, data in graph.nodes(data=True) if data['community'] == selected_community]

    # Create a subgraph containing only the filtered nodes and their edges
    #filtered_graph = graph.subgraph(filtered_nodes)

    #return filtered_graph


def Louvain_Algorithm(graph):
    louvain_communities = community_louvain.best_partition(graph)

    louvain_communities_sets = {}
    for node, community in louvain_communities.items():
        if community not in louvain_communities_sets:
            louvain_communities_sets[community] = set()
        louvain_communities_sets[community].add(node)
    louvain_modularity = modularity(graph, list(louvain_communities_sets.values()))
    return graph, len(louvain_communities_sets), louvain_modularity, louvain_communities


def Girvan_Newman_Algorithm(graph):
    girvan_newman_communities_generator = nx.algorithms.community.girvan_newman(graph)
    girvan_newman_communities = next(girvan_newman_communities_generator)
    girvan_newman_modularity = nx.algorithms.community.quality.modularity(graph, girvan_newman_communities)
    return graph, len(girvan_newman_communities), girvan_newman_modularity, girvan_newman_communities


def Fast_Greedy_Algorithm(graph):
    fast_greedy_communities = nx.algorithms.community.greedy_modularity_communities(graph)
    fast_greedy_modularity = nx.algorithms.community.quality.modularity(graph, fast_greedy_communities)
    return graph, len(fast_greedy_communities), fast_greedy_modularity, fast_greedy_communities


def partition_graph_based_on_attribute(graph, attribute_name):

    clusters = defaultdict(set)

    # Iterate over nodes and assign them to clusters based on the attribute value
    for node, data in graph.nodes(data=True):
        attribute_value = data.get(attribute_name)
        clusters[attribute_value].add(node)

    return clusters
