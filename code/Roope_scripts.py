import time
import networkx as nx
from networkx import Graph
import igraph as ig
import matplotlib.pyplot as plt
import pickle
import csv
import sys
import os
from pathlib import Path
import ast

import numpy as np

POS_ATTR = "pos"
RGBA_ATTR = "rgba"
COLOR_ATTR = "color"
WEIGHT_ATTR = "weight"
DISTANCE_ATTR = "distance"
DEGREE_CEN_ATTR = "degree_cen"
BETWEENNESS_ATTR = "betweenness"
CLOSENESS_ATTR = "closeness"
CLUSTER_ATTR = "cluster"

def writePkl(G: Graph, path):
    """
    Save undirected graph into .pkl file with pickle
    """
    with open(path, "wb") as file:
        pickle.dump(G, file)


def readPkl(path) -> Graph:
    """
    load undirected graph from .pkl with pickle
    """
    G = nx.Graph()
    with open(path, mode="rb") as file:
        G = pickle.load(file)
    return G


def largestComponent(G: Graph):
    """
    Get the largest component
    """
    components = []
    if nx.is_directed(G):
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    return max(components, key=len)


def setEdgeAttr(G: Graph, attr: str, values: dict):
    """
    For each edge of G, set given attribute with given values
    """
    nx.set_edge_attributes(G, values, attr)


def setNodeAttr(G: Graph, attr: str, values: dict):
    """
    For each node of G, set given attribute with given values
    """
    nx.set_node_attributes(G, values, attr)


def createWeightedFriendShipNetwork(graph_csv, out_csv, out_pickle):
    """
    Create weighted friendship network and the amount of recipes shared between 
    friends is saved as edge attribute \"weight\" and \"distance\" = 1.0/weight 

    Also creates .csv [user, friends, weights]
    """
    if os.path.exists(out_csv):
        print(f"{out_csv} already exists, change path or delete the old file", file=sys.stderr)
        exit()
    if os.path.exists(out_pickle):
        print(f"{out_pickle} already exists, change path or delete the old file", file=sys.stderr)
        exit()

    # [set(items)]
    items = []

    print(f"Reading from {graph_csv}")
    with open(graph_csv, mode="r") as file:
        csv_reader = csv.reader(file)
        
        # Skip header row
        next(csv_reader)

        for row in csv_reader:
            # userId is in order from 0-25075
            # Index of items is same as userId
            items.append(set(ast.literal_eval(row[2])))

    # [[(userId, weight), (userId, weight), ...], ...]
    friends_weights = []
    print("Checking friendships")
    for userId in range(0, len(items)):
        friends_weights.append([])
        # Progress, this can take a while because this algorithm is O(n^2)
        if userId % 100 == 0 or userId == len(items) - 1:
            print(f"{userId}/{len(items)-1}")

        for friend in range(0, len(items)):
            if userId == friend:
                continue
            # elif intersection of recipes between user and friend is non-empty
            # then add friend to the user
            common_items = items[userId] & items[friend]
            if bool(common_items):
                # Weight is amount of common recipes
                friends_weights[userId].append((friend, len(common_items)))
    
    print(f"Writing to path {out_csv} and {out_pickle}")
    G = nx.Graph()
    with open(out_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        data = [
            # Header row
            ["userId", "friends", "weights"]
        ]
        for userId in range(0, len(friends_weights)):
            user_friends_weights = friends_weights[userId]
            friends = []
            weights = []
            if user_friends_weights:
                friends, weights = zip(*user_friends_weights)
                # zip makes them tuples
                friends = list(friends)
                weights = list(weights)

            data.append([userId, friends, weights])
            G.add_node(userId)
            for i, friend in enumerate(friends):
                G.add_edge(userId, friend, weight=weights[i], distance=1.0/weights[i])

        writer.writerows(data)

    writePkl(G, out_pickle)
    
    print("Done!")


def setCoordinates(G: Graph, rng_seed=55):
    """
    Calculate and set coordinates as node attributes \"pos\" using spring_layout/Fruchterman-Reingold force-directed algorithm
    """
    
    # spring_layout uses Fruchterman-Reingold force-directed algorithm
    print("Calculating 2D positions with spring_layout / Fruchterman-Reingold force-directed algorithm")
    edge = next(iter(G.edges))
    pos = {}
    if WEIGHT_ATTR in G.edges[edge]:
        print("Using weights in graph for calculation")
        pos = nx.spring_layout(
            G, 
            k=0.5, 
            iterations=50, 
            seed=rng_seed, 
            dim=2,
            weight=WEIGHT_ATTR
        )
    else:
        print("No weight attribute, assume each edge has weight of 1")
        pos = nx.spring_layout(
            G, 
            k=0.5, 
            iterations=50, 
            seed=rng_seed, 
            dim=2,
            weight=None
        )

    print(f"Setting coordinates to the graph as \"{POS_ATTR}\" attribute for each node")
    setNodeAttr(G, POS_ATTR, pos)
    

def saveGraph(G: Graph, path:str, title=""):
    """
    Draws colored graph if nodes have \"color\" attribute

    Saves the figure in png format to specified file-path
    """
    # Create a large figure
    plt.figure(figsize=(15, 15))

    pos = nx.get_node_attributes(G, "pos")
    first_node = next(iter(G.nodes))
    print("Drawing graph")
    if COLOR_ATTR in G.nodes[first_node]:
        colors = [G.nodes[node][COLOR_ATTR] for node in G.nodes]
        nx.draw(
            G,
            pos,
            node_size=1,  # Smaller nodes
            node_color=colors,  # Color nodes
            edge_color="gray",
            width=0.5,  # Thinner edges for better clarity
            alpha=0.7,  # Slight transparency for edges
            with_labels=False
        )
    else:
        nx.draw(
            G,
            pos,
            node_size=1,  # Smaller nodes
            node_color="blue",
            edge_color="gray",
            width=0.5,  # Thinner edges for better clarity
            alpha=0.7,  # Slight transparency for edges
            with_labels=False
        )
    plt.title(title)
    plt.savefig(path, format="png", dpi=300)
    plt.close()


def addColorFromCommunities(G: Graph, communities: list[Graph]):
    for comm in communities:
        if not RGBA_ATTR in comm.nodes[next(iter(comm.nodes()))]:
            print(f"No \"{RGBA_ATTR}\" attribute in community")
            return
        for node, data in comm.nodes(data=True):
            G.nodes[node][COLOR_ATTR] = data[RGBA_ATTR]


def createCommunities(G: Graph, path, rng_seed=55):
    """
    Create communities using louvain method and sets node attribute \"rgba\" as color.

    Gives distinct colors up to 20 communities. Reuses same colors if more than 20 communities.
    
    Saves each community into given directory as index.pkl and creates .csv [index, #nodes, #edges]
    """
    
    if WEIGHT_ATTR in G.edges[next(iter(G.edges))]:
        print("Creating communities with weights")
        communities = nx.algorithms.community.louvain_communities(G, seed=rng_seed, weight=WEIGHT_ATTR)
    else:
        print(f"No \"{WEIGHT_ATTR}\" found, community weights = 1")
        communities = nx.algorithms.community.louvain_communities(G, seed=rng_seed, weight=None)
    print("Communities to list")
    communities = list(communities)
    # Up to 20 different colors
    colors = plt.cm.get_cmap("tab20", len(communities))
    csv_path = path + "info.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["community", "size", RGBA_ATTR])

        for i, comm in enumerate(communities):
            print(f"Community {i+1}.")
            comm_path = path + f"{i+1}.pkl"
            Gsub = nx.subgraph(G, list(comm)).copy()
            # tuple of np.float64 --> tuple of float, nicer format in csv
            color_rgba = tuple(float(c) for c in colors(i))
            writer.writerow([i+1, len(list(Gsub.nodes())), color_rgba])
            for node in Gsub.nodes():
                Gsub.nodes[node][RGBA_ATTR] = color_rgba
            writePkl(Gsub, comm_path)


def nxToIgraph(G_nx: Graph):
    """
    Converts simple undirected networkx graph into igraph and gives dictionary so that 
    you know which node in networkx is node on igraph {nx_node: ig_idx}
    """
    print("Converting NetworkX graph to igraph")
    # Map node labels to integer indices
    node_list = list(G_nx.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # Create igraph Graph
    edges = [(node_to_index[u], node_to_index[v]) for u, v in G_nx.edges()]
    
    # If no weight then 1.0
    if not WEIGHT_ATTR in G_nx.edges[next(iter(G_nx.edges))]:
        print(f"\"{WEIGHT_ATTR}\" attribute was not found, using default weight 1.0")
    else:
        print(f"\"{WEIGHT_ATTR}\" attribute was found")

    if not DISTANCE_ATTR in G_nx.edges[next(iter(G_nx.edges))]:
        print(f"\"{DISTANCE_ATTR}\" attribute was not found, using default distance 1.0")
    else:
        print(f"\"{DISTANCE_ATTR}\" attribute was found")
    print(f"Graph size: {len(node_list)} nodes and {len(edges)} edges")
    weights = [G_nx[u][v].get(WEIGHT_ATTR, 1.0) for u, v in G_nx.edges()]
    distances = [G_nx[u][v].get(DISTANCE_ATTR, 1.0) for u, v in G_nx.edges()]

    G_ig = ig.Graph(edges=edges, directed=False)
    G_ig.vs["name"] = [str(n) for n in node_list]
    G_ig.es[WEIGHT_ATTR] = weights
    G_ig.es[DISTANCE_ATTR] = distances

    return (G_ig, node_to_index)


def createNxDict(values, node_to_index):
    """
    Create networkx style {nx_node: value}
    """
    node_value = {}
    for node, idx in node_to_index.items():
        node_value[node] = values[idx]
    return node_value


def iClusteringCoeff(G_ig: ig.Graph, node_to_index: dict) -> dict:
    """
    Compute unweighted clustering coefficients and return {nx_node: value}
    """
    clustering = G_ig.transitivity_local_undirected(weights=None, vertices=None, mode="zero")
    return createNxDict(clustering, node_to_index)


def iClosenessCentrality(G_ig: ig.Graph, node_to_index: dict) -> dict:
    """
    Compute normalized closenesss using centrality \"distance\" attribute and return {nx_node: value}
    """
    closeness = G_ig.closeness(vertices=None, weights=DISTANCE_ATTR, normalized=True)
    return createNxDict(closeness, node_to_index)


def iBetweennessCentrality(G_ig: ig.Graph, node_to_index: dict) -> dict:
    """
    Compute normalized betweenness using \"distance\" attribute and return {nx_node: value}
    """
    betweenness = G_ig.betweenness(vertices=None, weights=DISTANCE_ATTR, directed=False)
    n = G_ig.vcount()
    bet_norm = ((n-1)*(n-2))/2
    betweenness_centrality = {}
    for node, idx in node_to_index.items():
        betweenness_centrality[node] = betweenness[idx] / bet_norm
    return betweenness_centrality


def iAvgPathLength(G_ig: ig.Graph) -> float:
    """
    Compute average path-length by using \"distance\" attribute
    """
    return float(G_ig.average_path_length(weights=DISTANCE_ATTR, directed=False))


def iDiameter(G_ig: ig.Graph) -> float:
    """
    Compute diameter using \"distance\" attribute
    """
    return float(G_ig.diameter(weights=DISTANCE_ATTR, directed=False))


def iGlobalTransitivity(G_ig: ig.Graph) -> float:
    """
    Compute undirected global transitivity
    """
    return float(G_ig.transitivity_undirected(mode="zero"))


def calcSetLocalNodeAttributes(G_nx: Graph):
    """
    Calculates and sets degree centrality \"degree_cen\", clustering coefficient \"cluster\", 
    closeness centrality \"closeness\", and betweenness centrality \"betweenness\".
    
    parameters:
    - G_nx: networkx.Graph (undirected)
    """

    start = time.time()
    G_ig, node_to_index = nxToIgraph(G_nx)

    print(f"Setting degree centrality as node attribute \"{DEGREE_CEN_ATTR}\"")
    setNodeAttr(G_nx, DEGREE_CEN_ATTR, nx.degree_centrality(G_nx))
    
    # Unweighted clustering since there is no real standard
    # which algorithm is used for weighted
    # networkx and igraph use different algorithms for weighted
    print(f"Setting unweighted clustering coefficients as node attribute \"{CLUSTER_ATTR}\"")
    setNodeAttr(G_nx, CLUSTER_ATTR, iClusteringCoeff(G_ig, node_to_index))
    
    print(f"Setting normalized closeness centrality as node attribute \"{CLOSENESS_ATTR}\"")
    setNodeAttr(G_nx, CLOSENESS_ATTR, iClosenessCentrality(G_ig, node_to_index))

    print(f"Setting normalized betweenness centrality as node attribute \"{CLOSENESS_ATTR}\"")
    setNodeAttr(G_nx, BETWEENNESS_ATTR, iBetweennessCentrality(G_ig, node_to_index))
    
    print(f"\"{BETWEENNESS_ATTR}\", \"{CLOSENESS_ATTR}\", \"{CLUSTER_ATTR}\", \"{DEGREE_CEN_ATTR}\"  node attributes added. Time taken: {time.time() - start}")


def calcGlobal(G: Graph):
    """
    #nodes, #edges, average degree, average path-length, global-transitivity and diameter

    Assumes Graph is connected

    Global transitivity medium, diameter expensive, average path-length expensive

    return all values in single list
    """
    start = time.time()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    # Sum of degree = 2 num_edges
    avg_degree = (2 * num_edges) / num_nodes
    G_ig, _ = nxToIgraph(G)
    print("Calculating global-transitivity")
    global_transitivity = iGlobalTransitivity(G_ig)

    if DISTANCE_ATTR in G.edges[next(iter(G.edges))]:
        print("Using distance to calculate")
    else:
        print(f"No \"{DISTANCE_ATTR}\" attribute found, assuming distance = 1.0")
    print("Calculating diameter")
    diameter = iDiameter(G_ig)
    print("Calculating average path-length")
    avg_path_length = iAvgPathLength(G_ig)

    print(f"Time taken: {time.time() - start}")
    return [num_nodes, num_edges, avg_degree, avg_path_length, global_transitivity, diameter]


def numSharedItemsCommunity(G: Graph, items: list[set]):
    nodes = list(G.nodes())
    if len(nodes) < 1:
        return 0

    # Start with the items of the first node
    shared_items = items[nodes[0]].copy()

    # Intersect with items of all other nodes
    for node in nodes[1:]:
        shared_items.intersection_update(items[node])
        if not shared_items:
            break  # Early exit if no shared items remain

    return len(shared_items)


def calcGlobalCommunities(G: Graph, communities: list[Graph], items:list[set]) -> tuple[list, float]:
    """
    Calculate global metrics for each community and the modularity of the partition.

    For each community, compute:
      - #nodes
      - #edges
      - average degree
      - average path length
      - global transitivity
      - diameter
      - number of shared items

    Also return:
      - modularity of the partition of all communities (global measure)

    Args:
        G: The original graph
        communities: List of community subgraphs (from G)
        items: List where index is node and value is a set of items (e.g., recipes)

    Returns:
        (community_infos, modularity): A tuple where
            community_infos is a list of per-community metric lists
            modularity is a float score of the overall partition
    """

    all_communities = []
    community_infos = []

    for comm in communities:
        all_communities.append(set(comm.nodes()))
        global_info = calcGlobal(comm)
        global_info.append(numSharedItemsCommunity(comm, items))
        community_infos.append(global_info)

    if WEIGHT_ATTR in communities[0].edges[next(iter(communities[0].edges()))]:
        print("Using weights to calculate modularity")
        modularity = nx.algorithms.community.modularity(G, all_communities, weight=WEIGHT_ATTR)
    else:
        print(f"No \"{WEIGHT_ATTR}\" attribute found, assume weight = 1 for modularity")
        modularity =  nx.algorithms.community.modularity(G, all_communities, weight=None)
    

    return (community_infos, modularity)
    

def calcLocalAndGlobalCommunities(G: Graph, pp_users_csv, community_directory: Path) -> float:
    """
    Calculates and sets degree centrality \"degree_cen\", clustering coefficient \"cluster\", 
    closeness centrality \"closeness\", and betweenness centrality \"betweenness\". 
    Also overwrites the old .pkl files in community_directory

    Calculate #nodes, #edges, average degree, average path-length, global-transitivity, diameter, 
    and number of shared items for each community. 
    Write the results into community_file_name.csv

    return modularity
    """
    print(f"Reading recipes from {pp_users_csv}")
    items = []
    with open(pp_users_csv, mode="r") as file:
        csv_reader = csv.reader(file)
        # Skip header row
        next(csv_reader)
        for row in csv_reader:
            # userId is in order from 0-25075
            # Index of items is same as userId
            items.append(set(ast.literal_eval(row[2])))

    header = ["file", "#nodes", "#edges", "avg_degree", "avg_path_length", "global_transitivity", "diameter", "#shared_items"]
    communities = []
    for file in community_directory.glob("*.pkl"):
        comm = readPkl(str(file))
        calcSetLocalNodeAttributes(comm)
        communities.append(comm)
        writePkl(comm, str(file))

    community_data, modularity = calcGlobalCommunities(G, communities, items)
    for i, file in enumerate(community_directory.glob("*.pkl")):
        data = [file.stem] + community_data[i]
        writeCsv(file.with_suffix(".csv"), [header, data])
    return modularity


def setCategory(G: Graph):
    """
    K-core decomposition using nx.core_number which is the result of K-core decomposition.

    K-core decomposition: Makes subgraph where each node has less than k degree. 
    This can be done iteratively where we increase k in each iteration and set core-number 
    for each node in the subgraph and then delete those nodes from graph. Core-number in way
    tells how many iterations that node "survived" the deletion and remained in the graph.

    nx.core_number is calculated however differently. Order nodes of graph by their degree 
    --> delete the lowest --> decrease neighbors' degree --> reorder --> delete the lowest --> ... 
    and the degree the node had at deletion is its core-number.

    Uses nx.core_number to figure out core of each node then 
    put split core values into 0%-30%, 30%-60%, 60-90%, 90%-100% percentiles.

    set node attribute \"category\" as a number between 0 and 3

    0-30%: 0, peripheral

    30%-60%: 1, intermediate

    60-90%: 2, core

    90%-100%: 3, super-core
    """
    category_attr = "category"
    coreness = nx.core_number(G)
    core_values = list(coreness.values())
    thresholds = np.percentile(core_values, [30, 60, 90, 100])

    for node, core in coreness.items():
        category = len(thresholds) - 1
        for i, thresh in enumerate(thresholds):
            if core <= thresh:
                category = i
                break
        G.nodes[node][category_attr] = category


def writeCsv(path: str, data: list):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def printNodeEdgeAttributes(G: Graph):
    print(f"Node attributes: {G.nodes[next(iter(G.nodes()))]}")
    print(f"Edge attributes: {G.edges[next(iter(G.edges()))]}")


if __name__ == "__main__":
    # You can create your own file structure
    # All of scripts assume that the directory already exists
    # they don't create new directories, but they do create new files
    datasets = "../datasets/"
    datasets_new = datasets + "new/"
    datasets_pickle = datasets_new + "pickle/"
    datasets_pickle_communities = datasets_pickle + "PP_weighted_friends_con_communities/"
    pp_users = datasets + "PP_users.csv"
    pp_recipes = datasets + "pp_recipes.csv"
    raw_interactions = datasets + "RAW_interactions.csv"
    pickle_PP = datasets_pickle + "PP_weighted_friends.pkl"
    pickle_PP_con = datasets_pickle + "PP_weighted_friends_con.pkl"
    # 1)
    print("1) \n")
    createWeightedFriendShipNetwork(
        pp_users,
        datasets_new + "PP_weighted_friends.csv",
        pickle_PP
    )

    G = readPkl(pickle_PP)
    G = nx.subgraph(G, largestComponent(G)).copy()
    # G is now connected graph
    writePkl(G, pickle_PP_con)
    printNodeEdgeAttributes(G)

    # 2)
    print("2) \n")
    setCoordinates(G, 55)
    writePkl(G, pickle_PP_con)
    printNodeEdgeAttributes(G)

    # 3)
    print("3) \n")
    setCategory(G)
    writePkl(G, pickle_PP_con)
    printNodeEdgeAttributes(G)

    # 4)
    print("4) \n")
    createCommunities(G, datasets_pickle_communities, rng_seed=55)

    # 5.1 Communities)
    print("5.1) \n")
    modularity = calcLocalAndGlobalCommunities(G, pp_users, Path(datasets_pickle_communities))

    # 5.2 Graph)
    print("5.2) \n")
    calcSetLocalNodeAttributes(G)
    writePkl(G, pickle_PP_con)
    printNodeEdgeAttributes(G)
    header = ["#nodes", "#edges", "avg_degree", "avg_path_length", "global_transitivity", "diameter", "modularity"]
    graph_properties = calcGlobal(G)
    writeCsv(datasets_new + "PP_weighted_friends_con_global.csv", [header, graph_properties + [modularity]])


    # Draw communities in graph
    #G = readPkl(pickle_PP_con)
    #saveGraph(G, datasets_new + "PP_weighted_con.png", "Weighted graph")
    #community_dir = Path(datasets_pickle_communities)
    #communities = []
    #for file in community_dir.glob("*.pkl"):
    #    comm = readPkl(str(file))
    #    communities.append(comm)

    #addColorFromCommunities(G, communities)

    #saveGraph(G, datasets_new + "PP_weighted_con_colored.png", "Weighted graph with colored louvain communities")
    






