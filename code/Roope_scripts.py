
from collections import Counter
import time
from matplotlib import figure
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
import powerlaw

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

    pos = nx.get_node_attributes(G, POS_ATTR)
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


def powerlawPlot(values, file_path, data_label="values"):
    """
    Plot values with line corresponding to powerlaw line, if values follow that line then
    your data follows powerlaw
    """
    # Plot the data and the fitted power law
    fit = powerlaw.Fit(values, xmin=min(values))
    plt.figure()
    fit.plot_pdf(color="b", label=data_label)
    fit.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
    plt.legend()
    plt.savefig(file_path)
    plt.close()


def plotHistogram(values, file_path, title="Histogram", x_label="Values", y_label="Frequency", bins=10):
    """
    Plot histogram y: Frequency, x: value, if bins < 1 then give each value its 
    own bin, default is 10 bins. If lots of unique values then giving each value its own bin 
    might look ugly
    """
    plt.figure(figsize=(8,6))
    if bins > 0:
        plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        counter = Counter(values)
        x = list(counter.keys())
        y = list(counter.values())
        plt.bar(x, y, color='skyblue', edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(file_path)
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


def createItemsFromCsv(pp_users_csv) -> list[set]:
    items = []
    with open(pp_users_csv, mode="r") as file:
        csv_reader = csv.reader(file)
        # Skip header row
        next(csv_reader)
        for row in csv_reader:
            # userId is in order from 0-25075
            # Index of items is same as userId
            items.append(set(ast.literal_eval(row[2])))
    return items


def createRecipesFromCsv(pp_recipes_csv) -> dict:
    """
    Read recipes from given csv, parse into dict i: {id:,name_tokens:, ...}
    """
    recipes = {}
    with open(pp_recipes_csv, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row["i"]
            value = {k: v for k, v in row.items() if k != "i"}
            recipes[key] = value
    return recipes


def analyze_shared_recipes(G: Graph, items: list[set]):
    """
    Analyze recipes shared within Graph

    Returns up to top 10 recipes
    """
    print("Analyzing shared recipes")
    # recipe_id: set(user, user2, ...)
    recipe_to_users = {}
    # set of different recipes found within graph
    recipes = set()
    for node in G.nodes():
        for recipe in items[node]:
            if recipe not in recipe_to_users:
                recipe_to_users[recipe] = set()
            recipe_to_users[recipe].add(node)
            recipes.add(recipe)

    # amount of different recipes shared by atleast 2 people
    shared_recipe_count = 0 
    # recipe id: shared_amount
    top_recipes = {}

    for recipe in recipes:
        num_users_with_recipe = len(recipe_to_users.get(recipe, set()))
        # Only count recipes shared by at least 2 users in the graph
        if num_users_with_recipe >= 2:
            top_recipes[recipe] = num_users_with_recipe
            shared_recipe_count += 1

    # Get top 10 most shared recipes
    top10_recipes = sorted(top_recipes.items(), key=lambda x: x[1], reverse=True)[:10]

    return (shared_recipe_count, top10_recipes)


def analyze_recipe(recipe_id, recipe_infos: dict):
    """
    get calorie level, amounts of steps, and amount of ingredients

    return list or None if recipe_id didn't have a match
    """
    info = recipe_infos.get(str(recipe_id), None)
    if info == None:
        return None
    
    return [
        info["calorie_level"],
        len(ast.literal_eval(info["steps_tokens"])),
        len(ast.literal_eval(info["ingredient_ids"]))
    ]


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
    items = createItemsFromCsv(pp_users_csv)

    header = ["file", "#nodes", "#edges", "avg_degree", "avg_path_length", "global_transitivity", "diameter"]
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


def readCsv(path: str):
    """
    Skip header, return rows
    """
    data = []
    with open(path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)
    return data


def readCommunities(community_dir: Path):
    """
    Return communities sorted by their id(community) inside info.csv
    """
    communities = []
    rows = readCsv(str(community_dir)+ "/info.csv")
    color_to_id = {ast.literal_eval(row[2]): int(row[0]) for row in rows}

    for file in community_dir.glob("*.pkl"):
        communities.append(readPkl(str(file)))
    communities.sort(key=lambda comm: color_to_id[comm.nodes[next(iter(comm.nodes()))][RGBA_ATTR]])
    return communities
    

def printNodeEdgeAttributes(G: Graph):
    print(f"Node attributes: {G.nodes[next(iter(G.nodes()))]}")
    print(f"Edge attributes: {G.edges[next(iter(G.edges()))]}")


############################################################
# Uswah's scripts but made compatible with my scripts

def visualize_network_bkp(G: Graph, output_file: str):
    """
    Create and save a visualization of the friendship network

    Parameters:
    -----------
    G : networkx.Graph
        Graph representation of the friendship network
    output_file : str, optional
        Filename to save the visualization
    """
    # Create figure with a specific axis for the graph and space for the colorbar
    fig, ax = plt.subplots(figsize=(12, 12))

    # get positions
    pos = nx.get_node_attributes(G, POS_ATTR)

    # Color nodes based on degree (number of connections)
    node_degrees = dict(G.degree())
    node_colors = [node_degrees[node] for node in G.nodes()]

    # Get the min and max for the colormap normalization
    vmin = min(node_colors) if node_colors else 0
    vmax = max(node_colors) if node_colors else 1

    # Draw the network on our specific axis
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
        node_size=50
    )

    # Draw edges separately
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edge_color='gray',
        alpha=0.7,
        width=0.5
    )

    # Now we can create a colorbar from the nodes mappable
    plt.colorbar(nodes, ax=ax, label="Number of Friends")

    ax.set_title("Weighted User Friendship Network (based on shared recipes)")
    ax.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


def create_community_statistics_table(community: Graph):
    """Create a table showing key statistics for each community"""
    print("Creating community statistics table...")
    # Prepare data for the table
    table_data = []

    for community_id, stats in community_stats.items():
        # Skip communities with missing key data
        if any(stats.get(k) is None for k in ['members', 'edges', 'density', 'clustering_coefficient']):
            continue

        row = {
            'Community': community_id,
            'Vertices': stats['members'],
            'Edges': stats['edges'],
            'Density': stats['density'],
            'Clustering Coefficient': stats['clustering_coefficient'],
            'Global Transitivity': stats['global_transitivity'],
            'Avg Degree Centrality': stats['avg_degree_centrality'],
            'Avg Path Length': stats.get('avg_path_length', 'N/A')
        }
        table_data.append(row)

    # Create DataFrame and save as CSV
    stats_df = pd.DataFrame(table_data)
    stats_df.to_csv('community_statistics_table.csv', index=False)

    # Also create a visual table using matplotlib for the report
    plt.figure(figsize=(12, len(table_data) * 0.5 + 2))

    # Display table
    cell_text = []
    for _, row in stats_df.iterrows():
        cell_text.append([
            row['Community'],
            f"{row['Vertices']:,}",
            f"{row['Edges']:,}",
            f"{row['Density']:.4f}",
            f"{row['Clustering Coefficient']:.4f}",
            f"{row['Global Transitivity']:.4f}",
            f"{row['Avg Degree Centrality']:.4f}",
            str(row['Avg Path Length']) if row['Avg Path Length'] != 'N/A' else 'N/A'
        ])

    # Create the table
    the_table = plt.table(
        cellText=cell_text,
        colLabels=stats_df.columns,
        loc='center',
        cellLoc='center'
    )

    # Adjust table appearance
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)

    # Hide axes
    plt.axis('off')
    plt.title('Community Statistics', fontsize=16)
    plt.tight_layout()
    plt.savefig('community_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    return stats_df


def visualize_communities(G: Graph, communities: list[Graph], file_path):
    """Visualize communities with different colors using the force-directed layout"""
    print("Visualizing communities...")
    # Create a figure with a specific axis
    plt.figure(figsize=(16, 16))

    pos = nx.get_node_attributes(G, POS_ATTR)

    # Count number of communities
    num_communities = len(communities)

    # Create mappings comm: id
    comm_to_ids = {}
    # Draw nodes
    for i, comm in enumerate(communities):
        comm_to_ids[comm] = i+1
        # Get list of nodes in this community
        community_nodes = list(comm.nodes())
        # Draw these nodes with a specific color
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=community_nodes,
            node_size=50,
            node_color=[comm.nodes[node][RGBA_ATTR] for node in comm.nodes()],
            label=f"Community {comm_to_ids[comm]}"
        )

    ids_to_comm = {v: k for k, v in comm_to_ids.items()}

    # Draw edges with transparency
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

    # Create custom legend with community sizes
    community_sizes = {comm_to_ids[comm]: len(list(comm.nodes())) for comm in communities}

    # Sort communities by size for the legend
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    legend_labels = [f"Community {comm_id}: {size} users" for comm_id, size in sorted_communities]

    # Use proxy artists for the legend
    proxy_artists = [
        plt.Line2D(
            [0], [0], marker='o', 
            color=ids_to_comm[com_id].nodes[next(iter(ids_to_comm[com_id].nodes()))][RGBA_ATTR],
            markersize=10, linestyle=''
        )
        for com_id, _ in sorted_communities
    ]

    # Add legend (place it outside the main plot area if too many communities)
    if num_communities > 10:
        plt.legend(proxy_artists, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    else:
        plt.legend(proxy_artists, legend_labels, loc='lower center', ncol=3)
    
    print("Saving fig")
    plt.title("Friendship Network Communities", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a more simplified version with just colors if there are many communities
    #if num_communities > 10:
    #    plt.figure(figsize=(16, 16))
    #    # Draw nodes colored by community
    #    node_colors = [cmap(partition[node] / num_communities) for node in G.nodes()]
    #    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)
    #    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    #    plt.title("Friendship Network Communities (Simplified View)", fontsize=20)
    #    plt.axis('off')
    #    plt.tight_layout()
    #    plt.savefig('community_visualization_simple.png', dpi=300)
    #    plt.close()

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
    figures = "../figures/"

    # 1)
    #print("1) \n")
    #createWeightedFriendShipNetwork(
    #    pp_users,
    #    datasets_new + "PP_weighted_friends.csv",
    #    pickle_PP
    #)

    #G = readPkl(pickle_PP)
    #G = nx.subgraph(G, largestComponent(G)).copy()
    # G is now connected graph
    #writePkl(G, pickle_PP_con)
    #printNodeEdgeAttributes(G)

    # 2)
    #print("2) \n")
    #setCoordinates(G, 55)
    #writePkl(G, pickle_PP_con)
    #printNodeEdgeAttributes(G)

    # 3)
    #print("3) \n")
    #setCategory(G)
    #writePkl(G, pickle_PP_con)
    #printNodeEdgeAttributes(G)

    # 4)
    #print("4) \n")
    #createCommunities(G, datasets_pickle_communities, rng_seed=55)

    # 5.1 Communities)
    #print("5.1) \n")
    #modularity = calcLocalAndGlobalCommunities(G, pp_users, Path(datasets_pickle_communities))

    # 5.2 Graph)
    #print("5.2) \n")
    #calcSetLocalNodeAttributes(G)
    #writePkl(G, pickle_PP_con)
    #printNodeEdgeAttributes(G)
    #header = ["#nodes", "#edges", "avg_degree", "avg_path_length", "global_transitivity", "diameter", "modularity"]
    #graph_properties = calcGlobal(G)
    #writeCsv(datasets_new + "PP_weighted_friends_con_global.csv", [header, graph_properties + [modularity]])

    # 6.1 shared recipes)
    #print("6.1 Shared recipes) \n")
    #items = createItemsFromCsv(pp_users)
    #community_dir = Path(datasets_pickle_communities)
    #header = ["file", "amount_shared_recipes", "top_recipe_ids", "top_recipe_shared_amounts"]
    #data = []
    #for file in community_dir.glob("*.pkl"):
    #    comm = readPkl(str(file))
    #    amount_shared_recipes, top10_recipes = analyze_shared_recipes(comm, items)
    #    recipe_ids = []
    #    recipe_shared_amounts = []
    #    if top10_recipes:
    #        recipe_ids, recipe_shared_amounts = zip(*top10_recipes)
    #        # zip makes them tuples, turn to list
    #        recipe_ids = list(recipe_ids)
    #        recipe_shared_amounts = list(recipe_shared_amounts)
    #    data.append([file.stem, amount_shared_recipes, recipe_ids, recipe_shared_amounts])
    #writeCsv(datasets_pickle_communities + "community_shared_recipes.csv", [header] + data)

    # 6.2 top10 recipes analysis)
    #print("6.2 Shared top10 recipes analysis) \n")
    #rows = readCsv(datasets_pickle_communities + "community_shared_recipes.csv")
    #recipe_infos = createRecipesFromCsv(pp_recipes)
    #header = ["file", "recipe_id", "share_count", "calorie_level", "steps_count", "ingredients_count"]
    #data = []
    #for row in rows:
    #    recipes = ast.literal_eval(row[2])
    #    share_counts = ast.literal_eval(row[3])
    #    for i, recipe in enumerate(recipes):
    #        calorie_level, step_count, ing_count = analyze_recipe(recipe, recipe_infos)
    #        data_row = [row[0], recipe, share_counts[i], calorie_level, step_count, ing_count]
    #        data.append(data_row)
    #writeCsv(datasets_pickle_communities + "community_top_recipes.csv", [header] + data)
    
    # 7 Visualize connected network)
    #visualize_network_bkp(G, figures + "ISNA25_RUE_pp_wcon.png")

    # 8 Distribution graph)
    #plotHistogram(
    #    list(nx.get_node_attributes(G, DEGREE_CEN_ATTR).values()),
    #    figures + "histogram_degree.png",
    #    "Degree centrality histogram",
    #    "Degree centrality",
    #    bins=50
    #)
    #plotHistogram(
    #    list(nx.get_node_attributes(G, BETWEENNESS_ATTR).values()),
    #    figures + "histogram_betweenness.png",
    #    "Normalized betweenness centrality histogram",
    #    "Normalized betweenness centrality",
    #    bins=50
    #)
    #plotHistogram(
    #    list(nx.get_node_attributes(G, CLOSENESS_ATTR).values()),
    #    figures + "histogram_closeness.png",
    #    "Normalized closeness centrality histogram",
    #    "Normalized closeness centrality",
    #    bins=50
    #)
    #plotHistogram(
    #    list(nx.get_node_attributes(G, CLUSTER_ATTR).values()),
    #    figures + "histogram_cluster.png",
    #    "Clustering coefficient histogram",
    #    "Clustering coefficients",
    #    bins = 10
    #)

    # 9 Powerlaw graph)
    #powerlawPlot(list(nx.get_node_attributes(G, DEGREE_CEN_ATTR).values()), figures + "powerlaw_degree.png", "Degree centrality")
    #powerlawPlot(list(nx.get_node_attributes(G, BETWEENNESS_ATTR).values()), figures + "powerlaw_betweenness.png", "Betweenness centrality")
    #powerlawPlot(list(nx.get_node_attributes(G, CLOSENESS_ATTR).values()), figures + "powerlaw_closeness.png", "Closeness centrality")

    # 10 Visualize communities on the largest component)
    G = readPkl(pickle_PP_con)
    visualize_communities(
        G,
        readCommunities(Path(datasets_pickle_communities)),
        figures + "community_graph.png"
    )


    # 8 Plot 
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
    






