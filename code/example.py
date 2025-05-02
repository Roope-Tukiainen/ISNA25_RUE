import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt
import pickle
import csv
import sys
import os
import ast

def writePkl(G: Graph, path):
    """
    Save undirected graph into .pkl file with pickle, wont overwrite existing files
    """
    if os.path.exists(path):
        print(f"{path} already exists, change path or delete the old file", file=sys.stderr)
        exit()
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


def createWeightedFriendShipNetwork(graph_csv, out_csv, out_pickle):
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
                G.add_edge(userId, friend, weight=weights[i])
        writer.writerows(data)

    writePkl(G, out_pickle)
    
    print("Done!")



def setCoordinates(G: Graph, rng_seed=55):
    # spring_layout uses Fruchterman-Reingold force-directed algorithm
    print("Calculating 2D positions with spring_layout / Fruchterman-Reingold force-directed algorithm")
    edge = next(iter(G.edges))
    pos = {}
    if "weight" in G.edges[edge]:
        print("Using weights in graph for calculation")
        pos = nx.spring_layout(
            G, 
            k=0.5, 
            iterations=50, 
            seed=rng_seed, 
            dim=2,
            weight="weight"
        )
    else:
        print("No weight attribute, assume each edge has weight of 1")
        pos = nx.spring_layout(
            G, 
            k=0.5, 
            iterations=50, 
            seed=rng_seed, 
            dim=2
        )

    print("Setting coordinates to the graph as \"pos\" attribute for each node")
    i = 0
    for node, coord in pos.items():
        if i % 100 == 0 or i == len(pos.items()) - 1:
            print(f"{i}/{len(pos.items())-1}")
        G.nodes[node]["pos"] = coord
        i += 1
    

def drawGraph(G: Graph, title=""):
    # Create a large figure
    plt.figure(figsize=(15, 15))

    pos = nx.get_node_attributes(G, "pos")
    first_node = next(iter(G.nodes))
    print("Drawing graph")
    if "color" in G.nodes[first_node]:
        colors = [G.nodes[node]["color"] for node in G.nodes]
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
    plt.show()

def createCommunities(G: Graph, path, rng_seed=55):
    communities = nx.algorithms.community.louvain_communities(G, seed=rng_seed)
    print("Communities to list")
    communities = list(communities)
    colors = plt.cm.get_cmap("tab20", len(communities))
    csv_path = path + "info.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["community", "size", "rgba"])

        for i, comm in enumerate(communities):
            print(f"Community {i+1}.")
            comm_path = path + f"{i+1}.pkl"
            Gsub = nx.subgraph(G, list(comm)).copy()
            # tuple of np.float64 --> tuple of float, nicer format in csv
            color_rgba = tuple(float(c) for c in colors(i))
            writer.writerow([i+1, len(list(Gsub.nodes())), color_rgba])
            for node in Gsub.nodes():
                Gsub.nodes[node]["rgba"] = color_rgba
            writePkl(Gsub, comm_path)

if __name__ == "__main__":
    # You can create your own file structure
    # All of scripts assume that the directory already exists
    # they don't create new ones, but they do create new files
    datasets = "../datasets/"
    datasets_new = datasets + "new/"
    datasets_pickle = datasets_new + "pickle/"
    pp_users = datasets + "PP_users.csv"
    pp_recipes = datasets + "pp_recipes.csv"
    raw_interactions = datasets + "RAW_interactions.csv"
    
    #createWeightedFriendShipNetwork(
    #    pp_users,
    #    datasets_new + "PP_weighted_friends.csv",
    #    datasets_pickle + "PP_weighted_friends.pkl"
    #)
    
    #G = readPkl(datasets_pickle + "PP_weighted_friends.pkl")
    #setCoordinates(G, 55)
    #writePkl(G, datasets_pickle + "PP_weighted_pos_friends.pkl")

    #G = readPkl(datasets_pickle + "PP_weighted_pos_friends.pkl")
    #drawGraph(G, "Weighted graph")

    #G = readPkl(datasets_pickle + "PP_weighted_friends.pkl")
    #createCommunities(G, datasets_pickle + "PP_weighted_friends_communities/", rng_seed=55)






def writePklFromCsv(csvp, pkl):
    G = nx.Graph()
    with open(csvp, "r") as file:
        reader = csv.reader(file)
        next(reader)
        rows = list(reader)
        for row in rows:
            id = int(row[0])
            if id % 100 == 0 or id == len(rows) - 1:
                print(f"{id}/{len(rows)-1}")
            friends = ast.literal_eval(row[1])
            weights = ast.literal_eval(row[2])
            G.add_node(id)
            for i, friend in enumerate(friends):
                G.add_edge(i, friend, weight=weights[i])
    writePkl(G, pkl)