import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

# Load dataset
G = nx.read_edgelist("dataset/facebook_combined.txt")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Community detection
partition = community_louvain.best_partition(G)

print("Communities found:", len(set(partition.values())))

# Graph visualization
pos = nx.spring_layout(G, k=0.15, iterations=20)

colors = [partition[node] for node in G.nodes()]

nx.draw_networkx_nodes(G, pos,
                       node_size=20,
                       node_color=colors,
                       cmap=plt.cm.rainbow)

nx.draw_networkx_edges(G, pos, alpha=0.3)

plt.title("Social Network Graph Communities")

# Save graph image
plt.savefig("images/graph_output.png", dpi=300)

# Show graph
plt.show()

# Influential user detection
degree = nx.degree_centrality(G)

top_users = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 Influential Users:")
for user, score in top_users:
    print(user, score)