import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_graph(filename):
    with open(filename, 'r') as f:
        adjacency_matrix = np.array([list(map(int, line.split())) for line in f])
    return adjacency_matrix

def get_laplacian(adjacency_matrix):
    return np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix

def get_coordinates(laplacian):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

    u = eigenvectors[:, 1] # second smallest eigenvector
    v = eigenvectors[:, 2] # third smallest eigenvector

    return [(u[i], v[i]) for i in range(len(u))]

if __name__ == '__main__':
    adjacency_matrix = read_graph('graphs/example.txt')
    laplacian = get_laplacian(adjacency_matrix)
    coordinates = get_coordinates(laplacian)

    # Create a graph from the adjacency matrix
    G = nx.Graph()
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Draw the graph using custom coordinates
    plt.figure(figsize=(6, 6))
    nx.draw(
        G, 
        pos=coordinates, 
        with_labels=True, 
        node_color='lightblue', 
        edge_color='gray', 
        node_size=800, 
        font_size=10
    )

    plt.title("Graph Visualization with Custom Coordinates")
    plt.show()