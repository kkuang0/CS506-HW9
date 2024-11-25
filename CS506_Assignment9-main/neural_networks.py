import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Activation functions and their derivatives
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_prime = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_prime = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_prime = lambda x: self.activation(x) * (1 - self.activation(x))
        else:
            raise ValueError("Activation function not supported")

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation_fn(self.z2) 
        return self.a2

    def backward(self, X, y):
        # Compute gradients using chain rule
        m = X.shape[0]
        
        # Output layer gradients
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_prime(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    # Plot decision hyperplane in the hidden space
    x_range = np.linspace(hidden_features[:, 0].min() - 0.5, hidden_features[:, 0].max() + 0.5, 50)
    y_range = np.linspace(hidden_features[:, 1].min() - 0.5, hidden_features[:, 1].max() + 0.5, 50)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros_like(xx)
    if mlp.W2.shape[0] == 3:  # Only plot if the hidden dimension is 3
        zz = (-mlp.W2[0, 0] * xx - mlp.W2[1, 0] * yy - mlp.b2[0, 0]) / mlp.W2[2, 0]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='y')

    # Plot input layer decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, alpha=0.8, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # Visualize gradients
    # Plot network graph
    pos = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y': (1.0, 0.0)
    }
    G = nx.DiGraph()

    input_nodes = ['x1', 'x2']
    hidden_nodes = ['h1', 'h2', 'h3']
    output_nodes = ['y']

    G.add_nodes_from(input_nodes + hidden_nodes + output_nodes)

    # Add edges with weights
    for i, in_node in enumerate(input_nodes):
        for j, h_node in enumerate(hidden_nodes):
            weight = mlp.W1[i, j]
            G.add_edge(in_node, h_node, weight=weight)

    for i, h_node in enumerate(hidden_nodes):
        weight = mlp.W2[i, 0]
        G.add_edge(h_node, 'y', weight=weight)

    node_size = 1000
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='blue',
                           node_size=node_size, ax=ax_gradient)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='blue',
                           node_size=node_size, ax=ax_gradient)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='blue',
                           node_size=node_size, ax=ax_gradient)

    for (u, v, d) in G.edges(data=True):
        weight = d['weight']
        color = 'red' if weight < 0 else 'purple'
        width = abs(weight) * 3
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                               width=width, ax=ax_gradient, alpha=0.6)

    nx.draw_networkx_labels(G, pos, ax=ax_gradient)
    ax_gradient.set_title(f'Network Graph at Step {frame * 10}')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
