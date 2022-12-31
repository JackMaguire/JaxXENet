import numpy as onp
import jax
import jax.numpy as jnp
import jax.numpy as np
from jax import grad, jit, vmap

def mpnn(edges, edge_features, node_features):
    """
    Message passing neural network (MPNN) implemented in Jax.
    
    Parameters
    ----------
    edges : np.ndarray
        Sparse adjacency matrix of shape (num_edges, 2). Each row represents an edge,
        with the first and second column containing the indices of the source and
        destination nodes, respectively.
    edge_features : np.ndarray
        Edge features of shape (num_edges, num_edge_features).
    node_features : np.ndarray
        Node features of shape (num_nodes, num_node_features).
    num_propagate_steps : int
        Number of propagation steps.
    
    Returns
    -------
    np.ndarray
        Updated node features of shape (num_nodes, num_node_features).
    """

    key = jax.random.PRNGKey(0)
    num_nodes = node_features.shape[0]
    num_edge_features = edge_features.shape[-1]
    num_node_features = node_features.shape[-1]
    node_features_updated = node_features

    # Define MLP weights and biases for edges
    weights_edge = jax.random.normal( key=key, shape=(num_edge_features, num_node_features) )
    biases_edge = jax.random.normal( key=key, shape=(num_node_features,) )
    # Define MLP weights and biases for nodes
    weights_node = jax.random.normal( key=key, shape=(num_node_features * 2, num_node_features) )
    biases_node = jax.random.normal( key=key, shape=(num_node_features,) )
    
    # Propagation step
    num_propagate_steps = 2
    for _ in range(num_propagate_steps):
        # Initialize messages with zeros
        messages = np.zeros((num_nodes, num_node_features))
        # Iterate over edges
        for edge_idx in range(edges.shape[0]):
            src_node, dest_node = edges[edge_idx]
            # Check if there is a reverse edge
            reverse_edge_mask = (edges[:, 0] == dest_node) & (edges[:, 1] == src_node)
            #if np.any(reverse_edge_mask):
            # Compute message for destination node using MLP
            message = np.tanh(
                    np.dot(edge_features[edge_idx], weights_edge) + biases_edge
                )
            message = np.dot(message, node_features[src_node])
            # Accumulate message for destination node
            messages[dest_node] += message
        # Aggregate messages and node features
        node_features_updated = np.concatenate(
            [node_features_updated, messages], axis=-1
        )
        # Apply MLP to update node features
        node_features_updated = np.tanh(
            np.dot(node_features_updated, weights_node) + biases_node
        )
    
    return node_features_updated

# Compile the function with JIT for faster evaluation
mpnn_jit = jit(mpnn)#, static_argnames=['num_propagate_steps'])

def test():
    # Test the function with some
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    num_nodes = 4
    num_node_features = 3
    num_edge_features = 5
    #num_propagate_steps = 2

    key = jax.random.PRNGKey(0)
    edge_features = jax.random.uniform( key, shape=(edges.shape[0], num_edge_features) )
    node_features = jax.random.uniform( key, shape=(num_nodes, num_node_features) )
    num_propagate_steps = 2
    output = mpnn_jit(edges, edge_features, node_features)
    print( output )
test()
