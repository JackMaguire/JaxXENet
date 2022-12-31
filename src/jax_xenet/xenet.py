import numpy as onp
import jax
import jax.numpy as jnp
import jax.numpy as np
from jax import grad, jit, vmap, nn

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

    debug_mode = True

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
    
    reverse_edge_error_flag = np.ones((1,))

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
            rev_edge_idx = np.argmax(reverse_edge_mask)

            if debug_mode:
                _dest_node, _src_node = edges[rev_edge_idx]
                reverse_edge_error_flag *= np.equal( dest_node, _dest_node )
                reverse_edge_error_flag *= np.equal( src_node, _src_node )

            x_i = node_features[ src_node ]
            x_j = node_features[ dest_node ]
            e_ij = edge_features[ edge_idx ]
            e_ji = edge_features[ rev_edge_idx ]

            message_input = np.concatenate(
                [ x_i, x_j, e_ij, e_ji ],
                axis=-1
            )
            #if np.any(reverse_edge_mask):
            # Compute message for destination node using MLP
            message = nn.relu(
                    np.dot(edge_features[edge_idx], weights_edge) + biases_edge
                )
            message = np.dot(message, node_features[src_node])
            # Accumulate message for destination node
            #
            #messages[dest_node] += message
            # TypeError: '<class 'jax.interpreters.partial_eval.DynamicJaxprTracer'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
            messages.at[dest_node].set( messages[dest_node] + message )

        # Aggregate messages and node features
        node_features_updated = np.concatenate(
            [node_features_updated, messages], axis=-1
        )
        # Apply MLP to update node features
        node_features_updated = nn.relu(
            np.dot(node_features_updated, weights_node) + biases_node
        )
    
    # flip all flags such that 0 means "pass"
    reverse_edge_error_flag = 1 - reverse_edge_error_flag

    if debug_mode:
        return node_features_updated, reverse_edge_error_flag
    else:
        return node_features_updated, None

# Compile the function with JIT for faster evaluation
mpnn_jit = jit(mpnn)#, static_argnames=['num_propagate_steps'])

def test():
    # Test the function with some
    #edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 0], [2, 1], [3, 2], [0, 3] ])
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
