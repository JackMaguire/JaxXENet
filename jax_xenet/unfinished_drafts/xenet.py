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
    
    Returns
    -------
    np.ndarray
        Updated node features of shape (num_nodes, num_node_features).
    """

    debug_mode = True

    key = jax.random.PRNGKey(0)
    num_nodes = node_features.shape[0]
    num_node_features = node_features.shape[-1]
    num_edges = edge_features.shape[0]
    num_edge_features = edge_features.shape[-1]

    Fin = num_node_features
    Sin = num_edge_features

    # TODO - decouple into its own setting
    Fout = 3
    Sout = 4

    # Define MLP weights and biases for edges
    stack_sizes = [ 64, 64 ]
    def stack_input_size( i ):
        if i == 0: return 2*Fin + 2*Sin
        else:      return stack_sizes[ i-1 ]
    n_stack_convs = len(stack_sizes)
    
    # Define MLP weights and biases for stacks
    weights_stack = [ jax.random.normal( key=key, shape=( stack_input_size(i), stack_sizes[i]) ) for i in range(len(stack_sizes)) ] 
    biases_stack = [ jax.random.normal( key=key, shape=( stack_sizes[i], ) ) for i in range(len(stack_sizes)) ] 

    # Define MLP weights and biases for nodes
    weights_node = jax.random.normal( key=key, shape=(num_node_features + (stack_sizes[-1]*2), Fout) )
    biases_node = jax.random.normal( key=key, shape=(Fout,) )

    # Define MLP weights and biases for edges
    weights_edge = jax.random.normal( key=key, shape=(stack_sizes[-1], Sout) )
    biases_edge = jax.random.normal( key=key, shape=(Sout,) )
    
    reverse_edge_error_flag = np.ones((1,))

    # Initialize messages with zeros
    incoming_stacks = np.zeros((num_nodes, stack_sizes[-1]))
    outgoing_stacks = np.zeros((num_nodes, stack_sizes[-1]))
    all_stacks = np.zeros((num_edges, stack_sizes[-1]))
    

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

        stack = np.concatenate(
            [ x_i, x_j, e_ij, e_ji ],
            axis=-1
        )

        # Compute message for destination node using MLP
        for i in range( n_stack_convs ):
            stack = np.dot(stack, weights_stack[i]) + biases_stack[i]
            stack = nn.relu( stack )

        # TODO attention

        # Accumulate messages for nodes
        incoming_stacks = incoming_stacks.at[dest_node].add( stack )
        outgoing_stacks = outgoing_stacks.at[src_node].add( stack )
        all_stacks = all_stacks.at[edge_idx].set( stack )

    #########
    # NODES #
    #########

    # Aggregate messages and node features
    node_features_updated = np.concatenate(
        [node_features, incoming_stacks, outgoing_stacks],
        axis=-1
    )
    # Apply MLP to update node features
    node_features_updated = nn.relu(
        np.dot(node_features_updated, weights_node) + biases_node
    )

    #########
    # EDGES #
    #########

    # Apply MLP to update edge features
    edge_features_updated = nn.relu(
        np.dot(all_stacks, weights_edge) + biases_edge
    )

    ##########
    # ERRORS #
    ##########
    
    # flip all flags such that 0 means "pass"
    reverse_edge_error_flag = 1 - reverse_edge_error_flag

    if debug_mode:
        return node_features_updated, edge_features_updated, reverse_edge_error_flag
    else:
        return node_features_updated, edge_features_updated, None

# Compile the function with JIT for faster evaluation
mpnn_jit = jit(mpnn)
#mpnn_jit = mpnn

def test():
    #edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 0], [2, 1], [3, 2], [0, 3] ])
    num_nodes = 4
    num_node_features = 3
    num_edge_features = 5

    key = jax.random.PRNGKey(0)
    edge_features = jax.random.uniform( key, shape=(edges.shape[0], num_edge_features) )
    node_features = jax.random.uniform( key, shape=(num_nodes, num_node_features) )
    num_propagate_steps = 2
    output = mpnn_jit(edges, edge_features, node_features)
    print( output )
    for x in output:
        print( x.shape )
test()
