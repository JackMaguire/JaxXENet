import numpy as onp
import jax
import jax.numpy as jnp
import jax.numpy as np
from jax import grad, jit, vmap, nn

import flax.linen as nn
from typing import Sequence

class KerasStylePReLU(nn.Module):
    #negative_slope_init: float = 0.01
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        negative_slope_init = 0.0 #keras style
        feat = x.shape[-1]
        negative_slope = self.param(
            "negative_slope",
            lambda k: jnp.zeros( shape=(feat,), dtype=x.dtype)
        )
        #print( x.shape, negative_slope.shape )
        return (negative_slope*x*(x<=0)) + (x*(x>0))
        
    #return jnp.where(x >= 0, x, negative_slope * x)

class XENet(nn.Module):
    stack_sizes: Sequence[int]
    Fout: int
    Sout: int
    debug_mode: bool = True

    @nn.compact
    def __call__(self, x_in, a_in, e_in):
        """
        Message passing neural network (MPNN) implemented in Jax.

        Parameters
        ----------
        x_in : np.ndarray
            Node features of shape (num_nodes, num_node_features).
        a_in : np.ndarray
            Sparse adjacency matrix of shape (num_edges, 2). Each row represents an edge,
            with the first and second column containing the indices of the source and
            destination nodes, respectively.
        e_in : np.ndarray
            Edge features of shape (num_edges, num_edge_features).

        Returns
        -------
        np.ndarray
            Updated node features of shape (num_nodes, num_node_features).
        """

        num_nodes = x_in.shape[0]
        num_node_features = x_in.shape[-1]
        num_edges = e_in.shape[0]
        num_edge_features = e_in.shape[-1]
        Fin = num_node_features
        Sin = num_edge_features

        reverse_edge_error_flag = np.ones((1,))

        # Initialize messages with zeros
        ss = self.stack_sizes[-1]
        incoming_stacks = np.zeros((num_nodes, ss))
        outgoing_stacks = np.zeros((num_nodes, ss))
        all_stacks      = np.zeros((num_edges, 2*Fin + 2*Sin))

        # Iterate over edges
        for edge_idx in range(a_in.shape[0]):
            src_node, dest_node = a_in[edge_idx]

            # Check if there is a reverse edge
            reverse_edge_mask = (a_in[:, 0] == dest_node) & (a_in[:, 1] == src_node)
            rev_edge_idx = np.argmax(reverse_edge_mask)

            if self.debug_mode:
                _dest_node, _src_node = a_in[rev_edge_idx]
                reverse_edge_error_flag *= np.equal( dest_node, _dest_node )
                reverse_edge_error_flag *= np.equal( src_node, _src_node )

            x_i = x_in[ src_node ]
            x_j = x_in[ dest_node ]
            e_ij = e_in[ edge_idx ]
            e_ji = e_in[ rev_edge_idx ]

            stack = np.concatenate(
                [ x_i, x_j, e_ij, e_ji ],
                axis=-1
            )

            all_stacks = all_stacks.at[edge_idx].set( stack )

        # Compute message for destination node using MLP
        for i, feat in enumerate( self.stack_sizes ):
            all_stacks = nn.Dense(feat)(all_stacks)
            if i < len(self.stack_sizes)-1:
                all_stacks = nn.relu( all_stacks )
            else:
                #all_stacks = nn.PReLU()( all_stacks )
                all_stacks = KerasStylePReLU()( all_stacks )
                #print( "!!!", all_stacks.get_variable() )

        # TODO attention
        incoming_att_sigmoid = nn.sigmoid(nn.Dense(1)(stack))
        outgoing_att_sigmoid = nn.sigmoid(nn.Dense(1)(stack))

        for edge_idx in range(a_in.shape[0]):
            src_node, dest_node = a_in[edge_idx]

            # Accumulate messages for nodes
            incoming_stacks = incoming_stacks.at[dest_node].add( all_stacks[edge_idx] )
            outgoing_stacks = outgoing_stacks.at[src_node].add( all_stacks[edge_idx] )

        #########
        # NODES #
        #########

        # Aggregate messages and node features
        x_out = np.concatenate(
            [x_in, incoming_stacks, outgoing_stacks],
            axis=-1
        )
        # Apply MLP to update node features
        x_out = nn.relu(nn.Dense(self.Fout, name="Dense_x_out" )(x_out))

        #########
        # EDGES #
        #########

        # Apply MLP to update edge features
        e_out = nn.relu(nn.Dense(self.Sout, name="Dense_e_out")(all_stacks))

        ##########
        # ERRORS #
        ##########

        # flip all flags such that 0 means "pass"
        reverse_edge_error_flag = 1 - reverse_edge_error_flag

        if self.debug_mode:
            return x_out, e_out, reverse_edge_error_flag
        else:
            return x_out, e_out, None

    

model = XENet( [64,64], 3, 4, True ) 

def test():
    #edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 0], [2, 1], [3, 2], [0, 3] ])
    num_nodes = 4
    num_node_features = 3
    num_edge_features = 5

    key = jax.random.PRNGKey(0)
    edge_features = jax.random.uniform( key, shape=(edges.shape[0], num_edge_features) )
    node_features = jax.random.uniform( key, shape=(num_nodes, num_node_features) )

    variables = model.init(jax.random.PRNGKey(0), node_features, edges, edge_features )
    print( "VAR", len(variables['params']) )
    print( variables )

    output = model.apply(variables, node_features, edges, edge_features )

    #print( output )
    for x in output:
        print( x.shape )
        print( x )
test()
