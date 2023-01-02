from jax_xenet.flax import XENet

import numpy as onp
import jax
import jax.numpy as jnp
import jax.numpy as np

import flax.linen as nn

def test_sparse_model_sizes():
    """
    This is a sanity check to make sure we have the same number of operations that we intend to have
    """
    N = 4
    F = 4
    S = 3

    key = jax.random.PRNGKey(0)
    node_features = jax.random.uniform( key, shape=(N, F) )
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 0], [2, 1], [3, 2], [0, 3] ])
    edge_features = jax.random.uniform( key, shape=(edges.shape[0], S) )
    

    def assert_n_params(model, expected_size):
        variables = model.init(jax.random.PRNGKey(0), node_features, edges, edge_features )
        param_count = 0
        for key, value in variables['params'].items():
            for k2, v2 in value.items():
                param_count += v2.size
        print( param_count, expected_size )
        assert param_count == expected_size
        # for test coverage:
        output = model.apply(variables, node_features, edges, edge_features )

    model = XENet( [5,], 10, 20, True )
    assert_n_params( model, 362 )
    # t = (4+4+3+3+1)*5 =  75
    # a = (5+1)*1   *2  =  12    # Attention
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 362

    model = XENet( [50, 5], 10, 20, True )
    assert_n_params( model, 1292 )
    # t1 = (4+4+3+3+1)*50   =  750
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *2      =   12    # Attention
    # x = (4+5+5+1)*10      =  150
    # e = (5+1)*20          =  120
    # p                     =    5    # Prelu
    # total = t+x+e+p       = 1292

test_sparse_model_sizes()
print( "Passed test_sparse_model_sizes" )
