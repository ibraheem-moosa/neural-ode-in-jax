import jax
import jax.numpy as jnp
import flax
from flax import linen
from jax.experimental.ode import odeint
from functools import partial


class HyperNetwork(linen.Module):
    hypernetwork_hidden_dim: int
    in_out_dim: int
    width: int

    @linen.compact
    def __call__(self, t):
        print(t, t.shape)
        batch_size = -1#t.shape[0]
        blocksize = self.width * self.in_out_dim
        # predict params
        params = jnp.tanh(linen.Dense(self.hypernetwork_hidden_dim)(t))
        params = jnp.tanh(linen.Dense(self.hypernetwork_hidden_dim)(params))
        W = linen.Dense(blocksize)(params)
        B = linen.Dense(self.width)(params)
        U = linen.Dense(blocksize)(params)
        G = linen.Dense(blocksize)(params) 
        U = U * linen.sigmoid(G)
        print('W, B, U:', W.shape, B.shape, U.shape)
        # restructure
        W = W.reshape((batch_size, self.width, self.in_out_dim))
        B = B.reshape((batch_size, self.width,))
        U = U.reshape((batch_size, self.width, self.in_out_dim))
        return [W, B, U]

class DZDT(linen.Module):
    in_out_dim: int
    width: int

    @linen.compact
    def __call__(self, Z, t):
        # t = jnp.tile(jnp.expand_dims(t, 0), (Z.shape[0], 1))
        params = jnp.hstack([Z, t])
        # params = jnp.tanh(linen.Dense(self.in_out_dim)(t)) + Z
        params = jnp.tanh(linen.Dense(self.width)(params))
        params = jnp.tanh(linen.Dense(self.width)(params))
        params = linen.Dense(self.in_out_dim)(params)
        return params

class CNF(linen.Module):
    in_out_dim: int
    width: int

    def setup(self):
        self.dz_dt_func = DZDT(self.in_out_dim, self.width)

    def __call__(self, states, t):
        Z, logp_z = states

        dz_dt = self.dz_dt_func(Z, t)
        df_dz = jax.jacrev(self.dz_dt_func)(Z, t)
        dlogp_z_dt = jnp.trace(df_dz)

        return dz_dt, dlogp_z_dt


batch_size = 512
in_out_dim = 2
hypernetwork_hidden_dim = 32
width = 64

"""
key, hypernetwork_init_key = jax.random.split(jax.random.PRNGKey(0), 2)
hypernetwork = HyperNetwork(hypernetwork_hidden_dim, in_out_dim, width)
params = hypernetwork.init(hypernetwork_init_key, jnp.zeros((1,)))
t = jnp.array([0.0, 1.0]).reshape((2, 1))
W, B, U = hypernetwork.apply(params, t)
print(W.shape, B.shape, U.shape)
"""

Z = jnp.ones((in_out_dim,))
logp_Z = jnp.zeros((1,))
t = jnp.array([0.])

key, cnf_init_key = jax.random.split(jax.random.PRNGKey(0), 2)
cnf = CNF(in_out_dim, width)
params = cnf.init(cnf_init_key, (Z, logp_Z), t)
y = cnf.apply(params, (Z, logp_Z), t)

def loss(params, x):
    start_and_end_times = jnp.array([0.,1.])
    z_t1 = x
    logp_z_t1 = jnp.zeros((1,))
    cnf_func = partial(cnf.apply, params)
    z, logp_z = odeint(cnf_func, (z_t1, logp_z_t1), start_and_end_times, atol=1e-3, rtol=1e-3)
    z_t0 = z[1, :]
    logp_z_t0 = logp_z[1, :]
    logp_x = jax.scipy.stats.multivariate_normal.logpdf(z_t0, p_z0_mean, p_z0_covariance_matrix)
    return -logp_x.mean() 


print(loss(params, Z))

# Z = jnp.ones((batch_size, in_out_dim,))
# logp_Z = jnp.zeros((batch_size, 1,))
# print(cnf.apply(params, (Z, logp_Z), t))
