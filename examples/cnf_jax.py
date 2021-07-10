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
        blocksize = self.width * self.in_out_dim
        print(t)
        # predict params
        params = jnp.tanh(linen.Dense(self.hypernetwork_hidden_dim)(t))
        params = jnp.tanh(linen.Dense(self.hypernetwork_hidden_dim)(params))
        W = linen.Dense(blocksize)(params)
        B = linen.Dense(self.width)(params)
        U = linen.Dense(blocksize)(params)
        G = linen.Dense(blocksize)(params) 
        U = U * linen.sigmoid(G)

        # restructure
        W = W.reshape(-1, self.width, self.in_out_dim)
        B = B.reshape(-1, self.width,)
        U = U.reshape(-1, self.width, self.in_out_dim)
        return [W, B, U]

class CNF(linen.Module):
    hypernetwork_hidden_dim: int
    in_out_dim: int
    width: int

    def f(Z, W, B, U):
        return jnp.matmul(jnp.tanh(jnp.matmul(W, Z) + B), U)


    @linen.compact
    def __call__(self, states, t):
        Z, logp_z = states
        W, B, U = HyperNetwork(self.hypernetwork_hidden_dim, self.in_out_dim, self.width)(t)
        dz_dt = CNF.f(Z, W, B, U)
        df_dz = jax.jacrev(CNF.f)(Z, W, B, U)
        dlogp_z_dt = jnp.trace(df_dz)
        return dz_dt, dlogp_z_dt


batch_size = 512
in_out_dim = 2
hypernetwork_hidden_dim = 32
width = 64

key, hypernetwork_init_key = jax.random.split(jax.random.PRNGKey(0), 2)
hypernetwork = HyperNetwork(hypernetwork_hidden_dim, in_out_dim, width)
params = hypernetwork.init(hypernetwork_init_key, jnp.zeros((1,)))
t = jnp.array([0.0, 1.0]).reshape((2, 1))
W, B, U = hypernetwork.apply(params, t)
print(W.shape, B.shape, U.shape)

Z = jnp.ones((in_out_dim,))
logp_Z = jnp.zeros((1,))
t = jnp.zeros((1,))

key, cnf_init_key = jax.random.split(jax.random.PRNGKey(0), 2)
cnf = CNF(hypernetwork_hidden_dim, in_out_dim, width)
params = cnf.init(cnf_init_key, (Z, logp_Z), t)

start_and_end_times = jnp.array([0.,1.])
cnf_func = partial(cnf.apply, params)
init_state, final_state = odeint(cnf_func, (Z, logp_Z), start_and_end_times, atol=1e-3, rtol=1e-3)


# Z = jnp.ones((batch_size, in_out_dim,))
# logp_Z = jnp.zeros((batch_size, 1,))
# print(cnf.apply(params, (Z, logp_Z), t))
