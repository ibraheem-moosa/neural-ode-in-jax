import abc
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from .misc import _handle_unused_kwargs


class ScipyWrapperODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, rtol, atol, solver="LSODA", **unused_kwargs):
        unused_kwargs.pop('norm', None)
        unused_kwargs.pop('grid_points', None)
        unused_kwargs.pop('eps', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.dtype = y0.dtype
        self.shape = y0.shape
        self.y0 = np.array(y0).reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.func = func
        self.func = convert_func_to_numpy(func, self.shape, self.dtype)

    def integrate(self, t):
        # if t.numel() == 1:
        #     return torch.tensor(self.y0)[None].to(self.device, self.dtype)
        t = np.array(t)
        sol = solve_ivp(
            self.func,
            t_span=[t.min(), t.max()],
            y0=self.y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        sol = jnp.array(sol.y, dtype=self.dtype).T
        sol = sol.reshape(-1, *self.shape)
        return sol


def convert_func_to_numpy(func, shape, dtype):

    def np_func(t, y):
        t = jnp.array(t, dtype=dtype)
        y = jnp.reshape(jnp.array(y, dtype=dtype), shape)
        f = func(jax.lax.stop_gradient(t), jax.lax.stop_gradient(y))
        return np.asarray(f).reshape(-1)

    return np_func
