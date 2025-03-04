import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from typing import Any, Callable, Sequence, Optional
import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen
from flax.core import freeze, unfreeze
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import sys
sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

"""
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
    """

# from easy_neural_ode.ode import odeint

from jax.experimental.ode import odeint


class DZDT(linen.Module):
    in_out_dim: int
    width: int

    @linen.compact
    def __call__(self, Z, t):
        print(Z, Z.shape, t, t.shape)
        # params = jnp.concatenate([Z, t], axis=0)
        params = Z + t
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
        dlogp_z_dt = -jnp.trace(df_dz)
        return dz_dt, dlogp_z_dt


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = jnp.array(points, dtype=jnp.float32)
    return x


if __name__ == '__main__':
    t0 = 0.
    t1 = 10.

    # model
    cnf = CNF(in_out_dim=2, width=args.width)
    loss_meter = RunningAverageMeter()

    """
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))
    """

    try:
        # @jax.jit
        def loss(params, x):
            start_and_end_times = jnp.array([0.,1.])
            z_t1 = x
            logp_z_t1 = jnp.zeros((1,))
            cnf_func = partial(cnf.apply, params)
            z, logp_z = odeint(cnf_func, (z_t1, logp_z_t1), start_and_end_times, atol=1e-3, rtol=1e-3)
            z_t0 = z[1, :]
            p_z0_mean = jnp.zeros((2,))
            p_z0_covariance_matrix = jnp.array([[0.1, 0.0], [0.0, 0.1]]) 
            logp_z_t0 = logp_z[1, :]
            logp_x = jax.scipy.stats.multivariate_normal.logpdf(z_t0, p_z0_mean, p_z0_covariance_matrix) - logp_z_t0
            return -logp_x.mean() 

        cnf_params = cnf.init(jax.random.PRNGKey(0), (jnp.ones((2,)), jnp.ones((1,))), jnp.array([0.]))
        x = get_batch(args.num_samples)[0,:]
        optimizer = optax.adam(learning_rate=args.lr)
        opt_state = optimizer.init(cnf_params)
        # loss = jax.vmap(loss, in_axes=(None, 0))
        # loss_grad_fn = jax.value_and_grad(lambda params, x: loss(params, x).mean())
        loss_grad_fn = jax.value_and_grad(loss)

        for itr in range(1, args.niters + 1):
            x = get_batch(args.num_samples)[0,:]
            loss_val, grads = loss_grad_fn(cnf_params, x)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(cnf_params, updates)
            loss_meter.update(loss_val.item())

            # if itr % 500 == 0:
            print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t_samples, _ = odeint(
                func,
                (z_t0, logp_diff_t0),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

            z_t_density, logp_diff_t = odeint(
                func,
                (z_t1, logp_diff_t1),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
