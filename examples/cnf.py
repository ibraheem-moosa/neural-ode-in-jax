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

from easy_neural_ode.ode import odeint

# from jax.experimental.ode import odeint


class CNF(linen.Module):
    in_out_dim: int
    hidden_dim: int
    width: int
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def _test_func(z, W, B, U, width, i):
        Z = jnp.expand_dims(z, 0).tile((width, 1, 1))
        h = jnp.tanh(jnp.matmul(Z, W) + B)
        dz_dt = jnp.matmul(h, U).mean(0)
        return dz_dt[:, i].sum(0)


    @linen.compact
    def __call__(self, t, states):
        z, logp_z = states

        batchsize = z.shape[0]
        W, B, U = HyperNetwork(self.in_out_dim, self.hidden_dim, self.width)(t)
        Z = jnp.expand_dims(z, 0).tile((self.width, 1, 1))
        h = jnp.tanh(jnp.matmul(Z, W) + B)
        dz_dt = jnp.matmul(h, U).mean(0)
        test_func_grad = jax.grad(CNF._test_func)
        sum_diag = 0
        for i in range(z.shape[1]):
            sum_diag = test_func_grad(z, W, B, U, self.width, 0)[:, i]
        # dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
        dlogp_z_dt = sum_diag

        return (dz_dt, dlogp_z_dt)


class HyperNetwork(linen.Module):
    in_out_dim: int
    hidden_dim: int
    width: int
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    @linen.compact
    def __call__(self, t):
        blocksize = self.width * self.in_out_dim
        # predict params
        params = t.reshape(1, 1)
        params = jnp.tanh(linen.Dense(self.hidden_dim)(params))
        params = jnp.tanh(linen.Dense(self.hidden_dim)(params))
        params = linen.Dense(3 * blocksize + self.width)(params)

        # restructure
        params = params.reshape(-1)
        W = params[:blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[blocksize:2 * blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * blocksize:3 * blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * jax.nn.sigmoid(G)

        B = params[3 * blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


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
    logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

    return(x, logp_diff_t1)


if __name__ == '__main__':
    t0 = 0
    t1 = 10

    # model
    cnf = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width)
    p_z0_mean = jnp.zeros((2,))
    p_z0_covariance_matrix = jnp.array([[0.1, 0.0], [0.0, 0.1]]) 
    """
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    )
    """
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
        def loss(params, batch):
            x, logp_diff_t1 = batch
            def cnf_func(t, states):
                print(states)
                return cnf.apply(params, t, states)
            z_t, logp_diff_t = odeint(
                cnf_func,
                (x, logp_diff_t1),
                jnp.array([t1, t0], dtype=jnp.float32),
                atol=1e-5,
                rtol=1e-5,
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            # logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            logp_x = jax.scipy.stats.multivariate_normal.log_pdf(z_t0, p_z0_mean, p_z0_covariance_matrix) - logp_diff_t0.reshape(-1)
            return -logp_x.mean(0)

        cnf_params = cnf.init(jax.random.PRNGKey(0), jnp.array([0]), (jnp.ones((args.num_samples, 2)), jnp.ones((args.num_samples, 1))))
        x, logp_diff_t1 = get_batch(args.num_samples)
        optimizer = optax.adam(learning_rate=args.lr)
        optimizer.init(cnf_params)
        loss_grad_fn = jax.value_and_grad(loss)

        for itr in range(1, args.niters + 1):
            x, logp_diff_t1 = get_batch(args.num_samples)
            loss_val, grads = loss_grad_fn(cnf_params, (x, logp_diff_t1))
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(cnf_params, updates)
            loss_meter.update(loss_val.item())

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
