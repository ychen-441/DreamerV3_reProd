# 1. grad update via loss functions
# 2. update metrics using the updated grads
# 3. scaling and rescaling if enabled
# 4. several grad transformations before applying grad to upd

import math

import jax
import jax.numpy as jnp
import ninjax as nj
import optax

from . import internal
from . import nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):

  summary_depth: int = 2

  def __init__(self, modules, opt):
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules
    self.opt = opt
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      # opt if grad's finite, and stop when error steps reach 1000
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(self, lossfn, *args, has_aux=False, **kwargs):
    # *args for tuples and **kwargs for dicts
    # has_aux: whether lossfn returns auxiliary outputs or not
    metrics = {}

    # loss function
    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype) # check dtype
      assert loss.shape == (), (self.name, loss.shape) # check shape
      if self.scaling:
        loss *= sg(self.grad_scale.read()) # loss scaled by grad_scale if scaling
      return loss, aux


    loss, params, grads, aux = nj.grad(
        lossfn2, self.modules, has_aux=True)(*args, **kwargs) # grad
    if self.scaling:
      loss *= 1 / self.grad_scale.read() # loss rescaled

    # num of total params
    # e.g., {'conv1/weights':~, 'conv1/bias':~} by prod shape
    counts = {k: math.prod(v.shape) for k, v in params.items()}
    if nj.creating():
      print(self._summarize_params(counts, self.summary_depth))

    # avg grad if distributed
    axes = internal.get_data_axes()
    if axes:
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

    if self.scaling:
      invscale = 1 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads) # grads rescaled

    # optax.chain(*grad_transformations).init and .update
    # .init(params): initialize optimizer state for the params -> opt_state
    state = self.sub('state', nj.Tree, self.opt.init, params)
    # opt.update(grads, opt_state, params) -> params delta, updated opt_state
    updates, new_state = self.opt.update(grads, state.read(), params)
    nj.context().update(optax.apply_updates(params, updates)) # params updated and stored
    state.write(new_state) # updated state stored

    grad_norm = optax.global_norm(grads) # global norm of pytree
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm)) # update grad scale
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan) # mask if not finite
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm))) # move fwd if finite
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    metrics['grad_norm'] = grad_norm
    metrics['grad_rms'] = nets.rms(grads)
    metrics['update_rms'] = nets.rms(updates)
    metrics['param_rms'] = nets.rms([x.values for x in self.modules])
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    # summary: e.g., {'name/loss':~, 'name/updates', ...}
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000)) # grad_scale keep the same
    incr = (finite & (self.good_steps.read() >= 1000)) # grad_scale incr
    decr = ~finite # infinite/NaN
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    # return updated grad_scale from 1e-4 to 1e5
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      # with summary_depth = 1
      # counts: {'conv1/weights':~, 'conv1/bias':~} ->
      #       pfxs: {'conv1', 'conv1'} ->
      #             subcounts: {'conv1':~}
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    # final summary of params: 'Optimiser has ~ params: ~ conv1'
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


# grad transformations before updating params
# 1. adaptive gradient clipping
def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      # L2 norm of flattened update/param
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      # unorm > upper then update will be scaled down
      # otherwise remains unchanged by *1
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)

# 2. RMS scaling
def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    # nu_new <- beta*nu_old + (1-beta)*(grad)^2
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    # debias due to zero initialization
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)

# 3. momentum scaling
def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    # mu_new <- (1-beta)*updates + beta*mu_old
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      # accel nesterov grad
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step) # debias
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)
