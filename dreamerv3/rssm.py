# def RSSM, encoder, and decoder;
# KL loss for transition pedictor (prior)
# and representation model (posterior).


import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  # block-diagonal recurrent weights for param efficiency
  blocks: int = 8
  free_nats: float = 1.0 

  def __init__(self, act_space, **kw):
    # **kw takes additional arguments implicitly
    assert self.deter % self.blocks == 0
    # tasks' defs in ./embodied/envs/
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    # elements is maintained by Danijar: https://github.com/danijar/elements
    # .Space(dtype, shape)
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    # zeros and transform to bfloat16
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    # take only the latest step, [:, -1, :]
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    # batch size = B
    B = len(jax.tree.leaves(carry)[0]) 
    # extract nlast steps then flatten B*nlast
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  # update carry with observations
  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      # single step
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      # _observe updated carry, entry, feat
      return carry, entry, feat
    else:
      # unroll = True: check shape[1] for sequence length
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      # scan over axis 1, i.e., sequence
      # like an offline if sequential?
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

# the main difference between obs and img is that
#   1. img uses action given by policy(deter, stoch), then update deter
#   2. obs updates logits (hence stoch) with (upd_deter, tokens)
#      while img updates logits with only (upd_deter)
  def _observe(self, carry, tokens, action, reset, training):
    # zeros states if reset
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    # act_space def in ./embodied/envs/minecraft_flat.py
    # .DictConcat def in ./embodied/jax/nets.py
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    deter = self._core(deter, stoch, action) # GRU updated deter
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    
    # a sample of disc stoch z_t with ST gradients
    # (, 2048) -> (, 32*32) -> (, 32, 32)
    logit = self._logit('obslogit', x)
    # one-hot sample with straight-through gradients
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      # get policy-based actions or pre-sampled ones
      # ???addr of policy def???
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action) # one hot
      deter = self._core(carry['deter'], carry['stoch'], actemb) # GRU update
      
      # prior stoch without obs
      # (bsize, deter) -> 
      #     (bsize, hidden) -> 
      #           (bsize, stoch*classes) -> 
      #                 (bsize, stoch, classes)
      logit = self._prior(deter)
      # one-hot sample with straight-through gradients
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # Danijar:
      #   We can also return all carry entries but it might be expensive.
      #   entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      #   return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training) # now with obs
    # here's the point:
    #   feat['deter'] is updated without obs, thus prior
    #   feat['logit'] is updated with concat deter and tokens, thus posterior
    prior = self._prior(feat['deter'])
    post = feat['logit']
    # KL losses
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    # freebits implementation
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  # block-wise GRU update using deter, stoch, and action
  def _core(self, deter, stoch, action):
    # keep bdim and flatten stoch and classes dim
    stoch = stoch.reshape((stoch.shape[0], -1))
    # normalization due to varying scales and magnitudes
    # sg for avoiding degenerated solutions while keeping numerical stability by norm
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks # 8
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g) # (g,h)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g) # (g*h)

    # deter/stoch/action encoded into a norm, activated hidden space (2048)
    # Linear/act/Norm def in ./embodied/jax/nets.py
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    # (, 2048) -> (, 2048*3) -> (, 1, 2048*3) -> (, 8, 2048*3)
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    # (, 4096) -> (, 8，512); x -> (, 8, 512+2048*3) -> (, 8*(512+2048*3))
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    
    # block-wise Linear (also in nets.py)
    for i in range(self.dynlayers):
      # each block: (, 512+2048*3) -> (, 512) then concat back to (, 4096)
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))

    # (, 4096/8) -> (, 4096*3/8) then concat back to (, 4096*3)
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    # (, 4096*3) -> (, 8, 4096*3/8) -> (, 8, 4096/8) * 3 gates
    gates = jnp.split(flat2group(x), 3, -1)
    # (, 8, 4096/8) -> (, 4096)
    reset, cand, update = [group2flat(x) for x in gates]
    # GRU updates
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

# Check the following three in ./embodied/jax/outs.py
  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    # reshape (, stoch*classes) to (, stoch, classes)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    # In fact a nested Agg(OneHot(Categorical(logits)))
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    # dims = 1 then agg along -1, i.e., the last axis
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space # check ./embodied/envs/minecraft_flat.py
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    # CNN (128, 192, 256, 256)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2 # (B,) or (B, T)
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      # vector obs from current input
      vecs = {k: obs[k] for k in self.veckeys}
      # symlog squish for stability (check V3 paper)
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs) # (B, T, concat_vec)
      # (B, T, concat_vec) -> (B*T, concat_vec)
      x = x.reshape((-1, *x.shape[bdims:]))
      # MLP with 3 dense layers 
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs) # ensure img dtype
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5 # norm (-0.5, 0.5)
      # (B, T, H, W, C*imgs_num) -> (B*T, H, W, C*imgs_num)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          # 2*2 max pooling manually
          # H,W -> 2*2 -> max -> (B, H, W, C)
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1)) # concat H, W, and C
      outs.append(x)

    x = jnp.concatenate(outs, -1) 
    tokens = x.reshape((*bshape, *x.shape[1:])) # (bsize, features)
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True 
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    # # CNN (128, 192, 256, 256)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    # channels and resolution of an image
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    # inp might for input
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')] # dtype conversion
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp] # flatten b_size
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:])) # reshape
      kw = dict(**self.kw, outscale=self.outscale)
      # check .DictHead in ./embodied/jax/heads.py
      # -> {'space': spaces}, {'space': outputs}
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      # initial minres
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1 ] <= 16, minres
       # (4, 4, 256), start with minres with deepest channel
      shape = (*minres, self.depths[-1])
      # blockwise linear ops with 8 blocks
      if self.bspace:
        # u = 4*4*256, g = 8
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1)) # flatten (, 32, 32)
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        # x0.shape: (4096)
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        #       -> (4, 4, 256)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        # x1.shape: (1024) -> (2048)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        #       -> (4, 4, 256)
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))

        
      for i, depth in reversed(list(enumerate(self.depths[:-1]))): # 256->192->128
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          # (8, 8, 256) ->
          #       (16, 16, 192) ->
          #             (32, 32, 128)
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        # (32, 32,) -> (64, 64,)
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        # (64, 64, imgdep)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x) # pixel val norm
      x = x.reshape((*bshape, *x.shape[1:])) # reshape bsize
      # split channels for each imgkey
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
