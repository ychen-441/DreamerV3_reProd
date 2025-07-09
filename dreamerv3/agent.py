import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
# Danijar:
#   concise func def using lambda
#   e.g. def sg(xs, skip=False):..
#   stop gradient as a const
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs) 
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    # better to check how config is used in ./main.py
    # config params check configs.yaml
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude} # envs
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    # encoder as an example:
    #   {'simple': rssm.Encoder}['simple'] -> rssm.Encoder
    #   **config.enc[config.enc.typ] fetch params from configs
    #   Encoder uses enc_space and params 
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc') # encoder
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn') # sequence model
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec') # decoder

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1) # (b, 4096+32*32)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    # check ./embodied/jax/heads.py
    # .MLPHead(space, output, **hkw)
    # where 'output' def in configs.yaml
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew') # reward/MLP*1
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con') # continue/MLP*1

    # KEY Actor-Critic part init
    # output for disc: categorical; for cont: bounded_normal
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol') # actor/MLP*3
    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val') # critic/MLP*3
    
    # check .SlowModel and .Normalize in ./embodied/jax/utils.py
    # slow target network for stablizing critic learning
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm') # return norm by percentile
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm') # critic norm
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm') # advantage norm

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    # check ./embodied/jax/opt.py
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    # 'rec' for reconstruction
    rec = scales.pop('rec')
    # add obs key-val pairs with val=rec
    scales.update({k: rec for k in dec_space})
    self.scales = scales

  @property
  def policy_keys(self):
    # matching with enc/, dyn/, dec/, and pol/
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      # single-level dict with something like 'enc/deter', 'dyn/stoch', etc.
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    # return zero carries with 'deter' and 'stoch', and act_space
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  # check rssm.py if you get confused about any terms
  def policy(self, carry, obs, mode='train'):
    # {'deter', 'stoch'}
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True) # single step
    reset = obs['is_first'] # .Space(bool)
    # encoding
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    # dynamics prediction
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      # decoding
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    # pol: MLPHead(x, bdims) (check ./embodied/jax/heads.py)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy) # actions sampled from policy
    out = {}
    # check invalid elements
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act) # carry update
    if self.config.replay_context:
     # add entries to replay buffer
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    # opt using lossfn below 
    # return loss, (carry, entries, outs, metrics) as outs in ~/opt.py
    # then outs -> loss, aux -> (metrics, aux)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update() # target update
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      # check if bsize the same for all items in updates
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates # ???context???
    
    # Danijar:
    #   if self.config.replay.fracs.priority > 0:
    #     outs['replay']['priority'] = losses['model']

    # add the latest actions of each key from act_space
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first'] # store B, T shape
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    # los: dyn, rep KL loss; met: dyn, rep entropy
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)

    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    # ~/heads.py MLPHead -> ~/outs.py .loss(~)
    # MLPHead(): 'rew'/'con' -> 
    #       .loss(): symexp_twohot(obs['reward'])/binary(con)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward']) # reward loss
    con = f32(~obs['is_terminal'])
    # contdisc for discount(con) factor flag
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con) # continue loss
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target)) # reconstruction loss

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    # feels like K step warm up
    K = min(self.config.imag_last or T, T) # 0 then K = T
    H = self.config.imag_length # 15, imag_horizon
    starts = self.dyn.starts(dyn_entries, dyn_carry, K) # extract K latest steps
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1)) # sample act from pol
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat) # feat: (B*K, 1,)
    # concat previous (first) and imag feat
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1) # feat: (B*K, H+1,)
    # action based on the latest imag_fat
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat)) # act: (B*K,)
    # add a T axis to match imgprevact
    lastact = jax.tree.map(lambda x: x[:, None], lastact) # act: (B*K, 1,)
    # concat previous and imag act
    imgact = concat([imgprevact, lastact], 1) # concat: (B*K, H+1,)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact, # imag_act
        self.rew(inp, 2).pred(), # pred_rew
        self.con(inp, 2).prob(1), # prob of con or terminated
        self.pol(inp, 2), # policy/actor
        self.val(inp, 2), # critic
        self.slowval(inp, 2), # critic target
        self.retnorm, self.valnorm, self.advnorm, # return,value, advantage norm
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon, # 333, discount horizon
        **self.config.imag_loss)
    # update actor, imag critic loss
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # an additional replay buffer critic loss
    # for envs where the rew is hard to predict
    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      # lambda return at the start of imag traj
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los) # update replay buffer critic loss
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    # scale as beta; 0.1 for val and 0.3 for repval
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B) # reduced batch size
    metrics = {}

    # Train metrics
    # update = False -> ret/adv/val norm without update; check ~/utils.py
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          # outs['losses'][key].mean()
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          # nj.grad() return loss, params, grads
          # [-1] gives grads only, which is grad of lossfn w.r.t. modules
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    # truncate
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    # dyn using first half
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False) # zeros if isfirst
    # imag using upd dyn and second half actions
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    # recon first half obs, recon good enough or not
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    # recon imag feat, dyn good enough or not
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]

      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      # convert [0, 1] back to [0, 255]
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      # padding on H and W for borders
      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True) # false the pads 
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8) # green for obs
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8)) # red for imag
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1) # 10 black frames

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space}) # latest action
    return carry, metrics

  # carry if is not a new chunk
  # then prevact shifts backwards to get the latest act to this chunk
  # otherwise, use context to warm up and full seq of act 
  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    # previous 1-step action prepend to (T-1) act sequence
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context # 1 by default
    # nested dict turns, e.g., 
    # 'enc/deter', 'dyn/stoch' to 'enc':{~}, 'dyn':{~}, etc.
    nested = elements.tree.nestdict(data)
    # keys extracted from nested data
    # entries[0/1/2] = 'enc'/'dyn'/'dec'
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]

    # replay context using lhs, rhs for training
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs) # [0]
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs) # [1:]
    rep_carry = (
        # truncate() fetch the latest entry and return as carry
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    # check if it is a start of a new chunk
    first_chunk = (data['consec'][:, 0] == 0)
    # replay if first_chunk, otherwise normal
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)), # normal
        (rep_carry, rep_obs, rep_prevact, rep_stepid)) # replay
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    # a series of grad transformation def in ~/opt.py
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd: # weight decay
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    # learning rate schedule
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      # linear decay from lr to 0.1 lr
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
      # same but cosine decay
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr) 
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup) # linear warm-up from 0 to lr
      sched = optax.join_schedules([ramp, sched], [warmup]) # switch to lr schedule
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


# losses, raw lambda returns, and metrics;
# regularize not only towards value estimation 
# but also towards a slow EMA version of itself
def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats() # 'none', so 0, 1; check ~/utils.py
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon # 1
  weight = jnp.cumprod(disc * con, 1) / disc # cumprod(con)
  last = jnp.zeros_like(con)
  term = 1 - con # terminal
  # lambda return 
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update) # 'perc'
  adv = (ret - tarval[:, :-1]) / rscale # calculate advantage, then norm
  aoffset, ascale = advnorm(adv, update) # 'none', so 0, 1
  adv_normed = (adv - aoffset) / ascale # then norm

  # log prob of pol-pred acts, and entropy of pol
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss # actor/policy loss

  voffset, vscale = valnorm(ret, update) # 'none', so 0, 1
  tar_normed = (ret - voffset) / vscale # then norm
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  # critic/value loss w.r.t. both ret-based val pred and slow target val
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale # norm

  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


# off-policy with imag 'ret' and obs['rew']
# details refer to imag_loss 
# which holds sort of the same workflow
def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  # norm
  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  # diff from imag_loss(~): disc and weight
  disc = 1 - 1 / horizon
  weight = f32(~last)
  # lambda return
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1] # same as imag_loss

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


# lambda return for value estimation
def lambda_return(last, term, rew, val, boot, disc, lam):
  # boot for bootstraps
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  # term = 1 - con
  # last = 0
  live = (1 - f32(term))[:, 1:] * disc # ≡ con * disc
  cont = (1 - f32(last))[:, 1:] * lam # ≡ lam
  # intermediate return = 
  #       immediate reward rew[:, 1:] + 
  #             discounted bootstrapped value live * boot[:, 1:] 
  # weighted by discount factor (1-cont)
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  # reverse recursion to build lambda return
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  # # and reverse it back to see lambda ret along the trajectory
  return jnp.stack(list(reversed(rets))[:-1], 1)
