import os
import threading
import collections

import ale_py
import ale_py.roms as roms
import elements
import embodied
import numpy as np

from PIL import Image


class Atari(embodied.Env):

  LOCK = threading.Lock()
  # weights for converting RGB to greyscale
  WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])
  ACTION_MEANING = (
      'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
      'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
      'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE')

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=0, lives='unused',
      sticky=True, actions='all', length=108000, pooling=2, aggregate='max',
      resize='pillow', autostart=False, clip_reward=False, seed=None):

    assert lives in ('unused', 'discount', 'reset'), lives
    assert actions in ('all', 'needed'), actions
    assert resize in ('opencv', 'pillow'), resize
    assert aggregate in ('max', 'mean'), aggregate
    assert pooling >= 1, pooling
    assert repeat >= 1, repeat
    # james bond act_space's a subset of ACTION_MEANING?
    # ale_py doc def: james_bond_actspace[18]
    # while actual op might be [6] (noop, 4 manips, fire), 
    # or [10] with redundant manips, [11] with 'upright_fire'
    if name == 'james_bond':
      name = 'jamesbond'

    # sticky action: might repeat prev_act
    # possible response delay or imperfect control
    # -> robustness, generalized policy
    self.repeat = repeat
    self.size = size
    self.gray = gray # grayscale
    self.noops = noops
    self.lives = lives
    self.sticky = sticky
    self.length = length
    self.pooling = pooling
    self.aggregate = aggregate
    self.resize = resize
    self.autostart = autostart
    self.clip_reward = clip_reward
    self.rng = np.random.default_rng(seed)

    with self.LOCK:
      self.ale = ale_py.ALEInterface()
      self.ale.setLoggerMode(ale_py.LoggerMode.Error)
      self.ale.setInt(b'random_seed', self.rng.integers(0, 2 ** 31))
      path = os.environ.get('ALE_ROM_PATH', None)
      if path:
        self.ale.loadROM(os.path.join(path, f'{name}.bin'))
      else:
        self.ale.loadROM(roms.get_rom_path(name))

    self.ale.setFloat('repeat_action_probability', 0.25 if sticky else 0.0)
    self.actionset = {
        'all': self.ale.getLegalActionSet,
        'needed': self.ale.getMinimalActionSet, # should be [6] for jb
    }[actions]()

    W, H = self.ale.getScreenDims()
    self.buffers = collections.deque(
        [np.zeros((W, H, 3), np.uint8) for _ in range(self.pooling)],
        maxlen=self.pooling)
    self.prevlives = None
    self.duration = None
    self.done = True # current episode ends or not

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (*self.size, 1 if self.gray else 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, len(self.actionset)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self.done:
      self._reset()
      self.prevlives = self.ale.lives() # extract lives agent still has
      self.duration = 0 # reset counter
      self.done = False
      return self._obs(0.0, is_first=True)

    reward = 0.0
    terminal = False
    last = False
    assert 0 <= action['action'] < len(self.actionset), action['action']
    act = self.actionset[action['action']]
    for repeat in range(self.repeat):
      reward += self.ale.act(act)
      self.duration += 1
      # store and rendering the last 2 frames
      # since Atari flickering sometimes, leading to info losses
      if repeat >= self.repeat - self.pooling:
        self._render()
      if self.ale.game_over():
        terminal = True
        last = True
      if self.duration >= self.length:
        last = True
      lives = self.ale.lives()
      if self.lives == 'discount' and 0 < lives < self.prevlives:
        terminal = True
      if self.lives == 'reset' and 0 < lives < self.prevlives:
        terminal = True
        last = True
      self.prevlives = lives
      if terminal or last:
        break
    self.done = last # True to reset
    obs = self._obs(reward, is_last=last, is_terminal=terminal)
    return obs

  def _reset(self):
    with self.LOCK:
      self.ale.reset_game()
    for _ in range(self.rng.integers(self.noops + 1)):
      # perfroms random nums of noop
      # introduce delay at start
      self.ale.act(self.ACTION_MEANING.index('NOOP'))
      if self.ale.game_over():
        with self.LOCK:
          self.ale.reset_game()
    if self.autostart and self.ACTION_MEANING.index('FIRE') in self.actionset:
      # some games reuire fire to start
      self.ale.act(self.ACTION_MEANING.index('FIRE'))
      if self.ale.game_over():
        with self.LOCK:
          self.ale.reset_game()
      # same, some need more than a press of fire
      self.ale.act(self.ACTION_MEANING.index('UP'))
      if self.ale.game_over():
        with self.LOCK:
          self.ale.reset_game()
    self._render() # render and store the initial frame
    for i, dst in enumerate(self.buffers):
      if i > 0:
        np.copyto(self.buffers[0], dst)

  def _render(self, reset=False):
    # shift right and overwrite with the newest
    self.buffers.appendleft(self.buffers.pop())
    self.ale.getScreenRGB(self.buffers[0])

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    if self.clip_reward:
      reward = np.sign(reward)
    if self.aggregate == 'max':
      image = np.amax(self.buffers, 0)
    elif self.aggregate == 'mean':
      image = np.mean(self.buffers, 0).astype(np.uint8)
    if self.resize == 'opencv':
      import cv2
      image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
    elif self.resize == 'pillow':
      image = Image.fromarray(image)
      image = image.resize(self.size, Image.BILINEAR)
      image = np.array(image)
    if self.gray:
      # Danijar:
      #   Averaging channels equally would not work. For example, a fully red
      #   object on a fully green background would average to the same color.
      # -> avg returns identical result instead of true blended color
      image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)[:, :, None]
    return dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    ) # formatting obs result
