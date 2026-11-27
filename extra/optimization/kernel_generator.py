#!/usr/bin/env python3
# a generator of small model-like tinygrad programs: a source of kernels for the search dataset that needs no list of real programs.
# each seed gives one random network (convolutional, transformer, recurrent or tabular) built from the blocks models are made of. weights and
# inputs are Tensor.empty, and the network runs once forward or for one training step, so the scheduler fuses kernels the way it does for a model.
# usage: python kernel_generator.py <seed> [<seed> ...]
import sys, random
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Context, prod
from tinygrad.nn.state import get_parameters

# how often each activation is a program's main one: in networks of convolutions, and in networks over sequences
ACTS = {"relu": 62, "silu": 11, "gelu": 4, "relu6": 7, "leaky_relu": 4, "hardswish": 5, "sigmoid": 1, "tanh": 2, "mish": 1, "elu": 1, "quick_gelu": 1,
        "softplus": 1}
SEQ_ACTS = {"gelu": 52, "relu": 22, "silu": 10, "quick_gelu": 6, "tanh": 4, "leaky_relu": 2, "mish": 2, "sigmoid": 1, "none": 1}
# a training step makes several times the kernels a forward pass does, so one program in five trains, which gives about two kernels in five
TRAIN = 0.2
# an exported network has its batch norms folded into the convolutions before them, which leaves a bias: most inference programs look like that
NORMS = {True: {"batch": 50, "none": 18, "group": 15, "layer": 10, "instance": 7},
         False: {"none": 55, "batch": 33, "group": 6, "layer": 4, "instance": 2}}
FAMILIES = {"conv2d": 44, "unet": 8, "decoder": 5, "transformer": 24, "conv1d": 5, "conv3d": 4, "recurrent": 4, "tabular": 6}
BATCHES = [1, 1, 1, 2, 2, 3, 4, 4, 6, 8, 8, 10, 12, 16, 16, 24, 32, 32, 48, 64, 100, 128, 256]
MAX_NUMEL = 1 << 25   # no activation in a program is bigger than this, so every kernel fits a small GPU

def E(*shape, dtype=None) -> Tensor: return Tensor.empty(*shape, dtype=dtype)

class Program:
  def __init__(self, seed:int, family:str|None=None):
    self.r, self.mods = random.Random(seed), []
    r = self.r
    self.family = family or r.choices(list(FAMILIES), list(FAMILIES.values()))[0]
    self.train = r.random() < TRAIN
    self.one = not self.train and r.random() < 0.6   # most forward passes are run on one input at a time
    acts = SEQ_ACTS if self.family in ("transformer", "recurrent") else ACTS
    self.act_name = r.choices(list(acts), list(acts.values()))[0]
    self.norm_name = r.choices(list(NORMS[self.train]), list(NORMS[self.train].values()))[0]
    self.ft = dtypes.half if r.random() < 0.06 else None   # a few programs keep their weights and inputs in half precision
    self.cond: Tensor|None = None
    self.snorm, self.prenorm, self.qkv_bias, self.layerscale, self.mlp_kind, self.mlp_ratio = "layer", True, True, False, "mlp", 4
    self.desc = [self.family, "train" if self.train else "infer", self.act_name, self.norm_name] + (["half"] if self.ft else [])

  # *** weights, activations, norms ***

  def reg(self, m):
    # layers are made with random or constant weights: swap them for empty buffers, which is what a model with loaded weights looks like
    for p in get_parameters(m): p.replace(E(*p.shape, dtype=self.ft if self.ft and dtypes.is_float(p.dtype) else p.dtype)).realize()
    self.mods.append(m)
    return m
  def P(self, *shape) -> Tensor:
    self.mods.append(t:=E(*shape, dtype=self.ft).realize())
    return t
  def pick(self, *xs): return self.r.choice(xs)
  def bs(self, per_item:int, lo:int=16, hi:int=21) -> int:
    # a batch size that gives the first activation between 2**lo and 2**hi elements
    want = 2**self.r.uniform(lo, hi)/per_item
    return 1 if self.one else min(BATCHES, key=lambda b: (abs(b-want), self.r.random()))
  def act(self, x:Tensor, name:str|None=None) -> Tensor:
    name = name or (self.act_name if self.r.random() < 0.92 else self.r.choice(list(ACTS)))
    return x if name == "none" else getattr(x, name)()
  def norm(self, x:Tensor, name:str|None=None) -> Tensor:
    name, c = name or self.norm_name, x.shape[1]
    if name == "layer" and x.ndim != 4: name = "group"
    if name == "none": return x
    if name == "batch": return self.reg(nn.BatchNorm(c))(x)
    if name == "group": return self.reg(nn.GroupNorm(self.r.choice([g for g in (1, 2, 4, 8, 16, 32) if c % g == 0]), c))(x)
    if name == "instance": return self.reg(nn.InstanceNorm(c))(x)
    return self.reg(nn.LayerNorm2d(c))(x)
  def seqnorm(self, x:Tensor) -> Tensor:
    return x if self.snorm == "none" else self.reg((nn.RMSNorm if self.snorm == "rms" else nn.LayerNorm)(x.shape[-1]))(x)
  def linear(self, x:Tensor, out:int, bias:bool=True) -> Tensor: return self.reg(nn.Linear(x.shape[-1], out, bias=bias))(x)
  def drop(self, x:Tensor, p:float=0.1) -> Tensor: return x.dropout(p) if self.train and self.r.random() < 0.3 else x

  # *** convolutional blocks, for any number of spatial dims ***

  def conv(self, x:Tensor, cout:int, k=3, stride=1, groups=1, dilation=1, bias:bool|None=None) -> Tensor:
    ks = (k,)*(x.ndim-2) if isinstance(k, int) else k
    if bias is None: bias = self.norm_name == "none" or self.r.random() < 0.15
    return self.reg(nn.Conv2d(x.shape[1], cout, ks, stride, tuple(dilation*(kk-1)//2 for kk in ks), dilation, groups, bias))(x)
  def cna(self, x:Tensor, cout:int, k=3, stride=1, groups=1, dilation=1, act:bool=True) -> Tensor:
    x = self.norm(self.conv(x, cout, k, stride, groups, dilation))
    return self.act(x) if act else x
  def pre(self, x:Tensor, cout:int, k=3, stride=1) -> Tensor: return self.conv(self.act(self.norm(x)), cout, k, stride)
  def width(self, c:int, mult:float) -> int: return max(8, min(2048, int(c*mult)//8*8))
  def with_cond(self, x:Tensor) -> Tensor:
    # a conditioning vector (a timestep or class embedding) added to every channel, as diffusion models do
    if self.cond is None: return x
    return x + self.linear(self.act(self.cond), x.shape[1]).reshape(x.shape[0], x.shape[1], *[1]*(x.ndim-2))
  def shortcut(self, x:Tensor, cout:int, stride:int) -> Tensor:
    if x.shape[1] == cout and stride == 1: return x
    if stride > 1 and all(d % stride == 0 for d in x.shape[2:]) and self.r.random() < 0.25: x, stride = self.pool(x, "avg", stride), 1
    return self.cna(x, cout, 1, stride, act=False)
  def pool(self, x:Tensor, kind:str, stride:int=2, k:int|None=None) -> Tensor:
    nd = x.ndim-2
    k = k or stride
    fn = x.max_pool2d if kind == "max" else x.avg_pool2d
    return fn((k,)*nd, stride, padding=(k-stride+1)//2 if k > stride else 0)
  def se(self, x:Tensor, ratio:int=4) -> Tensor:
    # squeeze and excitation: a gate per channel from the global average
    s = x.mean(tuple(range(2, x.ndim)), keepdim=True)
    s = self.act(self.conv(s, max(8, x.shape[1]//ratio), 1, bias=True))
    return x * getattr(self.conv(s, x.shape[1], 1, bias=True), self.pick("sigmoid", "sigmoid", "hardsigmoid"))()

  def b_plain(self, x, c, s): return self.cna(x, c, self.pick(1, 3, 3, 3, 5, 7), s, dilation=self.pick(1, 1, 1, 2))
  def b_basic(self, x, c, s):
    y = self.with_cond(self.cna(x, c, 3, s))
    return self.act(self.cna(y, c, 3, act=False) + self.shortcut(x, c, s))
  def b_preact(self, x, c, s):
    y = self.with_cond(self.pre(x, c, 3, s))
    return self.pre(y, c, 3) + (x if x.shape[1] == c and s == 1 else self.conv(x, c, 1, s))
  def b_bottleneck(self, x, c, s):
    mid = self.width(c, self.pick(0.25, 0.25, 0.5))
    y = self.cna(x, mid, 1)
    y = self.cna(y, mid, 3, s, groups=self.pick(1, 1, 1, mid, *([32] if mid % 32 == 0 else [])))
    return self.act(self.cna(y, c, 1, act=False) + self.shortcut(x, c, s))
  def b_mbconv(self, x, c, s):
    mid = self.width(x.shape[1], self.pick(1, 3, 4, 6))
    y = self.cna(x, mid, 1) if mid != x.shape[1] else x
    y = self.cna(y, mid, self.pick(3, 3, 5, 7), s, groups=mid)
    if self.r.random() < 0.5: y = self.se(y)
    y = self.cna(y, c, 1, act=False)
    return y + x if y.shape == x.shape else y
  def b_sepconv(self, x, c, s): return self.cna(self.cna(x, x.shape[1], 3, s, groups=x.shape[1], act=self.r.random() < 0.5), c, 1)
  def b_inception(self, x, c, s):
    n = self.r.randint(2, 4)
    widths = [max(8, c//n//8*8)]*n
    outs = []
    for i, w in enumerate(widths):
      kind = self.pick("1x1", "3x3", "5x5", "1xk", "pool") if i else "1x1"
      if kind == "1x1": y = self.cna(x, w, 1, s)
      elif kind == "3x3": y = self.cna(self.cna(x, w, 1), w, 3, s)
      elif kind == "5x5": y = self.cna(self.cna(x, max(8, w//2), 1), w, 5, s)
      elif kind == "1xk" and x.ndim == 4:
        k = self.pick(3, 5, 7)
        y = self.cna(self.cna(self.cna(x, w, 1), w, (1, k)), w, (k, 1), s)
      else: y = self.cna(self.pool(x, self.pick("max", "avg"), s, 3), w, 1)
      outs.append(y)
    return outs[0].cat(*outs[1:], dim=1)
  def b_dense(self, x, c, s):
    if s > 1: x = self.pool(self.pre(x, self.width(x.shape[1], 0.5), 1), "avg", s)
    growth = self.pick(16, 24, 32, 48)
    for _ in range(self.r.randint(2, 4)):
      x = x.cat(self.pre(self.pre(x, 4*growth, 1), growth, 3) if self.r.random() < 0.5 else self.pre(x, growth, 3), dim=1)
    return x
  def b_convnext(self, x, c, s):
    if s > 1 or x.shape[1] != c: x = self.conv(self.norm(x, "layer"), c, s, s, bias=True)
    y = self.norm(self.conv(x, c, 7, groups=c, bias=True), "layer")
    y = self.conv(self.act(self.conv(y, 4*c, 1, bias=True), "gelu"), c, 1, bias=True)
    return x + y * self.P(c, *[1]*(x.ndim-2))
  def b_shuffle(self, x, c, s):
    # split the channels, work on one half, join and shuffle
    if s > 1 or x.shape[1] != c or c % 4: return self.b_sepconv(x, c, s)
    a, b = x.chunk(2, dim=1)
    b = self.cna(self.cna(self.cna(b, c//2, 1), c//2, 3, groups=c//2, act=False), c//2, 1)
    y = a.cat(b, dim=1)
    return y.reshape(y.shape[0], 2, c//2, *y.shape[2:]).transpose(1, 2).reshape(y.shape)
  def b_aspp(self, x, c, s):
    if s > 1: x = self.pool(x, "max", s, 3)
    rates, w = (1, 2, 4, 6)[:self.r.randint(2, 4)], max(8, c//4//8*8)
    outs = [self.cna(x, w, 3, dilation=d) if d > 1 else self.cna(x, w, 1) for d in rates]
    return self.cna(outs[0].cat(*outs[1:], dim=1), c, 1)
  def b_attn(self, x, c, s):
    # self attention over the positions of a feature map
    x = self.shortcut(x, c, s)
    if prod(x.shape[2:]) > 1024: return self.b_basic(x, c, 1)
    y = self.norm(x, self.pick("group", "batch")).flatten(2).transpose(1, 2)
    y = self.attention(y, self.pick(1, 2, 4, 8) if c % 8 == 0 else 1)
    return x + y.transpose(1, 2).reshape(x.shape)
  CONV_BLOCKS = {"plain": 14, "basic": 16, "preact": 6, "bottleneck": 14, "mbconv": 14, "sepconv": 6, "inception": 6, "dense": 5, "convnext": 5,
                 "shuffle": 3, "aspp": 3, "attn": 6}
  def block(self, x:Tensor, c:int, s:int=1, names:list[str]|None=None) -> Tensor:
    names = [n for n in (names or list(self.CONV_BLOCKS)) if not (n == "convnext" and x.ndim != 4)]
    name = self.r.choices(names, [self.CONV_BLOCKS[n] for n in names])[0]
    self.desc.append(name)
    x = getattr(self, "b_"+name)(x, c, s)
    return x.contiguous() if self.r.random() < 0.04 else x

  def up(self, x:Tensor, c:int, size:tuple[int, ...]|None=None) -> Tensor:
    # twice the size, or the given size when the level it joins is an odd one
    kind, nd = self.pick("nearest", "nearest", "linear", "convt2", "convt4", "shuffle"), x.ndim-2
    if kind in ("nearest", "linear"): return self.cna(x.interpolate(size or tuple(d*2 for d in x.shape[2:]), mode=kind), c, 3)
    if kind == "shuffle" and nd == 2:
      y = self.conv(x, c*4, 3)
      y = y.reshape(y.shape[0], c, 2, 2, *y.shape[2:]).permute(0, 1, 4, 2, 5, 3).reshape(y.shape[0], c, y.shape[2]*2, y.shape[3]*2)
    else:
      k, p = (2, 0) if kind == "convt2" else (4, 1)
      y = self.act(self.norm(self.reg(nn.ConvTranspose2d(x.shape[1], c, (k,)*nd, 2, p, bias=self.norm_name == "none"))(x)))
    return y if size is None or y.shape[2:] == size else y.interpolate(size, mode="nearest")

  def image(self, nd:int) -> tuple[Tensor, int]:
    # an input batch and the width of the first stage. the batch size is picked to give a first activation of a realistic size
    r = self.r
    cin = r.choice([3]*8 + [1, 1, 4, 6, 9, 12]) if nd != 1 else r.choice([1, 2, 40, 64, 80, 128])
    if nd == 1: size: tuple[int, ...] = (r.choice([400, 1000, 1500, 3000, 8000, 16000]),)
    elif nd == 3: size = (r.choice([8, 16, 32, 64]), *[r.choice([32, 64, 96, 128])]*2)
    else:
      h = r.choice([28, 32, 32, 64, 64, 96, 112, 128, 160, 192, 224, 224, 224, 256, 256, 299, 320, 384, 416, 512, 640][10 if self.one else 0:])
      size = (h, h if r.random() < 0.8 else r.choice([h//2, h*3//2, h*2]))
    c = r.choice([16, 16, 24, 32, 32, 32, 40, 48, 64, 64, 64, 96, 128])
    bs = self.bs(c*prod(size)//4)
    if nd == 2 and cin == 3 and r.random() < 0.12:
      # bytes in channels-last order, as an image loader gives them
      x = E(bs, *size, cin, dtype=dtypes.uint8).permute(0, 3, 1, 2).cast(self.ft or dtypes.default_float) / 255
      if r.random() < 0.6: x = (x - self.P(1, cin, 1, 1)) / self.P(1, cin, 1, 1)
    else: x = E(bs, cin, *size, dtype=self.ft)
    self.desc.append(f"in{tuple(x.shape)}")
    return x, c
  def stem(self, x:Tensor, c:int) -> Tensor:
    kind = self.pick("3s1", "3s2", "3s2", "7s2", "7s2p", "patch4", "5s2", "3x3x2") if min(x.shape[2:]) >= 32 else "3s1"
    if kind == "patch4": return self.norm(self.conv(x, c, 4, 4, bias=True))
    if kind == "3x3x2": return self.cna(self.cna(x, max(8, c//2), 3, 2), c, 3)
    x = self.cna(x, c, int(kind[0]), int(kind[2]))
    return self.pool(x, "max", 2, 3) if kind.endswith("p") else x

  def conv_body(self, x:Tensor, c:int, n:int) -> Tensor:
    for _ in range(n):
      s = 2 if min(x.shape[2:]) >= 8 and self.r.random() < 0.4 else 1
      if s == 2: c = self.width(c, self.pick(1, 1.5, 2, 2))
      elif self.r.random() < 0.25: c = self.width(c, self.pick(0.5, 1.5, 2))
      if x.numel() * 6 * c // x.shape[1] > MAX_NUMEL * s**(x.ndim-2): c = x.shape[1]
      x = self.block(x, c, s)
    return x
  def conv_head(self, x:Tensor) -> tuple[Tensor, str]:
    r, B = self.r, x.shape[0]
    kind = r.choices(["classify", "dense", "detect", "embed", "image"], [50, 18, 8, 8, 16])[0] if x.ndim == 4 else r.choice(["classify", "dense"])
    self.desc.append(kind)
    if kind in ("classify", "embed"):
      if r.random() < 0.3: x = self.cna(x, self.width(x.shape[1], self.pick(2, 4)), 1)
      x = x.mean(tuple(range(2, x.ndim))) if r.random() < 0.8 or prod(x.shape[1:]) > 1 << 15 else x.flatten(1)
      if r.random() < 0.25: x = self.act(self.linear(self.drop(x, 0.5), self.pick(256, 512, 1024, 1280, 4096)))
      x = self.drop(x, 0.2)
      return self.linear(x, self.pick(10, 10, 100, 1000, 1000, 1000, 21, 80, 365) if kind == "classify" else self.pick(128, 256, 512)), kind
    if kind == "dense":
      return self.conv(self.cna(x, x.shape[1], 3) if r.random() < 0.5 else x, self.pick(1, 2, 3, 19, 21, 80, 150), 1, bias=True), kind
    if kind == "detect":
      a, ncls = self.pick(1, 3, 9), self.pick(1, 20, 80)
      return self.conv(x, a*(5+ncls), 1, bias=True).reshape(B, a, 5+ncls, *x.shape[2:]).permute(0, 1, 3, 4, 2), kind
    return self.conv(x, self.pick(1, 3, 3, 4), 3, bias=True), kind

  def features(self, nd:int) -> Tensor:
    # the activations somewhere in the middle of a network: the deeper, the smaller and the wider
    r = self.r
    sizes = {1: [50, 100, 375, 750, 1500], 2: [7, 7, 8, 10, 13, 14, 14, 16, 20, 26, 28, 28, 32, 40, 52, 56, 56, 64, 80, 112], 3: [4, 8, 16, 32]}
    hw = r.choice(sizes[nd])
    widths = (16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 240, 256, 320, 384, 480, 512, 640, 768, 960, 1024, 1280, 2048)
    c = r.choice([c for c in widths if 1 << 13 <= c*hw**nd <= 1 << 20])
    x = E(self.bs(c*hw**nd), c, *[hw]*nd, dtype=self.ft)
    self.desc.append(f"mid{tuple(x.shape)}")
    return x
  def convnet(self, nd:int=2) -> tuple[Tensor, str]:
    # a whole small network from the image on, or a stretch of a deep one
    n = self.r.randint(1, 4) if self.train else self.r.randint(2, 7)
    if self.r.random() < 0.5:
      x = self.features(nd)
      return self.conv_head(self.conv_body(x, x.shape[1], n))
    x, c = self.image(nd)
    return self.conv_head(self.conv_body(self.stem(x, c), c, n))
  def unet(self) -> tuple[Tensor, str]:
    # down, across and up again, joining each level of the way down to the same level on the way up
    r = self.r
    x, c = self.image(2)
    if r.random() < 0.5:
      self.cond = self.linear(self.act(self.linear(E(x.shape[0], self.pick(128, 256, 320), dtype=self.ft), 4*c)), 4*c)
    names = ["basic", "preact", "plain", "attn"]
    x, skips = self.cna(x, c, 3), []
    for _ in range(r.randint(1, 3)):
      if min(x.shape[2:]) < 8: break
      x = self.block(x, c, 1, names)
      skips.append(x)
      c = self.width(c, 2)
      x = self.pool(x, self.pick("max", "avg")) if r.random() < 0.5 else self.cna(x, c, 3, 2)
    x = self.block(x, c, 1, names)
    for skip in reversed(skips):
      x = self.up(x, skip.shape[1], skip.shape[2:])
      x = self.block(x.cat(skip, dim=1) if r.random() < 0.85 else x + skip, skip.shape[1], 1, names)
    self.desc.append("image")
    return self.conv(self.act(self.norm(x)), self.pick(1, 2, 3, 4, 21), self.pick(1, 3), bias=True), self.pick("image", "dense")
  def decoder(self) -> tuple[Tensor, str]:
    # an image made from a latent vector: a generator or the second half of an autoencoder
    r = self.r
    c, hw = r.choice([128, 256, 512]), r.choice([4, 7, 8])
    bs = self.bs(c*hw*hw*4, 17, 20)
    x = self.act(self.linear(E(bs, r.choice([32, 64, 100, 128, 256]), dtype=self.ft), c*hw*hw)).reshape(bs, c, hw, hw)
    for _ in range(r.randint(2, 4)):
      c = self.width(c, 0.5)
      if x.numel() * 4 > MAX_NUMEL: break
      x = self.up(x, c)
      if r.random() < 0.4: x = self.block(x, c, 1, ["basic", "plain", "preact"])
    self.desc.append("image")
    return getattr(self.conv(x, self.pick(1, 3), self.pick(3, 5, 7), bias=True), self.pick("tanh", "sigmoid"))(), "image"

  # *** sequence blocks ***

  def attention(self, x:Tensor, heads:int, mem:Tensor|None=None, causal:bool=False, bias:Tensor|None=None) -> Tensor:
    B, T, D = x.shape
    src = x if mem is None else mem
    if mem is None and self.r.random() < 0.3: q, k, v = self.linear(x, 3*D).chunk(3, dim=-1)
    else: q, k, v = self.linear(x, D, self.qkv_bias), self.linear(src, D, self.qkv_bias), self.linear(src, D, self.qkv_bias)
    q, k, v = [y.reshape(B, -1, heads, D//heads).transpose(1, 2) for y in (q, k, v)]
    if bias is None and self.r.random() < 0.7:
      o = q.scaled_dot_product_attention(k, v, is_causal=causal, dropout_p=0.1 if self.train and self.r.random() < 0.3 else 0.0)
    else:
      w = q @ k.transpose(-2, -1) * (D//heads)**-0.5
      if bias is not None: w = w + bias
      if causal: w = w.masked_fill(Tensor.ones(T, src.shape[1], dtype=dtypes.bool).triu(1), float("-inf"))
      o = self.drop(w.softmax(-1)) @ v
    return self.drop(self.linear(o.transpose(1, 2).reshape(B, T, D), D))
  def mlp(self, x:Tensor) -> Tensor:
    D = x.shape[-1]
    if self.mlp_kind == "swiglu":
      hid = int(D*self.mlp_ratio*2/3)//8*8
      return self.linear(self.linear(x, hid, False).silu() * self.linear(x, hid, False), D, False)
    return self.drop(self.linear(self.drop(self.act(self.linear(x, int(D*self.mlp_ratio)))), D))
  def tblock(self, x:Tensor, heads:int, mem:Tensor|None=None, causal:bool=False, bias:Tensor|None=None) -> Tensor:
    # self attention, attention over the other sequence if there is one, then the perceptron: each with its own norm and a way around it
    subs = [lambda y: self.attention(y, heads, causal=causal, bias=bias)]
    if mem is not None: subs.append(lambda y: self.attention(y, heads, mem))
    subs.append(self.mlp)
    for f in subs:
      if self.prenorm: x = x + (f(self.seqnorm(x)) * self.P(x.shape[-1]) if self.layerscale else f(self.seqnorm(x)))
      else: x = self.seqnorm(x + f(x))
    return x
  def seq_setup(self) -> tuple[int, int]:
    r = self.r
    self.snorm = r.choices(["layer", "rms", "none"], [70, 22, 8])[0]
    self.prenorm, self.qkv_bias, self.layerscale = r.random() < 0.65, r.random() < 0.7, r.random() < 0.1
    self.mlp_kind, self.mlp_ratio = r.choices(["mlp", "swiglu"], [80, 20])[0], r.choice([1, 2, 4, 4, 4])
    D = r.choice([64, 96, 128, 192, 256, 256, 320, 384, 512, 512, 768, 768, 1024])
    return D, r.choice([h for h in (1, 2, 3, 4, 6, 8, 12, 16) if D % h == 0 and D//h >= 16])
  def tokens(self, B:int, T:int, D:int, vocab:int) -> tuple[Tensor, nn.Embedding]:
    emb = self.reg(nn.Embedding(vocab, D))
    x = emb(E(B, T, dtype=dtypes.int32))
    if self.r.random() < 0.3: x = x * D**0.5
    return x, emb
  def transformer(self) -> tuple[Tensor, str]:
    r = self.r
    D, heads = self.seq_setup()
    kind = r.choices(["lm", "encoder", "vit", "seq2seq", "features"], [28, 22, 26, 12, 12])[0]
    self.desc += [kind, f"D{D}h{heads}", self.snorm, self.mlp_kind]
    T = r.choice([16, 32, 64, 77, 128, 128, 196, 256, 384, 512, 1024][4 if self.one else 0:])
    B = self.bs(T*D, 16, 22)
    vocab, causal, mem, emb = r.choice([1000, 8192, 30522, 32000, 50257]), kind == "lm" or (kind == "features" and r.random() < 0.3), None, None
    if kind == "vit":
      p, hw = r.choice([4, 8, 14, 16, 16, 32]), r.choice([32, 64, 96, 224, 224, 256, 384])
      hw = max(hw, 2*p)//p*p
      T = (hw//p)**2
      B = self.bs(T*D, 16, 22)
      x = self.conv(E(B, r.choice([1, 3, 3, 3]), hw, hw, dtype=self.ft), D, p, p, bias=True).flatten(2).transpose(1, 2)
      if r.random() < 0.7: x, T = self.P(1, 1, D).expand(B, 1, D).cat(x, dim=1), T+1
    elif kind == "features": x = self.linear(E(B, T, r.choice([13, 40, 80, 128, 512]), dtype=self.ft), D)
    else: x, emb = self.tokens(B, T, D, vocab)
    if kind == "seq2seq":
      S = r.choice([16, 32, 64, 128, 256])
      mem = self.tokens(B, S, D, vocab)[0] if r.random() < 0.6 else self.linear(E(B, S, r.choice([80, 512, 768]), dtype=self.ft), D)
      if r.random() < 0.8: mem = mem + self.P(S, D)
      mem = self.seqnorm(self.tblock(mem, heads))
      causal = True
    if r.random() < 0.75: x = x + self.P(T, D)
    if kind in ("lm", "encoder") and r.random() < 0.2: x = x + self.reg(nn.Embedding(2, D))(E(B, T, dtype=dtypes.int32))
    if not self.prenorm or r.random() < 0.3: x = self.seqnorm(x)
    x = self.drop(x)
    bias = self.P(1, heads, T, T) if r.random() < 0.12 else None
    if kind in ("encoder", "seq2seq", "features") and not causal and r.random() < 0.4:
      # a batch padded to one length: positions past the end of a sequence get a large negative number added before the softmax
      pad = ((1 - E(B, T, dtype=dtypes.int32).cast(self.ft or dtypes.default_float)) * -10000.0).reshape(B, 1, 1, T)
      bias = pad if bias is None else bias + pad
    for _ in range(r.randint(1, 3)): x = self.tblock(x, heads, mem, causal, bias)
    if self.prenorm: x = self.seqnorm(x)
    if kind in ("lm", "seq2seq") or (kind == "encoder" and r.random() < 0.4):
      if not self.train and r.random() < 0.5: x = x[:, -1:, :]   # one step of generation only needs the last position
      if kind == "encoder": x = self.seqnorm(self.act(self.linear(x, D), "gelu"))
      return (x @ emb.weight.T if emb is not None and r.random() < 0.35 else self.linear(x, vocab, r.random() < 0.3)), "tokens"
    if kind == "features" and r.random() < 0.5: return self.linear(x, r.choice([1, 29, 80, 1000])), "tokens" if r.random() < 0.5 else "image"
    x = x[:, 0] if r.random() < 0.5 else x.mean(1)
    if r.random() < 0.3: x = self.linear(x, D).tanh()
    return self.linear(self.drop(x), r.choice([2, 2, 10, 100, 1000, 1000])), "classify"

  def conv1d(self) -> tuple[Tensor, str]:
    # audio or signal: convolutions over time, then either a classifier or a few transformer blocks
    x, c = self.image(1)
    x = self.cna(x, c, self.pick(3, 5, 7, 11), self.pick(1, 2, 4))
    x = self.conv_body(x, c, self.r.randint(1, 3))
    if self.r.random() < 0.5 or x.shape[2] > 512: return self.conv_head(x)
    _, heads = self.seq_setup()
    x = x.transpose(1, 2)
    heads = max(h for h in (1, 2, 4, 8) if x.shape[-1] % h == 0 and h <= heads)
    x = x + self.P(*x.shape[1:])
    for _ in range(self.r.randint(1, 2)): x = self.tblock(x, heads)
    return self.linear(self.seqnorm(x), self.pick(29, 80, 512, 1000)), "tokens"

  def recurrent(self) -> tuple[Tensor, str]:
    r = self.r
    H, T, F = r.choice([128, 256, 256, 512, 512, 1024, 2048]), r.randint(2, 5), r.choice([40, 80, 128, 256, 512])
    self.one = False
    B = self.bs(H*8, 16, 20)
    cell, layers = r.choice(["lstm", "lstm", "gru", "rnn"]), r.randint(1, 2)
    self.desc += [cell, f"H{H}T{T}"]
    if r.random() < 0.5: xs = [y.squeeze(1) for y in self.tokens(B, T, F, r.choice([1000, 10000, 32000]))[0].split(1, dim=1)]
    else: xs = [E(B, F, dtype=self.ft) for _ in range(T)]
    for _ in range(layers):
      F, outs = xs[0].shape[-1], []
      if cell == "lstm":
        lstm, hc = self.reg(nn.LSTMCell(F, H)), None
        for x in xs: outs.append((hc:=lstm(x, hc))[0])
      else:
        gates = 3 if cell == "gru" else 1
        wi, wh, h = self.reg(nn.Linear(F, H*gates)), self.reg(nn.Linear(H, H*gates)), E(B, H, dtype=self.ft)
        for x in xs:
          if cell == "rnn": h = (wi(x) + wh(h)).tanh()
          else:
            (ir, iz, i_n), (hr, hz, hn) = wi(x).chunk(3, dim=-1), wh(h).chunk(3, dim=-1)
            z = (iz + hz).sigmoid()
            h = (1 - z) * (i_n + (ir + hr).sigmoid() * hn).tanh() + z * h
          outs.append(h)
      xs = [self.drop(o) for o in outs]
    if r.random() < 0.5: return self.linear(xs[-1], r.choice([2, 10, 1000])), "classify"
    return self.linear(xs[0].stack(*xs[1:], dim=1), r.choice([29, 1000, 10000])), "tokens"

  def tabular(self) -> tuple[Tensor, str]:
    # a multilayer perceptron on flat features, optionally with embedding tables for categorical ones
    r = self.r
    B = r.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    feats = [E(B, r.choice([13, 64, 256, 784, 1024, 3072]), dtype=self.ft)]
    if r.random() < 0.6:
      d, embs = r.choice([16, 32, 64, 128]), []
      for _ in range(r.randint(2, 8)):
        table = self.reg(nn.Embedding(r.choice([100, 1000, 10000, 100000]), d))
        embs.append(table(E(B, dtype=dtypes.int32)) if r.random() < 0.7 else table(E(B, r.choice([4, 16, 50]), dtype=dtypes.int32)).sum(1))
      feats = [self.act(self.linear(feats[0], d))] + embs
      if r.random() < 0.5:
        t = feats[0].stack(*feats[1:], dim=1)
        feats.append((t @ t.transpose(1, 2)).flatten(1))
    x = feats[0].cat(*feats[1:], dim=1) if len(feats) > 1 else feats[0]
    bn = r.random() < 0.3
    for _ in range(r.randint(1, 4)):
      x = self.linear(x, r.choice([128, 256, 512, 512, 1024, 2048, 4096]))
      if bn: x = self.reg(nn.BatchNorm(x.shape[1]))(x)
      x = self.drop(self.act(x), 0.5)
    return self.linear(x, r.choice([1, 1, 2, 10, 10, 100, 1000])), "classify"

  # *** running it ***

  def forward(self) -> tuple[Tensor, str]:
    if self.family == "conv2d": return self.convnet(2)
    if self.family == "conv3d": return self.convnet(3)
    return getattr(self, self.family)()
  def loss(self, out:Tensor, kind:str) -> Tensor:
    r = self.r
    if kind == "embed":
      # contrastive: every embedding against a second batch of them
      a, b = out.normalize(dim=-1), E(*out.shape, dtype=self.ft).normalize(dim=-1)
      return (a @ b.T * self.P(1).exp()).sparse_categorical_crossentropy(Tensor.arange(out.shape[0]))
    if kind == "classify" and out.shape[-1] > 1 and r.random() < 0.85:
      return out.sparse_categorical_crossentropy(E(out.shape[0], dtype=dtypes.int32), label_smoothing=r.choice([0.0, 0.0, 0.1]))
    if kind == "tokens":
      tgt = E(prod(out.shape[:-1]), dtype=dtypes.int32)
      return out.reshape(-1, out.shape[-1]).sparse_categorical_crossentropy(tgt, ignore_index=r.choice([-1, 0]))
    if kind == "dense" and out.shape[1] > 1 and r.random() < 0.7:
      return out.transpose(1, -1).reshape(-1, out.shape[1]).sparse_categorical_crossentropy(E(out.numel()//out.shape[1], dtype=dtypes.int32))
    tgt = E(*out.shape, dtype=self.ft)
    which = r.choice(["mse", "mse", "l1", "bce", "huber"]) if kind != "detect" else "bce"
    if which == "bce": return out.binary_crossentropy_logits(tgt)
    d = out - tgt
    if which == "huber": return (d.abs() < 1).where(0.5*d.square(), d.abs()-0.5).mean()
    return d.square().mean() if which == "mse" else d.abs().mean()
  def optimizer(self):
    r, params = self.r, get_parameters(self.mods)
    lr, wd = r.choice([1e-1, 1e-2, 1e-3, 3e-4, 1e-4]), r.choice([0.0, 0.0, 1e-4, 5e-4, 1e-2])
    kind = r.choices(["sgd", "momentum", "adam", "adamw", "lamb", "lars"], [12, 30, 22, 26, 5, 5])[0]
    self.desc.append(kind)
    if kind == "sgd": return nn.optim.SGD(params, lr, weight_decay=wd)
    if kind == "momentum": return nn.optim.SGD(params, lr, momentum=0.9, weight_decay=wd, nesterov=r.random() < 0.3)
    if kind == "adam": return nn.optim.Adam(params, lr)
    if kind == "adamw": return nn.optim.AdamW(params, lr, weight_decay=wd)
    if kind == "lamb": return nn.optim.LAMB(params, lr, weight_decay=wd)
    return nn.optim.LARS(params, lr, momentum=0.9, weight_decay=wd)
  def run(self):
    if self.train:
      with Context(TRAINING=1):
        loss = self.loss(*self.forward())
        opt = self.optimizer()
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:
      out, kind = self.forward()
      post = self.r.random()
      if kind in ("classify", "tokens") and post < 0.25: out = out.softmax(-1)
      elif kind in ("classify", "tokens") and post < 0.4: out = out.argmax(-1)
      elif kind in ("detect", "image") and post < 0.3: out = out.sigmoid()
      elif kind == "dense" and post < 0.3: out = out.argmax(1)
      out.realize()

if __name__ == "__main__":
  for seed in sys.argv[1:]:
    p = Program(int(seed))
    try: p.run()
    finally: print(f"*** {seed} {' '.join(p.desc)}", flush=True)
