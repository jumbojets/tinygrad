#!/usr/bin/env python3
# trains the policy of policy_search.py on what searches measured
#   collect: POLICY_COLLECT=<dir> BEAM=4 python -m extra.optimization.policy_search extra/optimization/kernel_generator.py <seed> ...
#            any tinygrad program will do in place of the generator. every kernel it compiles is searched, and each kernel the search expanded is
#            kept in <dir> with its legal actions and the speedups of the ones that were timed (policy_search.collect_search). a kernel that is in
#            <dir> is not searched again, so a collection can be stopped and started. on more machines: other seeds on each, then copy the
#            directories together
#   train:   python extra/optimization/policy_train.py <dir> [<dir> ...] --out policy.safetensors
#   search:  POLICY=policy.safetensors BEAM=200 python -m extra.optimization.policy_search <program.py> [args]
# a training row is one expanded kernel. the loss asks the policy to put its weight on the actions that led to fast kernels. kernels are padded to
# a few sizes, so the JIT compiles one training step per size.
import argparse, glob, json, math, os, time
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Context
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import safe_save
import tinygrad.codegen.opt.search as search
from extra.optimization.policy_search import PolicyNet, relations, SIZES, A_MAX, VOCAB, OPS, DTYPES, N_ARG, KINDS, AMTS, D_MAX, GLOBAL_NAMES, \
  N_NODE_FLOAT, N_ACT_FLOAT

BATCH = dict(zip(SIZES, (48, 32, 24, 16, 8, 8)))  # kernels in a training step, by the size they are padded to
BETA, TOP = 2.0, 3

class Data:
  # every expanded kernel of the files in the directories, as flat arrays with offsets. one kernel file in twenty is held out to score the policy
  def __init__(self, dirs:list[str]):
    files = sorted(fn for d in dirs for fn in glob.glob(os.path.join(d, "*.npz")))
    parts = [dict(np.load(fn)) for fn in files]
    assert len(parts) and all(json.loads(str(p["info"]))["vocab"] == VOCAB for p in parts), "no data, or data collected with another tinygrad"
    self.d = d = {k: np.concatenate([p[k] for p in parts]) for k in ("ni", "nf", "ed", "glob", "ai", "af", "y")}
    for k in ("noff", "eoff", "aoff"): d[k] = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(p[k]) for p in parts]))])
    self.n, self.na = np.diff(d["noff"]), np.diff(d["aoff"])
    self.size = np.array(SIZES)[np.searchsorted(SIZES, self.n)]
    held = np.repeat([int(os.path.basename(fn)[:8], 16) % 20 == 0 for fn in files], [len(p["glob"]) for p in parts])
    # kernels with two or more timed actions have something to rank
    timed = np.concatenate([[0], np.cumsum(np.isfinite(d["y"]))])
    ok = (timed[d["aoff"][1:]]-timed[d["aoff"][:-1]] >= 2) & (self.na <= A_MAX)
    self.train, self.held = np.flatnonzero(ok & ~held), np.flatnonzero(ok & held)
    assert len(self.train), "no kernel with two timed actions to train on"
    self.gm, self.gs = d["glob"][~held].mean(0), d["glob"][~held].std(0)+1e-6

  def batches(self, ids:np.ndarray, rng:np.random.Generator|None=None) -> list[tuple[int, np.ndarray]]:
    ret = []
    for size, bs in BATCH.items():
      mine = ids[self.size[ids] == size] if rng is None else rng.permutation(ids[self.size[ids] == size])
      ret += [(size, mine[i:i+bs]) for i in range(0, len(mine), bs)]
    if rng is not None: rng.shuffle(ret)
    return ret

  def batch(self, size:int, ids:np.ndarray) -> list[np.ndarray]:
    # the inputs of PolicyNet, then the speedups, which actions have one, and a weight per kernel: a batch that is not full repeats its first
    # kernel with weight 0
    d, B, N = self.d, BATCH[size], size
    ni, nf, key = np.zeros((B, N, 5), np.int32), np.zeros((B, N, N_NODE_FLOAT), np.float32), np.zeros((B, N+1), bool)
    slot, glob = np.zeros((B, N, N), np.int32), np.zeros((B, len(GLOBAL_NAMES)), np.float32)
    ai, af = np.zeros((B, A_MAX, 5), np.int32), np.zeros((B, A_MAX, N_ACT_FLOAT), np.float32)
    y, tm, wt = np.zeros((B, A_MAX), np.float32), np.zeros((B, A_MAX), bool), np.zeros(B, np.float32)
    for b in range(B):
      i = ids[b] if b < len(ids) else ids[0]
      n, a, e = self.n[i], self.na[i], d["ed"][d["eoff"][i]:d["eoff"][i+1]].astype(np.int64)
      ni[b, :n], nf[b, :n], key[b, :n+1] = d["ni"][d["noff"][i]:d["noff"][i+1]], d["nf"][d["noff"][i]:d["noff"][i+1]], True
      slot[b, e[:, 1], e[:, 0]] = e[:, 2]+1
      glob[b], yy = (d["glob"][i]-self.gm)/self.gs, d["y"][d["aoff"][i]:d["aoff"][i+1]]
      ai[b, :a], af[b, :a], tm[b, :a] = d["ai"][d["aoff"][i]:d["aoff"][i+1]], d["af"][d["aoff"][i]:d["aoff"][i+1]], np.isfinite(yy)
      y[b, :a], wt[b] = np.where(tm[b, :a], yy, 0), float(b < len(ids))
    return [ni, nf, key, np.pad(slot, ((0, 0), (1, 0), (1, 0))), relations(slot), glob, ai, af, y, tm, wt]

def init_weights(d:int, layers:int, heads:int) -> dict[str, Tensor]:
  # embeddings normal, linear layers uniform within 1/sqrt(inputs), the graph biases of the attention zero
  w: dict[str, Tensor] = {"cls": Tensor.randn(d)}
  def uni(i:int, *shape:int) -> Tensor: return Tensor.uniform(*shape, low=-i**-0.5, high=i**-0.5)
  def lin(n:str, i:int, o:int): w[n+".weight"], w[n+".bias"] = uni(i, o, i), uni(i, o)
  def norm(n:str): w[n+".weight"], w[n+".bias"] = Tensor.ones(d).contiguous(), Tensor.zeros(d).contiguous()
  for n, rows in (("op", len(OPS)), ("dt", len(DTYPES)), ("arg", N_ARG), ("axpos", 33), ("nsrc", 16), ("kind", len(KINDS)), ("amt", len(AMTS)+1),
                  ("act", len(search.actions)+2)): w[n+".weight"] = Tensor.randn(rows, d)
  for n, rows in (("rel", 2*D_MAX+7), ("slot_f", 9), ("slot_r", 9)): w[n+".weight"] = Tensor.zeros(rows, heads).contiguous()
  for n, i, o in (("nf.0", N_NODE_FLOAT, d), ("nf.2", d, d), ("glob.0", len(GLOBAL_NAMES), d), ("glob.2", d, d), ("af", N_ACT_FLOAT, d),
                  ("policy.0", 4*d, d), ("policy.2", d, 1)): lin(n, i, o)
  for b in range(layers):
    for n, i, o in (("qkv", d, 3*d), ("proj", d, d), ("mlp.0", d, 4*d), ("mlp.3", 4*d, d)): lin(f"blocks.{b}.{n}", i, o)
    for n in ("n1", "n2"): norm(f"blocks.{b}.{n}")
  norm("norm")
  return {k: v.realize() for k, v in w.items()}

def lost(dat:Data, fwd:dict[int, TinyJit]) -> float:
  # percent of kernel speed lost on the held-out kernels by timing only the policy's top actions in place of all the timed ones
  out = []
  for size, ids in dat.batches(dat.held):
    arr = dat.batch(size, ids)
    logits = fwd[size](*[Tensor(x) for x in arr[:8]]).numpy()
    for b in range(len(ids)):
      t = np.flatnonzero(arr[9][b])
      out.append(arr[8][b, t].max()-arr[8][b, t[np.argsort(-logits[b, t], kind="stable")[:TOP]]].max())
  return float(100*(2**np.mean(out)-1))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dirs", nargs="+", help="directories written by POLICY_COLLECT")
  parser.add_argument("--out", default="policy.safetensors")
  parser.add_argument("--epochs", type=int, default=12)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--wd", type=float, default=0.05)
  parser.add_argument("--warmup", type=int, default=500)
  parser.add_argument("--drop", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--d", type=int, default=256)
  parser.add_argument("--layers", type=int, default=6)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--steps", type=int, default=0, help="stop after this many steps, for a quick look")
  args = parser.parse_args()
  Tensor.manual_seed(args.seed)
  rng, dat = np.random.default_rng(args.seed), Data(args.dirs)
  w = init_weights(args.d, args.layers, args.heads)
  net, params = PolicyNet(w, args.drop), list(w.values())
  # weight decay on the matrices only
  opt = OptimizerGroup(AdamW([p for p in params if p.ndim >= 2], args.lr, b1=0.9, b2=0.98, weight_decay=args.wd),
                       AdamW([p for p in params if p.ndim < 2], args.lr, b1=0.9, b2=0.98, weight_decay=0.0))
  per_epoch = len(dat.batches(dat.train))
  total = args.steps or args.epochs*per_epoch
  print(f"{sum(p.numel() for p in params):,} parameters, {len(dat.train):,} expanded kernels to train on, {len(dat.held):,} held out, "
        f"{per_epoch} steps an epoch")

  @Context(TRAINING=1)
  def step(ni, nf, key, slot, rel, glob, ai, af, y, tm, wt) -> Tensor:
    logits = net(ni, nf, key, slot, rel, glob, ai, af)
    # the timed actions of a kernel as a distribution, sharper for bigger speedups, against the policy's
    target, lp = tm.where(y*BETA, -1e9).softmax(-1), tm.where(logits, -1e9).log_softmax(-1)
    loss = (-(target*tm.where(lp, 0)).sum(-1)*wt).sum()/wt.sum().maximum(1)
    opt.zero_grad()
    loss.backward()
    # gradients clipped to norm 1. contiguous computes the scale once: without it every parameter update computes the norm of all gradients again
    clip = (1/(Tensor.stack(*[p.grad.square().sum() for p in params]).sum().sqrt()+1e-6)).minimum(1).contiguous()
    for p in params: p.grad = p.grad*clip
    return loss.realize(*opt.schedule_step())
  steps, fwd = {n: TinyJit(step) for n in SIZES}, {n: TinyJit(lambda *x: net(*x).realize()) for n in SIZES}

  s, st, losses = 0, time.perf_counter(), []
  for epoch in range(args.epochs):
    for size, ids in dat.batches(dat.train, rng):
      lr = args.lr*min(1.0, (s+1)/args.warmup)*0.5*(1+math.cos(math.pi*min(1.0, s/max(1, total))))
      for o in opt.optimizers: o.lr.assign(Tensor([lr], device=o.lr.device, dtype=o.lr.dtype)).realize()
      losses.append(steps[size](*[Tensor(x) for x in dat.batch(size, ids)]).item())
      if (s:=s+1) % 100 == 0: print(f"step {s} of {total}  loss {np.mean(losses[-100:]):.3f}  {time.perf_counter()-st:.0f}s", flush=True)
      if s >= total: break
    # the file policy_search.py loads: the weights, and the mean and spread of the kernel-level inputs they were trained with
    safe_save({**w, "glob_mean": Tensor(dat.gm), "glob_std": Tensor(dat.gs)}, args.out)
    print(f"epoch {epoch+1}  loss {np.mean(losses[-per_epoch:]):.3f}  " +
          (f"speed lost at top {TOP} on held-out kernels {lost(dat, fwd):.2f}%  " if len(dat.held) else "") + f"-> {args.out}", flush=True)
    if s >= total: break
