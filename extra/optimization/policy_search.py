#!/usr/bin/env python3
# kernel search with a learned policy: a small transformer over the UOps of a kernel that scores each of its legal actions (search.actions)
# by how likely the action is to lead to a fast kernel. two searches that take the place of beam_search and time real kernels as it does:
#   policy_mcts_search: MCTS with the policy as the prior over each kernel's actions (PUCT), until `amt` kernels are timed
#   collect_search:     BEAM as beam_search does it, keeping every kernel it expands with the speedups of the actions it timed: the training data
# the policy reads the kernel and nothing the search has measured, so a kernel always gets the same scores.
# this file run as a program runs another program with one of the two in place of beam_search, so BEAM is the search's `amt`:
#   collect: POLICY_COLLECT=<dir> BEAM=4 python -m extra.optimization.policy_search extra/optimization/kernel_generator.py <seed> ...
#   train:   python extra/optimization/policy_train.py <dir> --out policy.safetensors
#   search:  POLICY=policy.safetensors BEAM=200 python -m extra.optimization.policy_search <program.py> [args]
import os, sys, json, math, queue, runpy, hashlib, functools, collections
from typing import Callable
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import getenv, Context, CACHELEVEL, IGNORE_BEAM_CACHE, diskcache_get, diskcache_put
from tinygrad.nn.state import safe_load
from tinygrad.uop.ops import UOp, Ops, AxisType, sym_infer
from tinygrad.engine.worker import get_worker_pool
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
import tinygrad.codegen.opt.search as search
from tinygrad.codegen.opt.search import beam_search

# *** a kernel and its actions as inputs of the policy ***

MAX_NODES, MAX_AXES, N_NODE_FLOAT, N_ACT_FLOAT = 256, 10, 18, 2
OPS = [o.name for o in Ops]
DTYPES = ["other", "dtypes.void", "dtypes.weakint", "dtypes.weakfloat"] + [str(x) for x in dtypes.all]
AXIS_TYPES = [a.name for a in AxisType]
REDUCE_OPS = ["ADD", "MUL", "MAX"]
AMTS = [0, 2, 3, 4, 5, 7, 8, 13, 16, 28, 29, 32, 49, 64, 256]
MODS = (2, 3, 4, 5, 7, 8, 16)
# arg category of a node: 0 none, RANGE axis type, REDUCE op, CONST kind, PARAM slot
ARG_RANGE, ARG_REDUCE, ARG_CONST, ARG_PARAM, N_ARG = 1, 1+len(AXIS_TYPES), 5+len(AXIS_TYPES), 9+len(AXIS_TYPES), 9+len(AXIS_TYPES)+16
KINDS = ["upcast", "local", "local_top", "tc", "swap", "padto"]  # action kinds
AXIS_GROUPS = ("global", "local", "warp", "upcast", "unroll", "loop", "reduce_loop")
GLOBAL_NAMES = ["depth", "log2_root_tm", "log2_ops", "log2_mem", "n_nodes", "n_ranges", "n_index", "n_params", "n_reduce", "has_wmma",
                "log2_iters", "log2_max_buf", "log2_out_buf", "out_itemsize", "n_float_alu", "n_mul", "n_add", "n_where", "n_cmp", "n_transc",
                "n_cast", "n_divmod", "last_kind", "last_amt", "last_axis", "log2_local_threads", "log2_reduce_iters"] + \
               [f"opts_{k}" for k in KINDS] + [f"{p}_{t}" for t in AXIS_GROUPS for p in ("n", "log2")] + \
               [f"ax{i}_{p}" for i in range(MAX_AXES) for p in ("type", "log2", "reduce")]

# the numbers the inputs above are coded with change when tinygrad's ops, dtypes or actions do: collected data carries this id of the coding
VOCAB = hashlib.md5(repr((OPS, DTYPES, AXIS_TYPES, KINDS, AMTS, GLOBAL_NAMES, search.actions)).encode()).hexdigest()[:8]

def log2p(x) -> float: return math.log2(abs(x)+1)
def value_feats(v) -> list[float]:
  # a number as features: sign, magnitude, divisibility
  if not isinstance(v, int):
    return [1.0, float(v < 0), max(-1.0, min(1.0, math.log2(abs(v)+1e-9)/32)) if math.isfinite(v) else 0.0] + [0.0]*(len(MODS)+1) + [1.0]
  return [1.0, float(v < 0), log2p(v)/32] + [float(v % m == 0) for m in MODS] + [float(v > 0 and v & (v-1) == 0), 0.0]

def kind_of(o:Opt) -> int:
  if o.op is OptOps.SPLIT: return 0 if o.arg[1] is AxisType.UPCAST else 2 if len(o.arg) == 3 and o.arg[2] else 1
  return {OptOps.TC: 3, OptOps.SWAP: 4, OptOps.PADTO: 5}[o.op]

def axis_group(t:AxisType, red:bool) -> int:
  if t is AxisType.GLOBAL: return 0
  if t is AxisType.LOCAL: return 1
  if t is AxisType.WARP: return 2
  if t is AxisType.UPCAST: return 4 if red else 3
  return 6 if red else 5

def featurize(k:Scheduler, root_tm:float, ops:int, mem:int) -> dict|None:
  # the kernel after its opts as node tokens, "i reads j" edges and a vector of hand features. root_tm, ops and mem are the time and the
  # estimates of the unoptimized kernel. None if the kernel has too many UOps
  topo = list(k.ast.toposort())
  if len(topo) > MAX_NODES: return None
  idx = {u:i for i,u in enumerate(topo)}
  rngs, red = k.rngs, k.reduce_ranges
  rng_pos = {r:i for i,r in enumerate(rngs)}
  ni, nf, edges, depth = np.zeros((len(topo), 5), np.int16), np.zeros((len(topo), N_NODE_FLOAT), np.float32), [], [0]*len(topo)
  cnt: collections.Counter = collections.Counter()
  bufs: list[int] = []
  for i,u in enumerate(topo):
    cnt[u.op] += 1
    arg, val = 0, None
    if u.op is Ops.RANGE: arg, val = ARG_RANGE+AXIS_TYPES.index(u.axis_type.name), int(u.vmax)+1
    elif u.op is Ops.REDUCE: arg = ARG_REDUCE + (REDUCE_OPS.index(u.arg[0].name) if u.arg[0].name in REDUCE_OPS else 3)
    elif u.op is Ops.CONST:
      if isinstance(u.arg, bool): arg, val = ARG_CONST+2, int(u.arg)
      elif isinstance(u.arg, int): arg, val = ARG_CONST, u.arg
      else:
        try: arg, val = ARG_CONST+1, float(u.arg)
        except (TypeError, ValueError): arg = ARG_CONST+3
    elif u.op is Ops.PARAM:
      arg, val = ARG_PARAM+min(max(u.arg.slot, 0), 15), int(u.max_numel())
      bufs.append(val*u.dtype.itemsize)
    dt = str(u.dtype)
    ni[i] = (OPS.index(u.op.name), DTYPES.index(dt) if dt in DTYPES else 0, arg, rng_pos[u]+1 if u in rng_pos else 0, min(len(u.src), 15))
    for j,s in enumerate(u.src):
      edges.append((idx[s], i, min(j, 7)))
      depth[i] = max(depth[i], depth[idx[s]]+1)
    r = [x for x in u.ranges if x.dtype is not dtypes.void]
    nf[i, :12] = value_feats(val) if val is not None else 0.0
    nf[i, 12:] = (float(u in red), len(r)/8, sum(math.log2(int(x.vmax)+1) for x in r)/32, u.dtype.itemsize/8, depth[i]/64, float(u in rng_pos))

  g = dict.fromkeys(GLOBAL_NAMES, 0.0)
  opts = k.applied_opts
  sizes, types = [int(r.vmax)+1 for r in rngs], [r.axis_type for r in rngs]
  out_buf = next((u.src[0].src[0] for u in topo if u.op is Ops.STORE and u.src[0].op is Ops.INDEX), None)
  while out_buf is not None and out_buf.op is not Ops.PARAM and len(out_buf.src): out_buf = out_buf.src[0]
  g.update(depth=len(opts), log2_root_tm=math.log2(root_tm), log2_ops=log2p(ops), log2_mem=log2p(mem), n_nodes=len(topo), n_ranges=cnt[Ops.RANGE],
           n_index=cnt[Ops.INDEX], n_params=cnt[Ops.PARAM], n_reduce=cnt[Ops.REDUCE], has_wmma=float(cnt[Ops.WMMA] > 0),
           log2_iters=sum(math.log2(s) for s in sizes), log2_max_buf=log2p(max(bufs, default=0)),
           log2_out_buf=log2p(int(out_buf.max_numel())*out_buf.dtype.itemsize) if out_buf is not None and out_buf.op is Ops.PARAM else 0.0,
           out_itemsize=out_buf.dtype.itemsize if out_buf is not None else 0,
           n_float_alu=sum(1 for u in topo if dtypes.is_float(u.dtype) and u.op not in (Ops.CONST, Ops.PARAM, Ops.INDEX, Ops.REDUCE, Ops.CAST)),
           n_mul=cnt[Ops.MUL], n_add=cnt[Ops.ADD], n_where=cnt[Ops.WHERE], n_cmp=cnt[Ops.CMPLT]+cnt[Ops.CMPEQ]+cnt[Ops.CMPNE],
           n_transc=cnt[Ops.EXP2]+cnt[Ops.LOG2]+cnt[Ops.SIN]+cnt[Ops.SQRT]+cnt[Ops.RECIP], n_cast=cnt[Ops.CAST]+cnt[Ops.BITCAST],
           n_divmod=cnt[Ops.DIV]+cnt[Ops.MOD], last_kind=-1.0)
  for o in opts: g[f"opts_{KINDS[kind_of(o)]}"] += 1
  if len(opts):
    o = opts[-1]
    g.update(last_kind=kind_of(o), last_axis=o.axis if o.axis is not None else -1, last_amt=o.arg[0] if o.op is OptOps.SPLIT else 0)
  for i,(s,t,r) in enumerate(zip(sizes, types, rngs)):
    grp = AXIS_GROUPS[axis_group(t, r in red)]
    g[f"n_{grp}"] += 1
    g[f"log2_{grp}"] += math.log2(s)
    if i < MAX_AXES: g[f"ax{i}_type"], g[f"ax{i}_log2"], g[f"ax{i}_reduce"] = 1+axis_group(t, r in red), math.log2(s), float(r in red)
  g.update(log2_local_threads=g["log2_local"]+g["log2_warp"], log2_reduce_iters=g["log2_unroll"]+g["log2_reduce_loop"])
  # the training data stores the node floats as float16
  return {"ni": ni, "nf": nf.astype(np.float16), "edges": np.array(edges, np.int16).reshape(-1, 3),
          "g": np.array([g[n] for n in GLOBAL_NAMES], np.float32), "rng_nodes": [idx[r] for r in rngs], "sink": len(topo)-1, "sizes": sizes}

def action_feats(i:int, st:dict) -> tuple[list[int], list[float]]:
  # an action as (index into search.actions + 1, kind, node of its axis, node of its second axis, amount category), (log2 amount, log2 axis size left)
  a = search.actions[i-1]
  kind, node, node2, amt_cat, amt, left = kind_of(a), st["sink"], st["sink"], 0, 0.0, 0.0
  if a.op is not OptOps.TC and a.axis is not None:
    node = node2 = st["rng_nodes"][a.axis]
    sz = st["sizes"][a.axis]
    if a.op is OptOps.SPLIT:
      amt_cat = 1+AMTS.index(a.arg[0]) if a.arg[0] in AMTS else 0
      amt, left = math.log2(a.arg[0] or sz), math.log2(sz/(a.arg[0] or sz))
    elif a.op is OptOps.SWAP: node2 = st["rng_nodes"][a.arg]
    elif a.op is OptOps.PADTO: amt = math.log2(a.arg)
  return [i, kind, node, node2, amt_cat], [amt/8, left/16]

# *** the policy network ***

D_MAX = 8  # graph distances up to this are told apart
def relations(slot:np.ndarray) -> np.ndarray:
  # relation id between every pair of tokens (token 0 is the summary token), from the "i reads j" slots: how many reads apart, and in which direction
  B, N, _ = slot.shape
  adj = (slot > 0).astype(np.float32)
  dist, reach = adj.copy(), adj
  for d in range(2, D_MAX+1):
    reach = np.minimum(reach @ adj, 1)
    dist = np.where((reach > 0) & (dist == 0), d, dist)
  eye = np.eye(N, dtype=bool)[None]
  closure = np.minimum(adj+eye, 1)
  for _ in range(7): closure = np.minimum(closure @ closure, 1)
  dist = np.where((closure > 0) & (dist == 0) & ~eye, D_MAX+1, dist).astype(np.int32)
  dt = dist.transpose(0, 2, 1)
  inner = np.where(eye, 1, np.where(dist > 0, dist+1, 0)+np.where(dt > 0, dt+D_MAX+2, 0))
  rel = np.full((B, N+1, N+1), 2*D_MAX+4, np.int32)
  rel[:, 1:, 0], rel[:, 0, 0] = 2*D_MAX+5, 2*D_MAX+6
  rel[:, 1:, 1:] = inner
  return rel

def gelu(x:Tensor) -> Tensor: return 0.5*x*(1+(x/math.sqrt(2)).erf())

class PolicyNet:
  # w is a dict of weights by name. dropout is only applied inside Context(TRAINING=1)
  def __init__(self, w:dict[str, Tensor], drop:float=0.0):
    self.w, self.drop, self.heads = w, drop, w["rel.weight"].shape[1]
    self.layers = 1+max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
  def lin(self, n:str, x:Tensor) -> Tensor: return x.linear(self.w[n+".weight"].T, self.w[n+".bias"])
  def ln(self, n:str, x:Tensor) -> Tensor: return x.layernorm()*self.w[n+".weight"]+self.w[n+".bias"]
  def emb(self, n:str, i:Tensor) -> Tensor: return self.w[n+".weight"][i]

  def __call__(self, ni:Tensor, nf:Tensor, key:Tensor, slot:Tensor, rel:Tensor, glob:Tensor, act_int:Tensor, act_float:Tensor) -> Tensor:
    # ni (B,N,5) nf (B,N,F) key (B,N+1) slot, rel (B,N+1,N+1) glob (B,G) act_int (B,A,5) act_float (B,A,Fa) -> logits (B,A)
    B, N, H, w = ni.shape[0], ni.shape[1], self.heads, self.w
    x = self.emb("op", ni[..., 0])+self.emb("dt", ni[..., 1])+self.emb("arg", ni[..., 2])+self.emb("axpos", ni[..., 3].minimum(32)) \
      + self.emb("nsrc", ni[..., 4])+self.lin("nf.2", gelu(self.lin("nf.0", nf)))
    x = (w["cls"]+self.lin("glob.2", gelu(self.lin("glob.0", glob)))).unsqueeze(1).cat(x, dim=1)
    D = x.shape[-1]
    # the graph reaches the attention as a bias per pair of tokens: which input slot reads which, and how far apart two UOps are
    bias = (self.emb("slot_f", slot)+self.emb("slot_r", slot.transpose(1, 2))+self.emb("rel", rel)).permute(0, 3, 1, 2)
    bias = key.reshape(B, 1, 1, N+1).where(bias, -math.inf)
    for i in range(self.layers):
      q, k, v = self.lin(f"blocks.{i}.qkv", self.ln(f"blocks.{i}.n1", x)).reshape(B, N+1, 3, H, D//H).permute(2, 0, 3, 1, 4)
      a = ((q@k.transpose(-2, -1))/math.sqrt(D//H)+bias).softmax(-1).dropout(self.drop)@v
      x = x+self.lin(f"blocks.{i}.proj", a.transpose(1, 2).reshape(B, N+1, D)).dropout(self.drop)
      x = x+self.lin(f"blocks.{i}.mlp.3", gelu(self.lin(f"blocks.{i}.mlp.0", self.ln(f"blocks.{i}.n2", x))).dropout(self.drop)).dropout(self.drop)
    x = self.ln("norm", x)
    # an action is scored from the tokens of its two axes, the summary token, and what the action is
    A, rows = act_int.shape[1], Tensor.arange(B).reshape(B, 1)
    what = self.emb("kind", act_int[..., 1])+self.emb("amt", act_int[..., 4])+self.emb("act", act_int[..., 0])+self.lin("af", act_float)
    z = x[rows, (act_int[..., 2]+1).minimum(N)].cat(x[rows, (act_int[..., 3]+1).minimum(N)], x[:, :1].expand(B, A, D), what, dim=-1)
    return self.lin("policy.2", gelu(self.lin("policy.0", z))).squeeze(-1)

# a kernel is padded to the first of these UOp counts that holds it and its actions to A_MAX, so the network is compiled once per size
SIZES, A_MAX = (32, 64, 96, 128, 192, 256), 96
def pad_kernel(f:dict) -> list[np.ndarray]:
  n, N, e = len(f["ni"]), next(s for s in SIZES if s >= len(f["ni"])), f["edges"].astype(np.int64)
  ni, nf = np.zeros((1, N, 5), np.int32), np.zeros((1, N, N_NODE_FLOAT), np.float32)
  key, slot = np.zeros((1, N+1), bool), np.zeros((1, N, N), np.int32)
  ni[0, :n], nf[0, :n], key[0, :n+1] = f["ni"], f["nf"], True
  slot[0, e[:, 1], e[:, 0]] = e[:, 2]+1
  return [ni, nf, key, np.pad(slot, ((0, 0), (1, 0), (1, 0))), relations(slot)]

class Policy:
  def __init__(self, fn:str):
    w = {k: v.to(Device.DEFAULT).realize() for k, v in safe_load(fn).items()}
    self.gm, self.gs = w.pop("glob_mean").numpy(), w.pop("glob_std").numpy()
    net = PolicyNet(w)
    self.fwd = {n: TinyJit(lambda *x: net(*x).realize()) for n in SIZES}
  def __call__(self, f:dict, acts:list[int]) -> np.ndarray:
    # a score for each of the actions of the kernel with features f. an action's score does not depend on the other actions, so they can go in chunks
    kern, feats, out = pad_kernel(f)+[((f["g"]-self.gm)/self.gs)[None]], [action_feats(a, f) for a in acts], []
    for i in range(0, len(acts), A_MAX):
      ai, af = np.zeros((1, A_MAX, 5), np.int32), np.zeros((1, A_MAX, N_ACT_FLOAT), np.float32)
      for j, (x, y) in enumerate(feats[i:i+A_MAX]): ai[0, j], af[0, j] = x, y
      # BEAM=0: the kernels of the policy itself are not searched
      with Context(BEAM=0): out.append(self.fwd[kern[0].shape[1]](*[Tensor(x) for x in kern+[ai, af]]).numpy()[0])
    return np.concatenate(out)[:len(acts)]

@functools.cache
def get_policy() -> Policy: return Policy(getenv("POLICY", ""))

# *** the searches ***

class Timer:
  # times kernels as beam_search does. the unoptimized kernel comes first: its time and estimates are inputs of the policy
  def __init__(self, s:Scheduler, allow_test_size:bool):
    # the arguments of the kernel as beam_search makes them: a new buffer per buffer, the middle of its range for a scalar
    self.var_vals = var_vals = {p.expr:int(p.vmax+p.vmin)//2 for p in s.ast.variables()}
    params, dname = {p.arg.slot:p for p in s.ast.toposort() if p.op is Ops.PARAM and p.arg.slot >= 0}, s.ren.target.device
    self.args = [UOp.const(0, dtypes.int)]*(max(params, default=-1)+1)
    for i, p in params.items():
      self.args[i] = UOp.const(var_vals[p.expr], p.dtype) if p.addrspace is AddrSpace.ALU else UOp.new_buffer(dname, p.max_numel(), p.dtype)
    dev = Device[dname]
    self.kw = {"allow_test_size": allow_test_size, "clear_l2": hasattr(dev, "invalidate_caches"), "dev_timeout": getenv("BEAM_DEV_TIMEOUT", 1)}
    self.seen: set[bytes] = set()
    self.least_ops, self.root_tm, self.ops, self.mem = math.inf, math.inf, 0, 0
    if (proc:=search._try_compile((0, s))[1]) is None: return
    if (est:=proc[0].src[0].arg.estimates) is not None: self.ops, self.mem = int(sym_infer(est.ops, var_vals)), int(sym_infer(est.mem, var_vals))
    try: self.root_tm = min(search._time_program(proc[0], var_vals, self.args, **self.kw))
    except RuntimeError: pass
    self.seen.add(proc[0].src[3].arg)

  def __call__(self, proc:tuple[UOp, float]|None, early_stop:float) -> float|None:
    # None: nothing to time (it did not compile, is the same binary as a kernel timed before, or needs 1000x the compute of the smallest kernel).
    # inf: it did not run in time
    if proc is None or (lib:=proc[0].src[3].arg) in self.seen: return None
    est = proc[0].src[0].arg.estimates
    self.least_ops = min(ops:=sym_infer(est.ops if est is not None else 0, self.var_vals), self.least_ops)
    if self.least_ops*1000 < ops: return None
    self.seen.add(lib)
    try: return min(search._time_program(proc[0], self.var_vals, self.args, early_stop=early_stop, **self.kw))
    except RuntimeError: return math.inf

MIN_OPS = 1_000_000  # a kernel of fewer operations runs in microseconds and is not collected
def collect_search(s:Scheduler, amt:int, allow_test_size=True) -> Scheduler:
  # beam_search that keeps what it measures as training data for the policy: one file per searched kernel in POLICY_COLLECT, with every kernel
  # the search expanded, its legal actions, and for each action the log2 speedup over the unoptimized kernel of the kernel it gave (nan if that
  # one was not timed). a kernel that has its file is not searched again. POLICY_COLLECT_EVERY=3 searches one kernel in three, picked by its key,
  # which gets through three times the programs in the same time
  os.makedirs(out:=getenv("POLICY_COLLECT", ""), exist_ok=True)
  if os.path.exists(fn:=os.path.join(out, f"{(key:=s.ast.key.hex())}.npz")) or len(s.ast.toposort()) > MAX_NODES: return s
  if int(key[:8], 16) % getenv("POLICY_COLLECT_EVERY", 1): return s
  timer, pool = Timer(s, allow_test_size), get_worker_pool()
  if math.isinf(timer.root_tm) or timer.ops < MIN_OPS: return s
  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  beam: list[tuple[Scheduler, float]] = [(s, math.inf)]
  rows: list[tuple[dict, list[int], list[float]]] = []
  while True:
    cands: list[tuple[Scheduler, list[float], int]] = []
    for k, _ in beam:
      acts, y = search.get_kernel_actions(k, include_0=False), []
      if (f:=featurize(k, timer.root_tm, timer.ops, timer.mem)) is not None: rows.append((f, list(acts), y))
      y += [math.nan]*len(acts)
      cands += [(ck, y, j) for j, ck in enumerate(acts.values())]
    timed: list[tuple[Scheduler, float]] = []
    timer.least_ops = math.inf
    for n, proc in (map if pool is None else pool.imap_unordered)(search._try_compile, enumerate(ck for ck, _, _ in cands)):
      if (tm:=timer(proc, beam[0][1]*3)) is None or math.isinf(tm): continue
      timed.append((cands[n][0], tm))
      cands[n][1][cands[n][2]] = math.log2(timer.root_tm/tm)
    opts = sorted(timed, key=lambda x: x[1])
    if len(opts) == 0 or opts[0][1] < min_progress or beam[0][1]-opts[0][1] < min_progress: break
    beam = opts[:amt]
  feats = [action_feats(a, f) for f, acts, _ in rows for a in acts]
  with open(fn+".tmp", "wb") as fh:
    np.savez_compressed(fh, ni=np.concatenate([f["ni"] for f, _, _ in rows]), nf=np.concatenate([f["nf"] for f, _, _ in rows]),
      noff=np.cumsum([0]+[len(f["ni"]) for f, _, _ in rows]), ed=np.concatenate([f["edges"] for f, _, _ in rows]),
      eoff=np.cumsum([0]+[len(f["edges"]) for f, _, _ in rows]), glob=np.stack([f["g"] for f, _, _ in rows]),
      ai=np.array([x for x, _ in feats], np.int16).reshape(-1, 5), af=np.array([x for _, x in feats], np.float32).reshape(-1, N_ACT_FLOAT),
      aoff=np.cumsum([0]+[len(acts) for _, acts, _ in rows]), y=np.array([v for _, _, y in rows for v in y], np.float32),
      info=json.dumps({"vocab": VOCAB, "device": s.ren.target.device, "amt": amt, "root_tm": timer.root_tm, "ops": timer.ops, "mem": timer.mem}))
  os.replace(fn+".tmp", fn)
  return opts[0][0] if len(opts) and opts[0][1] < beam[0][1] else beam[0][0]

PENDING, DEAD = 1, 2
class Node:
  # a timed kernel. y is its log2 speedup over the unoptimized kernel, best is the best y at or below it
  def __init__(self, k:Scheduler, y:float, parent:"Node|None"=None):
    self.k, self.y, self.best, self.parent = k, y, y, parent
    self.acts: dict[int, Scheduler]|None = None  # legal actions, None until the search first walks through this kernel
    self.prior: dict[int, float] = {}
    self.kids: dict[int, Node] = {}              # timed children by action
    self.visits: dict[int, int] = {}             # walks through each action, counting the ones whose kernel is still compiling
    self.state: dict[int, int] = {}              # actions that are compiling (PENDING) or gave no timing (DEAD)
    self.n, self.exhausted = 0, False            # walks through this kernel, nothing left to try at or below it

def policy_mcts_search(s:Scheduler, amt:int, allow_test_size=True, callback:Callable[[int, Scheduler, int], None]|None=None) -> Scheduler:
  # amt is the number of kernels to time. callback gets (kernels timed, best kernel, kernels compiled) after every timing.
  # what the search ends on is kept in the disk cache, as beam_search keeps it
  key = {"ast": s.ast.key, "amt": amt, "policy": getenv("POLICY", ""), "allow_test_size": allow_test_size, "device": s.ren.target.device,
         "suffix": s.ren.suffix}
  if not IGNORE_BEAM_CACHE and CACHELEVEL >= 1 and (val:=diskcache_get("policy_mcts_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret
  timer, pool = Timer(s, allow_test_size), get_worker_pool()
  # with no time for the unoptimized kernel there is nothing to feed the policy: a narrow BEAM instead
  if math.isinf(timer.root_tm): return beam_search(s, 2, allow_test_size)
  # c weighs the prior against the measured speedups. bs kernels compile in the worker pool at once
  c, bs = getenv("POLICY_MCTS_C", 0.5), 1 if pool is None else getenv("POLICY_WINDOW", 16)

  def select(root:Node) -> tuple[Node, int]|None:
    # walk down by PUCT to an action that is not timed yet. a kernel whose untried actions are all compiling is skipped for this pick
    scale, blocked = max(root.best, 1.0), set()
    while root not in blocked and not root.exhausted:
      node, path = root, []
      while True:
        if node.acts is None:
          node.acts = search.get_kernel_actions(node.k, include_0=False)
          if len(node.acts):
            # a kernel with too many UOps for the policy gets the same prior for every action
            f = featurize(node.k, timer.root_tm, timer.ops, timer.mem)
            logits = np.zeros(len(node.acts)) if f is None else get_policy()(f, list(node.acts)).astype(np.float64)
            node.prior = dict(zip(node.acts, ((pr:=np.exp(logits-logits.max()))/pr.sum()).tolist()))
        sq, pick, pick_u, waiting = math.sqrt(node.n+1), 0, -math.inf, False
        for a in node.acts:
          if (state:=node.state.get(a)) is not None:
            waiting = waiting or state == PENDING
            continue
          if (kid:=node.kids.get(a)) is not None and (kid.exhausted or kid in blocked):
            waiting = waiting or kid in blocked
            continue
          # an action that is not timed yet counts as no better or worse than the kernel it is applied to
          u = max((node.y if kid is None else kid.best)/scale, -1.0)+c*node.prior[a]*sq/(1+node.visits.get(a, 0))
          if u > pick_u: pick, pick_u = a, u
        if pick == 0:
          if waiting: blocked.add(node)
          else: node.exhausted = True
          break
        path.append((node, pick))
        if (kid:=node.kids.get(pick)) is None:
          for n, a in path: n.n, n.visits[a] = n.n+1, n.visits.get(a, 0)+1
          node.state[pick] = PENDING
          return node, pick
        node = kid
    return None

  root = Node(s, 0.0)
  best, best_tm, timed, compiled = s, timer.root_tm, 0, 0
  inflight: dict[int, tuple[Node, int]] = {}
  done: queue.Queue = queue.Queue()
  while timed < amt and compiled < amt*4+bs:
    while timed+len(inflight) < amt and len(inflight) < bs and (sel:=select(root)) is not None:
      inflight[compiled], job = sel, (compiled, sel[0].acts[sel[1]])  # type: ignore
      if pool is None: done.put(search._try_compile(job))
      else: pool.apply_async(search._try_compile, (job,), callback=done.put, error_callback=lambda e,i=compiled: done.put((i, None)))
      compiled += 1
    if len(inflight) == 0: break  # the whole tree is explored
    i, proc = done.get()
    p, a = inflight.pop(i)
    p.state[a] = DEAD
    if (tm:=timer(proc, best_tm*3)) is None: continue
    timed += 1
    if not math.isinf(tm):
      assert p.acts is not None
      del p.state[a]
      node = p.kids[a] = Node(p.acts[a], math.log2(timer.root_tm/tm), p)
      if tm < best_tm: best, best_tm = node.k, tm
      up: Node|None = p
      while up is not None: up.best, up = max(up.best, node.y), up.parent
    if callback is not None: callback(timed, best, compiled)
  if CACHELEVEL >= 1: diskcache_put("policy_mcts_search", key, best.applied_opts)
  return best

if __name__ == "__main__":
  # apply_opts looks beam_search up in the search module every time it compiles a kernel, so this is all it takes to stand in for it
  search.beam_search = collect_search if getenv("POLICY_COLLECT", "") else policy_mcts_search
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], run_name="__main__")
