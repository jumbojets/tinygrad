# the whole loop of the learned kernel search on two small kernels: collect what BEAM measures, train a tiny policy on it, search with it
import unittest, os, sys, json, glob, subprocess, tempfile
import numpy as np
from tinygrad.nn.state import safe_load
from extra.optimization.policy_search import VOCAB, GLOBAL_NAMES, N_NODE_FLOAT, N_ACT_FLOAT

# a kernel compiled on its own and one compiled inside a TinyJit
PROGRAM = """from tinygrad import Tensor, TinyJit
(Tensor.empty(384, 384) @ Tensor.empty(384, 384)).realize()
@TinyJit
def conv(x:Tensor, w:Tensor) -> Tensor: return x.conv2d(w).relu().realize()
for _ in range(3): conv(Tensor.empty(4, 16, 64, 64).realize(), Tensor.empty(32, 16, 3, 3).realize())
"""
RUN = [sys.executable, "-m", "extra.optimization.policy_search"]

class TestPolicySearch(unittest.TestCase):
  def run_with(self, cmd:list[str], **env:str):
    r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": ".", "IGNORE_BEAM_CACHE": "1", **env}, capture_output=True, text=True)
    self.assertEqual(r.returncode, 0, r.stderr[-2000:])

  def test_collect_train_search(self):
    with tempfile.TemporaryDirectory() as d:
      prog, data, policy = os.path.join(d, "prog.py"), os.path.join(d, "data"), os.path.join(d, "policy.safetensors")
      with open(prog, "w") as f: f.write(PROGRAM)
      # collect: one file per searched kernel, every expanded kernel in it with its actions and the speedups of the timed ones
      self.run_with(RUN+[prog], POLICY_COLLECT=data, BEAM="2")
      files = glob.glob(os.path.join(data, "*.npz"))
      self.assertEqual(len(files), 2)
      for fn in files:
        k = np.load(fn)
        n, a = len(k["noff"])-1, k["aoff"][-1]
        self.assertEqual(json.loads(str(k["info"]))["vocab"], VOCAB)
        self.assertEqual((k["glob"].shape, k["nf"].shape[1], k["ai"].shape, k["af"].shape, k["y"].shape),
                         ((n, len(GLOBAL_NAMES)), N_NODE_FLOAT, (a, 5), (a, N_ACT_FLOAT), (a,)))
        self.assertGreaterEqual(int(np.isfinite(k["y"]).sum()), 2)
      # a second run searches nothing again
      before = {fn: os.path.getmtime(fn) for fn in files}
      self.run_with(RUN+[prog], POLICY_COLLECT=data, BEAM="2")
      self.assertEqual(before, {fn: os.path.getmtime(fn) for fn in glob.glob(os.path.join(data, "*.npz"))})
      # train: one file with the weights and the scale of the kernel-level inputs
      self.run_with([sys.executable, "extra/optimization/policy_train.py", data, "--out", policy, "--steps", "3", "--d", "32", "--layers", "1",
                     "--heads", "2"])
      w = safe_load(policy)
      self.assertEqual(w["glob_mean"].shape, (len(GLOBAL_NAMES),))
      # search with it: MCTS that times 20 kernels per kernel
      self.run_with(RUN+[prog], POLICY=policy, BEAM="20")

if __name__ == "__main__":
  unittest.main()
