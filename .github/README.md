# Kernel search with a learned policy

This branch adds a learned kernel search to [tinygrad](https://github.com/tinygrad/tinygrad): MCTS over a kernel's opts with a small
transformer as the prior, in place of `beam_search`. It is three files in [`extra/optimization/`](/extra/optimization), with no change
under `tinygrad/`:

- [`kernel_generator.py`](/extra/optimization/kernel_generator.py): random small model-like programs. Their kernels are the training data,
  so no list of real programs is needed.
- [`policy_search.py`](/extra/optimization/policy_search.py): the policy network, the MCTS (`policy_mcts_search`) and the BEAM that collects
  training data (`collect_search`).
- [`policy_train.py`](/extra/optimization/policy_train.py): trains the policy in tinygrad.

[`test_policy_search.py`](/extra/optimization/test_policy_search.py) runs the three end to end. tinygrad's own README is [here](/README.md).

## Use

```sh
# collect: BEAM=4 searches every kernel the programs compile and keeps what it measured in data/
POLICY_COLLECT=data BEAM=4 python -m extra.optimization.policy_search extra/optimization/kernel_generator.py 0 1 2 3
# train
python extra/optimization/policy_train.py data --out policy.safetensors
# search: BEAM is the number of kernels MCTS times for each kernel
POLICY=policy.safetensors BEAM=200 python -m extra.optimization.policy_search program.py
```

Any tinygrad program can stand in for the generator. A kernel that is already in `data/` is not searched again, so a collection can be stopped
and started, and collections from several machines (other seeds on each) can be copied into one directory.

## Results

One run from nothing on tinygrad `f0b800324`, 22 and 23 November 2026:

- **Data:** 492 generated programs, searched with `BEAM=4` on A100s with the CUDA backend: 3,106 kernels, 53,745 kernel states with timed actions.
- **Policy:** 5.3 million parameters, 12 epochs, trained three times from different seeds on an A10 with the NV backend.
- **Evaluation:** kernels of real programs, none of them in the training data. Each kernel is searched with `BEAM=4` and with MCTS under each
  of the three policies, and the kernels the searches end on are timed together.

The tables give the time of the final kernels against `BEAM=4`'s as a geometric mean over kernels, averaged over the three policies: negative
is faster than `BEAM=4`. The range comes from resampling the kernels. Search time is per kernel.

In short: MCTS at 200 timed kernels ends on kernels as fast as `BEAM=4`'s on both devices in 66% to 84% of its search time.
At 100 timed kernels it still matches `BEAM=4` on the A100's held-out programs in about a quarter of the time, and is 7% slower on the A10.

### A100, CUDA: the device the training data was timed on

On the held-out programs MCTS at 200 timed kernels ends on kernels 1.2% faster (−3.2% to +0.8%) than `BEAM=4`'s in 84% of
the search time, and at 100 timed kernels 0.1% faster (−1.8% to +1.6%) in 26%.

Held-out programs, kept out of all earlier work (272 kernels of 6 programs):

| search | kernels timed | search time | final kernel time against BEAM=4 | range | each policy |
|---|---|---|---|---|---|
| hand-coded opts, no search | 0 | 0 | +41.8% | +32.6% to +52.4% | |
| `BEAM=4` | 324 | 27.7 s | baseline | | |
| MCTS, `BEAM=50` | 51 | 2.9 s (0.10x) | +6.5% | +4.1% to +9.1% | +7.4% +6.4% +5.8% |
| MCTS, `BEAM=100` | 101 | 7.2 s (0.26x) | −0.1% | −1.8% to +1.6% | −0.5% +0.2% −0.2% |
| MCTS, `BEAM=200` | 201 | 23.3 s (0.84x) | −1.2% | −3.2% to +0.8% | −1.4% −1.6% −0.7% |

Other real programs (122 kernels of 15 programs, one kernel in three to one in nine):

| search | kernels timed | search time | final kernel time against BEAM=4 | range | each policy |
|---|---|---|---|---|---|
| hand-coded opts, no search | 0 | 0 | +74.3% | +57.1% to +95.4% | |
| `BEAM=4` | 361 | 38.1 s | baseline | | |
| MCTS, `BEAM=50` | 51 | 3.3 s (0.09x) | +12.3% | +7.5% to +17.6% | +12.5% +10.7% +13.7% |
| MCTS, `BEAM=100` | 101 | 7.6 s (0.20x) | +2.5% | −0.3% to +5.5% | +1.6% +2.1% +3.9% |
| MCTS, `BEAM=200` | 201 | 25.0 s (0.66x) | −0.9% | −3.8% to +2.1% | −0.4% −1.5% −0.7% |

By program, the sum of the final kernel times against `BEAM=4`'s (`*`: other programs):

| program | kernels | hand-coded | MCTS 50 | MCTS 100 | MCTS 200 |
|---|---|---|---|---|---|
| seq_ops | 83 | +86% | +11% | +1% | −1% |
| yolov8 | 51 | +160% | +21% | +5% | +1% |
| convnext | 49 | +556% | +32% | +2% | 0% |
| onnx_fbnetc_100 | 43 | +63% | −1% | −1% | −2% |
| onnx_hardcorenas_b | 39 | +50% | +2% | +2% | +4% |
| t5 | 7 | +87% | +42% | +1% | +1% |
| stable_diffusion_unet * | 26 | +225% | +45% | +10% | −1% |
| resnet18_train_bs32 * | 23 | +196% | +64% | +25% | +13% |
| cifar_train * | 16 | +299% | +85% | +34% | +9% |
| efficientnet_b0 * | 11 | +36% | 0% | −4% | −5% |
| llama * | 9 | +94% | +8% | +4% | −1% |
| bert * | 8 | +111% | +1% | −15% | −20% |
| gpt2 * | 6 | +175% | −2% | −3% | −2% |
| retinanet * | 6 | +104% | +5% | −5% | −6% |
| dit * | 4 | +68% | +70% | +4% | −1% |
| resnet50_bs1 * | 4 | +1057% | +6% | +1% | −1% |
| vit * | 4 | +19% | +9% | −6% | −6% |
| whisper * | 2 | +42% | +36% | +30% | +31% |
| lstm * | 1 | +57% | +5% | +7% | +1% |
| mamba * | 1 | −17% | −21% | −22% | −10% |
| rnnt * | 1 | +95% | +12% | +6% | +5% |

### A10, NV: a GPU and backend with no training data

The policies have no input for the device, so this shows how they carry over. One kernel in two of the held-out programs
(143 kernels of 6 programs):

| search | kernels timed | search time | final kernel time against BEAM=4 | range | each policy |
|---|---|---|---|---|---|
| hand-coded opts, no search | 0 | 0 | +114.1% | +92.0% to +141.5% | |
| `BEAM=4` | 390 | 17.5 s | baseline | | |
| MCTS, `BEAM=50` | 51 | 2.3 s (0.13x) | +28.1% | +21.9% to +34.9% | +28.9% +28.3% +27.0% |
| MCTS, `BEAM=100` | 101 | 4.9 s (0.28x) | +7.2% | +4.4% to +10.0% | +10.0% +6.1% +5.5% |
| MCTS, `BEAM=200` | 200 | 11.5 s (0.66x) | +0.6% | −1.7% to +3.0% | +1.6% +0.6% −0.3% |

### Notes

- `BEAM=4` times every candidate, 324 kernels for a kernel on the A100 on average. MCTS stops at the number it is given.
- On the A100 every candidate is timed at full size (`BEAM_ESTIMATE=0`), because the CUDA backend adds a launch delay to every timing.
- The policy was trained on an A10 because upstream tinygrad cannot yet train it on an A100: with CUDA the host program of a training step
  exceeds a limit of the tcc compiler, and NV does not start on that GPU.
- Not every kernel of every program was searched. On the A100, seq_ops has 83 of about 89 kernels: the searches of two kernels were
  stopped as too slow, and the few kernels after them were not reached. On the A10 it stops at 48 kernels for the same reason.
  stable_diffusion_unet, bert and resnet18_train_bs32 stop partway on the A100, where one of their own kernels does not compile with the
  CUDA backend. 3 evaluation kernels that the generator had also produced were left out.
- The other programs gave training kernels to earlier experiments, which shaped the generator. None of their kernels trained these policies.
- A guided BEAM (BEAM that times the policy's best candidates first and ends a step early) was measured on the first 234 A100
  kernels and 80 A10 kernels: +1.8% (−0.3% to +4.0%) against `BEAM=4` at 0.33x the search time on the A100 and
  +5.4% (+2.8% to +8.0%) at 0.47x on the A10. It was removed.
