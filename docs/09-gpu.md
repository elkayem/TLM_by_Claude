# Chapter 9 — Stage 4a: training on a GPU

*Code: `pick_device`, `autocast_for`, `get_batch` and the training loop in
`train.py`; `CausalSelfAttention.forward` in `model.py`; the `grad_accum`
and `fused_attention` fields in `config.py`*

Stages 1–3 were trained on a 20-core CPU. Stage 4 adds an NVIDIA RTX 3050
(6 GB), and the first thing worth saying is what *doesn't* change:

> **A GPU doesn't run a different algorithm. It runs the same matrix
> multiplies — thousands of them at once.**

Same model, same loss, same optimizer. The GPU port is about 100 lines,
and on the Stage 3 model it took training from ~1,250 to ~32,700 tokens
per second: **~27× faster**, turning a 5-day run into ~4.5 hours. This
chapter is about those 100 lines and why they work.

## Why a GPU is so much faster

Training a transformer is almost entirely matrix multiplication: roughly
6 floating-point operations per parameter per token (2 for the forward
pass, 4 for the backward pass), plus attention. A matrix multiply is
thousands of independent multiply-adds — the most parallel job there is.

- The CPU has 20 cores built for *complicated* work, one instruction
  stream each. Measured on Stage 3, it sustained ~0.24 TFLOPS (trillion
  floating-point operations per second).
- The RTX 3050 has 2,304 simple cores plus 72 **tensor cores** — units
  that do nothing but small matrix multiplies, in reduced precision. Its
  datasheet peak is ~13.5 TFLOPS in bf16 (see below). Measured: ~6.2
  TFLOPS sustained on the Stage 3 model, ~8.5 on the 126M Stage 4 model —
  bigger matrices keep the hardware busier (~63% of peak).

Nothing about this is specific to language models; it's why every deep
learning framework is built around the GPU.

## 1. Devices: same code, different processor

In PyTorch every tensor lives on a **device**, and an operation runs
wherever its inputs live. So the port is mostly *moving things*:

- `pick_device("auto")` picks `"cuda"` if PyTorch can see a GPU, else
  `"cpu"` (`--device` overrides it). Everything else is written once.
- `TLM(config).to(device)` moves every weight. The weights are created on
  the CPU first, under the usual seed, so a GPU run starts from exactly the
  weights a CPU run would.
- `get_batch` builds each batch in ordinary RAM, then copies it over.
  `pin_memory()` puts it in page-locked RAM the GPU can read directly, and
  `non_blocking=True` lets the copy overlap with other work.
- Checkpoints are loaded with `map_location="cpu"` and then copied into
  the model wherever it lives — so a run started on one device can
  `--resume` on the other.

One subtlety: GPU work is **asynchronous**. Python only *queues* kernels;
the GPU runs them in order while Python races ahead. Anything that needs
a number back (`loss.item()` at evaluation time) waits for the queue to
drain. That's why throughput is measured over hundreds of steps, never
one.

## 2. Mixed precision: bfloat16

A float32 number spends its 32 bits as 1 sign + 8 exponent + 23 fraction
bits. **bfloat16** keeps the sign and the full 8-bit exponent but only 7
fraction bits: the same *range* (tiny to huge), about 3 significant
digits instead of 7. Tensor cores multiply bf16 matrices at twice this
GPU's fp32 rate, and every activation takes half the memory.

Three significant digits sounds alarming, but most of training doesn't
need more. `autocast_for(device)` returns `torch.autocast(..., bfloat16)`
and the forward pass runs inside it, which gives the **mixed** part:

- matrix multiplies run in bf16;
- precision-sensitive operations — softmax, LayerNorm statistics, the
  cross-entropy loss — are automatically kept in fp32;
- the weights themselves, their gradients and AdamW's arithmetic stay
  fp32, so millions of tiny updates don't round away.

The older recipe used float16 (1 + **5** + 10 bits), whose small exponent
made gradients underflow to zero unless the loss was scaled up and back
down ("loss scaling", Micikevicius et al. 2017). bf16's fp32-sized
exponent makes that unnecessary — which is why it's the default on any
GPU that supports it (Ampere and later). On the CPU, `autocast_for`
returns a do-nothing context: CPU runs are unchanged.

**How much does it cost?** On a fixed validation batch, the finished
Stage 3 model scores 1.21786 in fp32 and 1.21799 under bf16 autocast — a
difference in the fourth decimal place.

## 3. Fused attention

Chapter 3's attention builds the full `(B, H, T, T)` score matrix, softmaxes
it and multiplies by `v` — and backpropagation needs that matrix kept in
memory. For the Stage 4 model (12 heads, T = 1,024) that's ~12.6 million
scores per sequence, per layer, times 16 layers.

`F.scaled_dot_product_attention` computes *exactly the same function* in
one fused kernel. On the GPU it streams through the score matrix in tiles
held in the chip's fast on-chip memory, keeping running softmax totals,
and never writes the whole matrix out — it recomputes tiles during the
backward pass instead (the FlashAttention idea, Dao et al. 2022). Less
memory traffic, which on a GPU is often the real bottleneck.

TLM now uses it by default (`fused_attention=True`), and keeps the
explicit version as `explicit_attention()` because *seeing* the matrix is
the point of chapter 3. Measured on the Stage 3 model:

| | explicit | fused |
|---|---|---|
| loss on a fixed batch (fp32) | 1.21786 | 1.21786 |
| training speed (GPU, bf16) | ~22,000 tok/s | ~41,000 tok/s |
| peak GPU memory | 4.3 GB | 2.4 GB |
| training speed (CPU) | 7.46 s/step | 6.73 s/step |

Same answer to five decimals, twice the speed, half the memory — and even
the CPU gets 10% faster.

## 4. Gradient accumulation

Stage 4's model is 126M parameters, and a good batch for it is ~32,768
tokens per optimizer step. Storing the activations for all of that at once
would need far more than 6 GB. **Gradient accumulation** gets the same
step from smaller pieces:

```python
optimizer.zero_grad()
for _ in range(config.grad_accum):         # e.g. 16 micro-batches
    x, y = get_batch(...)                  # each 2 x 1,024 tokens
    _, loss = model(x, y)
    (loss / config.grad_accum).backward()  # gradients ADD UP in .grad
optimizer.step()                           # one step, 32,768 tokens
```

`backward()` accumulates into `.grad` (chapter 6 — that's why
`zero_grad()` exists), so the micro-batches' gradients sum. Dividing each
loss by `grad_accum` makes the sum the gradient of the *average* loss,
which is exactly what one big batch would have produced. Checked: two
micro-batches of 8 sequences give the same gradient as one batch of 16,
to within 1.5e-5 (rounding). The cost is time, not quality: the step
takes 16 small passes instead of one big one.

The mixtape pilot measured **3.73 GB** of peak GPU memory at micro-batch 2
— comfortable. Micro-batch 4 would need ~1.7 GB more, too tight beside
the share of the 6 GB that Windows' desktop uses, so the preset stays at
2 × 16. Every earlier preset has `grad_accum = 1` and behaves as before.

## 5. Small GPU-only touches

- **Fused AdamW** (`fused=True` on the GPU): updates all 126M weights in
  a few big kernel launches instead of a loop over every weight tensor.
- The startup banner names the device and whether bf16 is on, and each
  evaluation line ends with `mem` — the peak GPU memory so far, the
  number that decides your batch size.

## How we checked it: port, then prove

A port that runs fast but trains slightly wrong is worse than no port, so
before any big run:

1. **Same function.** The finished Stage 3 model, one fixed batch, every
   combination of CPU/GPU × explicit/fused attention: all 1.21786.
2. **Same training.** The Stage 3 pilot (1,000 steps, same seed for the
   weights) on the GPU, against the CPU pilot's log:

   | step | CPU train / val | GPU train / val |
   |---|---|---|
   | 100 | 4.4319 / 4.4325 | 4.4391 / 4.4466 |
   | 200 | 3.9372 / 3.9255 | 3.9395 / 3.9350 |
   | 300 | 3.6468 / 3.6341 | 3.6442 / 3.6415 |
   | 400 | 3.4285 / 3.4506 | 3.4237 / 3.3787 |

   The curves agree to within the noise of a 20-batch estimate (the batches
   themselves are random, so they never match exactly). The GPU took 4
   minutes 27 seconds for all 1,000 steps; the CPU took ~1 h 52 min.
3. **Same checkpoints.** A CPU checkpoint resumes on the GPU, and the
   other way round.

## Running it

Install the CUDA build of PyTorch (see the README's Setup section) and do
nothing else: `--device auto` finds the GPU. Without one — or with the CPU
build installed — everything runs on the CPU just as in Stages 1–3.

Experiments:
- Train `instruct-pilot` with `--device cpu` and again without it;
  compare the `tok/s` column.
- Set `fused_attention=False` in a preset and watch the `mem` column
  climb.
- Halve `batch_size` and double `grad_accum` in a preset: same
  tokens/step, same loss curve (within noise), less memory.
- Make `autocast_for` return `contextlib.nullcontext()` on the GPU too:
  pure fp32. How much speed does bf16 buy?
- Run `nvidia-smi` in a second window during training: utilization near
  100%, power at the card's 70 W cap. That's what "compute-bound" looks
  like.
