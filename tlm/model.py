"""
The TLM transformer: a decoder-only GPT, built from torch.nn primitives.

This file is the heart of the project. Read it top to bottom alongside
docs/03-attention.md through docs/05-full-model.md.

SHAPE NOTATION used in comments throughout:
    B = batch size          (sequences processed in parallel)
    T = time / sequence len (number of tokens, <= config.block_size)
    C = channels = n_embd   (width of the residual stream)
    H = n_head              (number of attention heads)
    hs = C // H             (head size: channels each head works with)

So a tensor annotated (B, T, C) holds, for every sequence in the batch and
every token position, a C-dimensional vector.

THE ONE-SENTENCE SUMMARY of the whole architecture:
    A token becomes a vector; a stack of blocks repeatedly lets each
    position (a) pull in information from earlier positions [attention]
    and (b) think about what it has so far [MLP]; a final linear layer
    turns the vector at each position into a probability for every
    possible next token.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: the mechanism that lets each token
    look back at earlier tokens and copy in information from them.

    THE INTUITION
    -------------
    Every position emits three vectors, produced by three learned linear
    maps of its input:

      query (q): "what am I looking for?"
      key   (k): "what do I contain / how should others find me?"
      value (v): "what information do I hand over if someone attends to me?"

    Position i scores every position j<=i by the dot product q_i . k_j
    (large when the query 'matches' the key), turns the scores into weights
    with a softmax, and receives the weighted average of the values. That
    weighted average is the attention output: information physically moved
    from earlier positions into this one.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must divide evenly into n_head heads"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One big linear layer computes q, k and v for ALL heads at once
        # (3 * n_embd outputs). This is just an efficiency trick: it is
        # mathematically identical to three separate n_embd->n_embd layers.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)

        # After the heads are concatenated back together, this output
        # projection lets them mix. Without it each head's result would land
        # in its own private slice of the channel dimension forever.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # THE CAUSAL MASK. A lower-triangular matrix of ones:
        #   position 0 may see: [0]
        #   position 1 may see: [0, 1]
        #   position 2 may see: [0, 1, 2]  ... etc.
        # This is what makes the model a LANGUAGE model: during training,
        # every position predicts its next token using only the past, so
        # one sequence of T tokens gives T separate prediction problems.
        # register_buffer = part of the module's state but NOT a trainable
        # parameter (the optimizer never touches it).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask",
                             mask.view(1, 1, config.block_size,
                                       config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        # Project the input into q, k, v (each (B, T, C)) in one matmul,
        # then split the last dimension into three.
        qkv = self.qkv_proj(x)                       # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # 3 x (B, T, C)

        # SPLIT INTO HEADS. Reshape (B, T, C) -> (B, T, H, hs) then swap
        # the T and H axes -> (B, H, T, hs). Each head now operates on its
        # own hs-dimensional slice of every token, and everything below is
        # batched over both B and H.
        # Why multiple heads? Each head has its own q/k/v maps, so each can
        # learn a DIFFERENT lookup pattern (one may track "the previous
        # word", another "the subject of the sentence").
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)

        # ---- THE ATTENTION EQUATION:  softmax(q k^T / sqrt(hs)) v --------
        #
        # Step 1: raw scores. q @ k^T computes ALL query-key dot products
        # at once: att[b, h, i, j] = q_i . k_j = "how relevant is token j
        # to token i, according to head h". Shape: (B, H, T, T) - a full
        # token-by-token relevance matrix. This T x T matrix is why context
        # length is expensive: double T and this quadruples.
        #
        # The 1/sqrt(hs) scaling: a dot product of two hs-dimensional
        # vectors with ~unit-variance entries has variance ~hs, so its
        # typical magnitude grows like sqrt(hs). Dividing by sqrt(hs)
        # keeps scores O(1) regardless of head size. Without it, softmax
        # saturates (one weight ~1, rest ~0) and gradients through it
        # vanish - training stalls.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)   # (B, H, T, T)

        # Step 2: causality. Overwrite every score where j > i (the future)
        # with -infinity. After softmax, e^{-inf} = 0: the future gets
        # exactly zero weight. We mask with -inf BEFORE softmax (rather
        # than zeroing after) so the remaining weights still sum to 1.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0,
                              float("-inf"))

        # Step 3: softmax over the LAST axis (the j axis, "who to look
        # at"), turning each row of scores into a probability distribution
        # over visible positions.
        att = F.softmax(att, dim=-1)                       # (B, H, T, T)
        att = self.attn_dropout(att)

        # Step 4: use those probabilities to take a weighted average of the
        # value vectors. y[b, h, i] = sum_j att[i, j] * v[j]. This is the
        # actual information transfer.
        y = att @ v                                        # (B, H, T, hs)

        # RE-ASSEMBLE THE HEADS: (B, H, T, hs) -> (B, T, H, hs) -> (B, T, C)
        # (undoing the earlier split), then mix them with the output
        # projection. .contiguous() makes the memory layout match the new
        # axis order so .view() can flatten it.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """The feed-forward network: attention's counterpart in each block.

    Attention MOVES information between positions; the MLP PROCESSES it.
    It is applied to every position independently and identically - no
    interaction between tokens happens here at all.

    Shape story: expand to 4x width, nonlinearity, contract back.
    The 4x is the transformer-standard ratio from the original paper.
    A useful mental model: the wide layer is a bank of 4*C learned feature
    detectors, and the contraction re-summarizes their findings into the
    residual stream. Most of the model's parameters (and, plausibly, its
    stored "knowledge") live in these MLP weights.
    """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)        # (B, T, C)  -> (B, T, 4C)
        # GELU: a smooth version of ReLU (used by GPT-2 onwards). The
        # nonlinearity is what lets the MLP compute non-trivial functions -
        # a stack of linear layers with no nonlinearity collapses to one
        # big linear layer, no matter how deep.
        x = F.gelu(x)
        x = self.proj(x)      # (B, T, 4C) -> (B, T, C)
        return self.dropout(x)


class Block(nn.Module):
    """One transformer block: attention + MLP, each wrapped in a residual
    connection and preceded by LayerNorm. The full model is just this
    block stacked n_layer times.

    THE RESIDUAL STREAM VIEW
    ------------------------
    Note the shape of forward(): x = x + f(x), twice. Nothing ever
    REPLACES x; sub-layers only ADD to it. Think of x as a shared
    "communication bus" flowing up through the network which every layer
    reads from and writes small updates onto. Two reasons this matters:

    1. Gradients: through x = x + f(x), the gradient of the loss flows
       back through the identity path untouched. Even 100 layers deep,
       layer 1 receives a clean gradient signal. Without residuals, deep
       transformers essentially don't train.
    2. Optimization: each block only needs to learn a small CORRECTION to
       the stream, not re-derive the whole representation.

    PRE-NORM: we apply LayerNorm on the way INTO each sub-layer
    (x + f(norm(x))), rather than after the addition as in the original
    2017 paper. This keeps the residual path itself completely
    normalization-free (pure additions from embedding to output), which
    trains much more stably - it's what GPT-2 and essentially everything
    since uses.

    LayerNorm itself: for each individual token vector, shift/rescale its
    C numbers to mean 0 and variance 1, then apply a learned per-channel
    scale and offset. It keeps activation magnitudes in a healthy range no
    matter how the sums on the residual stream grow.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # communicate: mix across positions
        x = x + self.mlp(self.ln2(x))   # compute: process each position
        return x


class TLM(nn.Module):
    """The full Tiny Language Model."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size > 0, "vocab_size must be set (see data prep)"
        self.config = config

        # Token embedding: a learned lookup table, one C-dim vector per
        # vocabulary entry. "Embedding" sounds fancy but it is literally
        # vocab_size x C numbers indexed by token id.
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding: one learned C-dim vector per POSITION
        # (0..block_size-1). Necessary because attention itself is
        # order-blind: it's a weighted average over a SET of positions, so
        # without this, "dog bites man" and "man bites dog" would be
        # indistinguishable. Adding a position-dependent vector lets the
        # model know where each token sits.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final norm (pre-norm twin)

        # The language-model head maps the final C-dim vector at each
        # position to vocab_size raw scores ("logits"), one per possible
        # next token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size,
                                 bias=False)

        # WEIGHT TYING: the lm_head and the token embedding share one
        # weight matrix (both are vocab_size x C). Intuition: the map from
        # "token -> vector" and "vector -> token scores" are two directions
        # of the same relationship. Saves vocab_size*C parameters and
        # slightly improves quality (Press & Wolf 2016).
        self.lm_head.weight = self.tok_emb.weight

        # Initialize all weights, then apply the GPT-2 residual-projection
        # trick: layers that WRITE ONTO the residual stream (attn out_proj,
        # mlp proj) are scaled down by 1/sqrt(2*n_layer). With 2*n_layer
        # writes adding up along the stream, this keeps the total variance
        # at initialization independent of depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        # GPT-2's scheme: small gaussian weights, zero biases. "Small" so
        # that early softmaxes/activations start in their well-behaved
        # near-linear regime.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # Subtract pos_emb to match the GPT-2 counting convention; note the
        # tied lm_head/tok_emb matrix is naturally counted only once.
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        """idx: (B, T) integer token ids.
        Returns (logits, loss); loss is None unless `targets` is given.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"

        # Look up token vectors and add position vectors. Broadcasting:
        # (B, T, C) + (T, C) -> the same position vectors are added to
        # every sequence in the batch.
        pos = torch.arange(T, device=idx.device)          # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)         # (B, T, C)
        x = self.drop(x)

        # The main event: N rounds of communicate-then-compute.
        for block in self.blocks:
            x = block(x)                                  # (B, T, C)

        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab)

        loss = None
        if targets is not None:
            # CROSS-ENTROPY LOSS = -log(probability the model assigned to
            # the true next token), averaged over all B*T positions. Every
            # position is its own prediction problem (thanks to the causal
            # mask), so we flatten batch and time together:
            #   logits: (B, T, V) -> (B*T, V),  targets: (B, T) -> (B*T,)
            # cross_entropy applies the softmax internally, in a
            # numerically stable way - that's why the model outputs raw
            # logits rather than probabilities.
            #
            # Yardstick: a model guessing uniformly at random scores
            # loss = ln(vocab_size) (e.g. ~4.17 for a 65-char vocab).
            # Watching loss fall from there is watching learning happen.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, stop_token=None):
        """Autoregressive sampling: the loop that turns a next-token
        predictor into a text generator.

        Feed the context in, get a distribution for the next token, sample
        one, APPEND IT, repeat. The model has no memory between iterations
        - the growing sequence itself is the memory.

        Knobs (see docs/07-sampling.md):
          temperature - divide logits by t before softmax. t<1 sharpens
                        (safer, more repetitive), t>1 flattens (wilder).
                        As t->0 this becomes greedy argmax decoding.
          top_k       - only sample among the k highest-probability tokens.
          top_p       - "nucleus": only sample among the smallest set of
                        tokens whose probabilities sum to p.
          Both cut off the long tail of low-probability tokens, which is
          where degenerate gibberish comes from.
          stop_token   - if the model samples this id (e.g. the
                        <|endoftext|> special token), stop generating:
                        the model has said "the story is over". Only
                        meaningful for batch size 1.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens: feed it the LAST
            # block_size tokens of the running sequence (a sliding window).
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)

            # Only the LAST position's logits matter here: they hold the
            # prediction for the token after the end of the sequence.
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, V)

            if top_k is not None:
                # Zero out (set to -inf) everything below the k-th best.
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                logits[logits < kth[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                # Nucleus sampling: sort descending, find where cumulative
                # probability passes top_p, drop everything after that.
                sorted_p, sorted_ix = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                # shift right by one so the token that CROSSES the
                # threshold is kept (otherwise we could drop everything).
                drop = cum - sorted_p > top_p
                sorted_p[drop] = 0.0
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                # sample in sorted space, then map back to token ids
                next_sorted = torch.multinomial(sorted_p, num_samples=1)
                idx_next = sorted_ix.gather(-1, next_sorted)
            else:
                # Sample from the distribution (NOT argmax - always taking
                # the single most likely token produces dull, loop-prone
                # text; the randomness is doing real work here).
                idx_next = torch.multinomial(probs, num_samples=1)

            if (stop_token is not None and idx.size(0) == 1
                    and idx_next.item() == stop_token):
                break  # end-of-text sampled: don't append it, just stop

            idx = torch.cat((idx, idx_next), dim=1)  # append; length grows
        return idx
