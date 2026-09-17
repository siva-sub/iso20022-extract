"""The ISO 20022 extraction model: a small decoder-only transformer.

Port of Needle's design decisions to PyTorch so the artifact is shareable on
HuggingFace: grouped-query attention, rotary positions, pre-norm RMS norm,
SwiGLU feed-forward, a hashed n-gram memory ("engram"), a correctness confidence
head, tied embeddings, no linear biases.

Blocks, and what each is for:

* **GQA (8 query / 4 KV heads).** Halves the K/V projection cost and cache for a
  loss that is negligible at this depth.
* **Engram.** A hashed n-gram memory read. Token n-grams are hashed into a table
  of learnable vectors, fetched, and content-addressed against the current
  hidden state; the result is added as a residual *before* self-attention. It is
  a memory read, not extra attention positions -- which makes it cheap. Payment
  documents are dense in repeated short n-grams (currency codes, BIC shapes,
  IBAN country prefixes, XML element names), so recall of exactly those patterns
  is worth more here than at general language scale.
* **Confidence head.** Pools hidden states through learned probes into one
  correctness logit for the whole answer. PROCESS.md 2.4 removed the confidence
  gate, and 16.3 makes hallucination the decisive risk, so this is the capability
  the project is actually missing: without a calibrated correctness score the
  pipeline cannot be safely automated at all.
* **Multi-token prediction** (optional). A second block predicting the token
  after next, densifying the training signal on ~120-token targets.

Everything is standard PyTorch plus `scaled_dot_product_attention`, so the model
runs on CPU or GPU without custom kernels.
"""

# pyright: reportPrivateImportUsage=false
#
# This must be a real comment before any code: pyright reads comments, so the
# same text inside the docstring above has no effect.
#
# Why: torch/__init__.py does `from torch._C import *`, and torch._C is a compiled
# extension with no stub, so pyright cannot follow the wildcard and reports every
# `torch.zeros` / `torch.cat` / `torch.int64` / `torch.dtype` as a private import
# from torch._C._VariableFunctions. All of them are present in torch.__all__ (910
# entries) and resolve at runtime -- verified by hasattr and by the model training
# on GPU. Importing from `torch._C._VariableFunctions` to satisfy the checker
# would be private API that breaks on torch upgrades, so the public API stays and
# this one rule is disabled for this file only.
#
# Note this rule fires under pyright-langserver but not under basedpyright, which
# is why the CLI reports 0 errors on this package while the daemon reports 30.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .configuration_iso20022 import ISO20022Config

from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# FNV-style mixing constants, matching Needle's engram hash.
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class ISO20022RMSNorm(nn.Module):
    """RMS norm with a (1 + scale) parameterisation.

    Needle's ZCRMSNorm initialises scale at zero, so the layer starts as an
    identity map and the residual stream is unperturbed at step 0. That matters
    at this depth: 20 pre-norm layers with a randomly initialised scale multiply
    the signal on the way in and slow the early epochs measurably.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        x = hidden.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * x * rms).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    The frequencies are derived on every forward rather than held in a buffer.

    They used to live in ``register_buffer("inv_freq", ..., persistent=False)``,
    and a non-persistent buffer is not written to the checkpoint. Loading
    allocates every buffer the module declares but only fills the ones present in
    the state dict, so a reloaded model ran on whatever memory it was handed --
    measured as ``[-9.65e-11, 3.09e-41, 0.0, 0.0]`` where the trained model had
    ``[1.0, 0.6978, 0.4869, 0.3398]``.

    Nothing about the checkpoint looked wrong. All 214 weight tensors matched
    byte for byte, every config field round-tripped, and this buffer's name and
    shape were correct; only its contents were absent. Identical parameters
    therefore computed a *different function*: one run logged ``loss 0.0129
    recall 100.0%`` while every checkpoint it wrote measured ``loss 3.7`` and
    generated repetition loops. Restoring the right frequencies made the two
    forward passes agree exactly, ``max|diff| = 0.0``.

    Deriving them from ``head_dim`` and ``theta`` removes the failure mode
    instead of guarding it: there is no stored state to go stale, the values
    follow the input device, and nothing is added to the state dict, so
    checkpoints written before this fix still load.
    """

    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.theta = theta

    def inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """RoPE frequencies, recomputed from this module's own parameters."""
        exponents = torch.arange(0, self.head_dim, 2, device=device).float()
        return 1.0 / (self.theta ** (exponents / self.head_dim))

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq(position_ids.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand KV heads to the query head count for grouped-query attention."""
    if repeats == 1:
        return hidden
    batch, heads, seq, dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, heads, repeats, seq, dim)
    return hidden.reshape(batch, heads * repeats, seq, dim)


def _roll_time(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Shift a (B, T, D) tensor right along time, zero-filling the front.

    When `offset >= T` every element is shifted out, so the result is all zeros.
    Slicing alone returns a tensor of length `offset` instead -- `x[:, :-offset]`
    goes empty rather than raising -- and the elementwise product with the tap
    mask then fails on shape. That is a crash at short sequences only, which is
    exactly the regime a generated-token loop hits.
    """
    if offset == 0:
        return x
    if offset >= x.shape[1]:
        return torch.zeros_like(x)
    pad = x.new_zeros(x.shape[0], offset, x.shape[2])
    return torch.cat((pad, x[:, :-offset]), dim=1)


def engram_indices(
    tokens: torch.Tensor,
    orders: tuple[int, ...],
    heads: int,
    slots: int,
) -> torch.Tensor:
    """Hash each position's trailing n-grams into table indices.

    FNV-style: XOR the shifted token into an accumulator, multiply by a prime,
    then mix high bits down before the modulo. The seed varies per (order, head)
    so the streams do not collide on the same slots.
    """
    u = tokens.to(torch.int64) & _MASK32
    collected: list[torch.Tensor] = []
    for order_index, order in enumerate(orders):
        for head in range(heads):
            seed = (_ENGRAM_SEED * (order_index * heads + head + 1)) & _MASK32
            acc = torch.full_like(u, seed)
            for offset in range(order):
                acc = ((acc ^ _roll_time_int(u, offset)) * _ENGRAM_PRIME) & _MASK32
            acc = acc ^ (acc >> 15)
            collected.append(acc % slots)
    return torch.stack(collected, dim=-1)  # (B, T, num_tables)


def _roll_time_int(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Same shift for 2-D token ids; same `offset >= T` guard, for the same reason."""
    if offset == 0:
        return x
    if offset >= x.shape[1]:
        return torch.zeros_like(x)
    pad = x.new_zeros(x.shape[0], offset)
    return torch.cat((pad, x[:, :-offset]), dim=1)


def _positions(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device)


class EngramMemory(nn.Module):
    """Hashed n-gram memory, read by content address.

    For each position the trailing n-grams are hashed, the corresponding table
    rows are fetched, and those vectors are compared against the current hidden
    state. The similarity becomes a gate on the fetched value, which is added as
    a residual. This is a memory read: it lets the model recall a BIC prefix or a
    currency code without spending an attention head re-deriving it.

    Two details are load-bearing and easy to get wrong:

    * **Causal validity.** An n-gram of order `o` ending at position `t` spans
      `t-o+1 .. t`, so it only exists once `t >= o-1`. Without zeroing the
      invalid prefixes the model reads a hash of padding, which is noise that
      looks like signal.
    * **Scaled value projection.** Needle initialises the value projection with
      `std / sqrt(2 * num_layers)` so the memory starts as a near-null addition.
      An unscaled residual here injects noise into every layer at step 0.
    """

    def __init__(self, config: ISO20022Config):
        super().__init__()
        heads, sub_dim = config.engram_geometry()
        self.orders = tuple(config.engram_orders)
        self.heads = heads
        self.sub_dim = sub_dim
        self.slots = config.engram_slots
        self.num_tables = len(self.orders) * heads
        self.conv_taps = config.engram_conv_taps
        self.dilation = max(self.orders)
        width = self.num_tables * sub_dim

        std = config.initializer_range
        self.tables = nn.Parameter(torch.empty(self.num_tables, self.slots, sub_dim))
        nn.init.normal_(self.tables, mean=0.0, std=std)
        self.key_proj = nn.Linear(width, config.d_model, bias=False)
        self.value_proj = nn.Linear(width, config.d_model, bias=False)
        # Identity at tap 0 so the temporal filter starts as a pass-through.
        taps = torch.zeros(self.conv_taps, config.d_model)
        taps[0] = 1.0
        self.taps = nn.Parameter(taps)
        # `tables`, `key_proj` and the scaled `value_proj` are initialised by
        # ISO20022PreTrainedModel._init_weights, which runs on every submodule.

    def _validity(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(T, num_tables) mask: 1 where the n-gram is fully inside the prefix."""
        t = _positions(seq_len, device)
        masks = []
        for order in self.orders:
            ok = (t >= order - 1).to(torch.float32)
            masks.extend([ok] * self.heads)
        return torch.stack(masks, dim=-1)

    def _tap_validity(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(T, taps) mask for the dilated temporal filter."""
        t = _positions(seq_len, device)
        return torch.stack(
            [(t >= j * self.dilation).to(torch.float32) for j in range(self.conv_taps)],
            dim=-1,
        )

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        indices = engram_indices(tokens, self.orders, self.heads, self.slots)
        flat = indices.reshape(-1, self.num_tables)
        # Each COLUMN indexes its own table's slot space, so offset column t by
        # t * slots before gathering. Without this every stream reads table 0,
        # three quarters of the table is unreachable, and the four n-gram orders
        # collapse into one -- which trains fine and quietly does nothing.
        offsets = torch.arange(self.num_tables, device=indices.device) * self.slots
        flat = flat + offsets
        table = self.tables.reshape(self.num_tables * self.slots, self.sub_dim)
        fetched = F.embedding(flat, table).reshape(
            batch, seq_len, self.num_tables, self.sub_dim
        )
        validity = self._validity(seq_len, tokens.device)
        fetched = fetched * validity[None, :, :, None]
        encoded = fetched.reshape(batch, seq_len, self.num_tables * self.sub_dim)

        keys = self.key_proj(encoded)
        values = self.value_proj(encoded)
        taps = self._tap_validity(seq_len, tokens.device)
        smoothed = torch.zeros_like(values)
        for j in range(self.conv_taps):
            weight = self.taps[j].to(values.dtype)
            smoothed = (
                smoothed
                + weight * _roll_time(values, j * self.dilation) * taps[None, :, j, None]
            )

        # Content address: scaled cosine similarity, squashed to a gate.
        similarity = (
            F.normalize(hidden.float(), dim=-1) * F.normalize(keys.float(), dim=-1)
        ).sum(-1, keepdim=True) / math.sqrt(self.num_tables)
        gate = torch.sigmoid(similarity).to(hidden.dtype)
        return hidden + gate * smoothed


class ISO20022Attention(nn.Module):
    """Causal grouped-query attention with rotary positions."""

    def __init__(self, config: ISO20022Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.d_model, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            .view(batch, seq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .view(batch, seq, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .view(batch, seq, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class ISO20022MLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, config: ISO20022Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class ISO20022DecoderLayer(nn.Module):
    """Pre-norm residual block with an optional engram read and attention gate."""

    # Same reason as RotaryEmbedding: without an explicit annotation this reads
    # as 'Tensor | Module' through nn.Module.__getattr__.
    engram: EngramMemory | None

    def __init__(self, config: ISO20022Config, use_engram: bool = False):
        super().__init__()
        self.self_attn = ISO20022Attention(config)
        self.mlp = ISO20022MLP(config)
        self.input_layernorm = ISO20022RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_attention_layernorm = ISO20022RMSNorm(config.d_model, config.rms_norm_eps)
        self.engram = EngramMemory(config) if use_engram else None
        # A learned scalar gate on the attention residual, as Needle uses.
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """`tokens` is threaded through because the engram hashes token ids.

        An earlier version stashed the ids in a module global for the duration of
        one forward pass. That is wrong for two reasons: gradient checkpointing
        re-runs the layer later, by which time the global has been cleared, and
        two concurrent forwards silently overwrite each other's ids.
        """
        if self.engram is not None:
            hidden = self.engram(tokens, hidden)
        skip = hidden
        attended = self.self_attn(self.input_layernorm(hidden), cos, sin)
        hidden = skip + torch.sigmoid(self.attn_gate).to(hidden.dtype) * attended
        skip = hidden
        hidden = skip + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class ConfidenceHead(nn.Module):
    """Predicts whether the model's answer for this input is correct.

    Needle pools hidden states through learned probes. The pooling matters: the
    signal for "was this answer right" is spread across the whole sequence, not
    concentrated at the last token, and reading only the final position throws
    away the input span the answer depends on.

    Attached for training as a binary classifier over whole documents. At
    inference its score is what makes selective prediction possible -- route the
    low-confidence documents to a human and the high-confidence ones straight
    through.
    """

    def __init__(self, config: ISO20022Config):
        super().__init__()
        self.probes = nn.Parameter(torch.empty(config.confidence_probes, config.d_model))
        nn.init.normal_(self.probes, mean=0.0, std=config.initializer_range)
        self.proj = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map (B, T, D) hidden states to a (B,) correctness logit."""
        scale = 1.0 / math.sqrt(hidden.shape[-1])
        weights = (
            torch.einsum(
                "btd,pd->bpt",
                F.normalize(hidden.float(), dim=-1),
                F.normalize(self.probes.float(), dim=-1),
            )
            * scale
        )
        pooled = torch.einsum("bpt,btd->bpd", weights.softmax(dim=-1), hidden.float())
        pooled = pooled.mean(dim=1)
        return self.proj(pooled.to(hidden.dtype)).squeeze(-1)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass
class ISO20022CausalLMOutput(CausalLMOutputWithPast):
    """Causal LM output plus the confidence logit.

    A plain dict is not enough: transformers' `generate()` does
    `outputs.logits`, so returning a dict makes model.generate() fail with
    "'dict' object has no attribute 'logits'" on a checkpoint that loaded
    perfectly. Subclassing the standard output type keeps attribute access for
    transformers while carrying the confidence head's score through the same
    return value.
    """

    confidence_logit: torch.Tensor | None = None


class ISO20022PreTrainedModel(PreTrainedModel):
    """Base wiring config, initialisation and HuggingFace integration."""

    config_class = ISO20022Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ISO20022DecoderLayer"]
    # transformers 5.x types this as a {target: source} mapping, not a list, so an
    # empty list here fails to override the base annotation.
    _tied_weights_keys: dict[str, str] = {}

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, EngramMemory):
            # The engram is handled here, in the same pass that initialises
            # everything else. An earlier version re-applied the scaled value
            # init after post_init(), which reintroduced exactly the ordering
            # hazard the scaled init exists to avoid.
            nn.init.normal_(module.tables, mean=0.0, std=std)
            nn.init.normal_(module.key_proj.weight, mean=0.0, std=std)
            nn.init.normal_(
                module.value_proj.weight,
                mean=0.0,
                std=std / math.sqrt(2 * self.config.num_layers),
            )
            with torch.no_grad():
                module.taps.zero_()
                module.taps[0] = 1.0
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        # RMSNorm weights and the attention gate stay at their zero init.


class ISO20022Model(ISO20022PreTrainedModel):
    """The transformer body, without a language-modelling head."""

    def __init__(self, config: ISO20022Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        engram_layers = set(config.engram_layers) if config.use_engram else set()
        self.layers = nn.ModuleList(
            [
                ISO20022DecoderLayer(config, use_engram=index in engram_layers)
                for index in range(config.num_layers)
            ]
        )
        self.norm = ISO20022RMSNorm(config.d_model, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = attention_mask, kwargs
        batch, seq = input_ids.shape
        if seq > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        hidden = self.embed_tokens(input_ids)
        positions = (
            torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        )
        cos, sin = self.rotary(positions, hidden.dtype)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = _checkpoint(
                    layer, hidden, cos, sin, input_ids, use_reentrant=False
                )
            else:
                hidden = layer(hidden, cos, sin, input_ids)
        return self.norm(hidden)


class ISO20022ForCausalLM(ISO20022PreTrainedModel, GenerationMixin):
    """The model with a tied language-modelling head and an optional confidence head.

    GenerationMixin is listed explicitly, after PreTrainedModel. transformers warns
    that PreTrainedModel will stop providing `generate` in a future release, and
    without this a Hub user's first `model.generate(...)` call would break on a
    checkpoint that loads perfectly. Declaring it now also silences the warning the
    loader emits for every model that defines `prepare_inputs_for_generation`.
    """

    # transformers 5.x expects {"target": "source"} and save_pretrained needs it to
    # know which keys are the same tensor. Without it, saving raises "shared tensors
    # ... which are not properly defined", because the tying is real but undeclared.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ISO20022Config):
        super().__init__(config)
        self.config = config
        self.model = ISO20022Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.confidence_head = (
            ConfidenceHead(config) if config.use_confidence_head else None
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        confidence_targets: torch.Tensor | None = None,
        confidence_weight: float = 0.1,
        **kwargs: Any,
    ) -> ISO20022CausalLMOutput:
        _ = kwargs
        hidden = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        loss: torch.Tensor | None = None
        if labels is not None:
            # Shift so position t predicts t+1, and ignore padding so a padded
            # batch does not train the model to emit pad tokens.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                # config.ignore_index, NOT pad_token_id. The labels use -100 for
                # prompt and pad positions, and pad_token_id is the real vocabulary
                # index 0, so passing it here made cross_entropy treat every -100 as
                # a class and trip the t >= 0 && t < n_classes device assertion.
                ignore_index=self.config.ignore_index,
            )

        confidence_logit: torch.Tensor | None = None
        if self.confidence_head is not None:
            confidence_logit = self.confidence_head(hidden)
            if confidence_targets is not None and confidence_logit is not None:
                conf_loss = F.binary_cross_entropy_with_logits(
                    confidence_logit, confidence_targets.float()
                )
                loss = (
                    conf_loss * confidence_weight
                    if loss is None
                    else loss + conf_loss * confidence_weight
                )

        return ISO20022CausalLMOutput(
            # CausalLMOutputWithPast narrows `loss` to FloatTensor while
            # cross_entropy is typed as returning a bare Tensor. The cast records
            # that these are the same thing rather than widening the output type,
            # which would lose the guarantee for every consumer.
            loss=cast("torch.FloatTensor | None", loss),
            logits=logits,
            confidence_logit=confidence_logit,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: int | None = None,
        past_key_values: Any = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return only `input_ids`.

        The signature mirrors GenerationMixin's exactly, parameter for parameter.
        A narrower one is rejected as an incompatible override, and the check
        compares NAMES as well as count, so these cannot be renamed or reordered.
        `cache_position` is not a named parameter in this version; it arrives
        through kwargs.

        This model recomputes its full window every step instead of carrying a KV
        cache, so there is no past state, mask or length to thread through. The
        unused parameters are ignored deliberately.
        """
        _ = (
            next_sequence_length,
            past_key_values,
            attention_mask,
            inputs_embeds,
            is_first_iteration,
            kwargs,
        )
        return {"input_ids": input_ids}

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        end_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy decoding, stopping on either the sequence or the answer terminator.

        Deliberately not `model.generate`: extraction is deterministic, so
        sampling adds variance with no benefit, and a hand-rolled loop is easier
        to audit than `GenerationMixin` for a model this small.

        Two stop ids, not one, because they are genuinely different tokens here.
        `<eos>` (id 2) ends a *sequence* -- padding and document boundaries. The
        answer terminator is `<|end|>` (id 6), which is what training puts after
        the JSON target. Stopping only on `<eos>` means generation never halts on
        a correct answer: it runs to `max_new_tokens` emitting text conditioned on
        its own committed output, which wastes roughly 40% of decode compute and
        nondeterministically corrupts the parse when the trailing noise contains a
        brace. That is not hypothetical -- it is what this model did before the
        `end_token_id` field existed.

        The id comes from the config rather than the tokenizer module on purpose:
        this file ships for `trust_remote_code=True` and must not import the
        surrounding package.
        """
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        end = (
            getattr(self.config, "end_token_id", None)
            if end_token_id is None
            else end_token_id
        )
        stops = {t for t in (eos, end) if t is not None}
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.max_position_embeddings :]
            outputs = self.forward(input_ids=window)
            logits = outputs.logits
            if logits is None:
                raise RuntimeError("forward returned no logits; cannot decode greedily")
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if next_token.numel() and int(next_token.flatten()[0]) in stops:
                break
        if was_training:
            self.train()
        return input_ids


def parameter_count(model: nn.Module) -> int:
    """Live parameter count, used to verify the config's arithmetic."""
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "ConfidenceHead",
    "EngramMemory",
    "ISO20022Attention",
    "ISO20022DecoderLayer",
    "ISO20022ForCausalLM",
    "ISO20022CausalLMOutput",
    "ISO20022MLP",
    "ISO20022Model",
    "ISO20022PreTrainedModel",
    "ISO20022RMSNorm",
    "RotaryEmbedding",
    "engram_indices",
    "parameter_count",
]
