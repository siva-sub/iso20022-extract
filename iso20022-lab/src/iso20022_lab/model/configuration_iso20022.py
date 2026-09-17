"""Configuration for the ISO 20022 extraction model.

Shaped after Needle (Cactus Compute, 45M parameters): a small decoder-only
transformer with grouped-query attention, RoPE, pre-norm RMSNorm, a hashed
n-gram memory ("engram"), a correctness confidence head, and multi-token
prediction. Needle is built in JAX/Flax; this is a PyTorch port, because the
model has to be shareable on HuggingFace and the `transformers` ecosystem is
PyTorch-first. Users get `AutoModel.from_pretrained(...)`, not a bespoke loader.

What was taken from Needle, and why each one earns its place here:

* **Engram.** A hashed n-gram table supplying extra key/value pairs to attention
  at chosen layers. Payment documents are dense in repeated short n-grams --
  currency codes, BIC shapes, IBAN country prefixes, XML element names -- so
  this gives associative recall of exactly the tokens the task turns on, instead
  of spending attention capacity re-deriving them at every position.
* **Confidence head.** Needle pools hidden states through learned probes into a
  single correctness logit. PROCESS.md 2.4 removed the confidence gate for tuned
  weights and 16.3 identified hallucination as the decisive risk, which makes a
  calibrated per-document correctness score the capability this project is
  actually missing. Without it the pipeline cannot be safely automated.
* **Multi-token prediction.** A second lightweight block predicting the token
  after next. Cheap, and it densifies the training signal on short sequences --
  which matters because a JSON extraction target is only ~120 tokens.

Deviation: **the vocabulary is trained, not inherited.** Needle's 8k vocab is
tuned for function calling. ISO 20022 has its own distribution, so the tokenizer
is trained on the corpus rather than inherited.

Not taken from DSpark: speculative decoding. DSpark's gain is throughput when a
large target model makes each verification pass expensive. At 45M parameters
producing ~120 tokens per document, the draft-and-verify overhead exceeds the
saving, and there is no throughput problem to solve at any plausible volume --
100k documents/month is ~2.3 per minute. Its *confidence-scheduling* idea is
good but is already served by the confidence head above, which is cheaper and
directly serves the safety case.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class ISO20022Config(PretrainedConfig):
    """Config for a small ISO 20022 extraction model.

    Defaults target roughly 45M parameters at a 512-wide, 20-layer body.
    """

    model_type = "iso20022"

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 512,
        num_layers: int = 20,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        d_ff: int = 768,
        max_position_embeddings: int = 1024,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        # --- engram: hashed n-gram memory ---
        use_engram: bool = True,
        engram_orders: tuple[int, ...] = (2, 3),
        engram_slots: int = 8192,
        engram_layers: tuple[int, ...] = (2, 12),
        engram_sub_dim: int = 128,
        engram_conv_taps: int = 4,
        # --- confidence head: selective prediction ---
        use_confidence_head: bool = True,
        confidence_probes: int = 8,
        # --- multi-token prediction ---
        use_mtp: bool = False,
        # The label value that contributes no loss. -100 is PyTorch's
        # cross_entropy default. It is NOT pad_token_id: pad_token_id is a real
        # vocabulary index (0), so using it as the ignore index makes cross_entropy
        # treat every -100 as a class and fail the t >= 0 && t < n_classes
        # assertion. Kept in the config so the model and the data pipeline cannot
        # drift apart on it.
        ignore_index: int = -100,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # The token that ends an *answer*, distinct from the one that ends a
        # sequence. `<eos>` (2) marks padding and document boundaries; `<|end|>`
        # (6) is what training puts after the JSON target. Decoding must stop on
        # both -- stopping only on `eos_token_id` means a correct answer never
        # terminates generation, because `<eos>` is not the token the model emits
        # there. Defaulted here rather than imported from the tokenizer module
        # because this file ships for `trust_remote_code=True` and must stand
        # alone.
        end_token_id: int = 6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout

        self.use_engram = use_engram
        self.engram_orders = tuple(engram_orders)
        self.engram_slots = engram_slots
        self.engram_layers = tuple(engram_layers)
        self.engram_sub_dim = engram_sub_dim
        self.engram_conv_taps = engram_conv_taps

        self.use_confidence_head = use_confidence_head
        self.confidence_probes = confidence_probes

        self.use_mtp = use_mtp
        self.ignore_index = ignore_index

        # transformers' GenerationMixin, DynamicCache and several utilities read the
        # conventional attribute names, not these. Without the aliases,
        # model.generate() raises AttributeError('num_hidden_layers') on a
        # checkpoint that loaded perfectly -- so every Hub user hits it on their
        # first call. Real attributes rather than properties, so they serialise
        # into config.json and stay assignable like any other config field.
        self.hidden_size = d_model
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.intermediate_size = d_ff

        # transformers 5.x moved the special-token ids and tying out of
        # PretrainedConfig.__init__ into plain attributes, so they are set here
        # before the super() call that serialises the config.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.end_token_id = end_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Required for the Hub artifact to be loadable at all. Without this map,
        # AutoConfig/AutoModel look 'iso20022' up in transformers' built-in
        # registries, fail with KeyError('iso20022'), and never load the code
        # shipped alongside the weights -- even with trust_remote_code=True. The
        # paths are relative to the repository root, which is why the export
        # copies configuration_iso20022.py and modeling_iso20022.py next to the
        # weights.
        self.auto_map = {
            "AutoConfig": "configuration_iso20022.ISO20022Config",
            "AutoModel": "modeling_iso20022.ISO20022Model",
            "AutoModelForCausalLM": "modeling_iso20022.ISO20022ForCausalLM",
        }
        super().__init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def engram_geometry(self) -> tuple[int, int]:
        """(heads, sub_dim) for the engram table.

        Mirrors Needle: the width is split across orders and heads, so adding an
        order costs table entries rather than widening the model.
        """
        n_orders = len(self.engram_orders)
        heads = max(1, self.d_model // (n_orders * self.engram_sub_dim))
        sub_dim = self.d_model // (n_orders * heads)
        return heads, sub_dim

    def parameter_count(self) -> dict[str, int]:
        """Exact parameter count, computed rather than claimed.

        An earlier attempt to match Needle's stated 45M used a 4x wider FFN and
        landed at 70M while still calling itself 45M. Reporting the arithmetic is
        the only way that number stays honest.
        """
        embed = self.vocab_size * self.d_model
        q = self.d_model * self.d_model
        kv = self.d_model * self.num_kv_heads * self.head_dim
        out = self.d_model * self.d_model
        attn = q + 2 * kv + out
        ffn = 3 * self.d_model * self.d_ff  # SwiGLU: gate, up, down
        norms = 2 * self.d_model
        # +1 for the learned scalar gate on the attention residual.
        per_layer = attn + ffn + norms + 1
        body = per_layer * self.num_layers
        final_norm = self.d_model

        engram = 0
        if self.use_engram:
            heads, sub_dim = self.engram_geometry()
            n_tables = len(self.engram_orders) * heads
            width = n_tables * sub_dim
            # tables + the key/value projections over the fetched n-grams + the
            # dilated temporal taps. Omitting the two projections understated the
            # total by 1.05M and made the declared figure disagree with the model.
            per_engram = (
                n_tables * self.engram_slots * sub_dim
                + 2 * width * self.d_model
                + self.engram_conv_taps * self.d_model
            )
            engram = len(self.engram_layers) * per_engram

        confidence = 0
        if self.use_confidence_head:
            confidence = self.confidence_probes * self.d_model + self.d_model + 1

        mtp = 0
        if self.use_mtp:
            mtp = self.d_model * 2 * self.d_model + per_layer

        untied_lm_head = 0 if self.tie_word_embeddings else embed
        total = embed + body + final_norm + engram + confidence + mtp + untied_lm_head
        return {
            "embedding": embed,
            "per_layer": per_layer,
            "body": body,
            "final_norm": final_norm,
            "engram": engram,
            "confidence_head": confidence,
            "mtp": mtp,
            "lm_head_extra": untied_lm_head,
            "total": total,
        }


__all__ = ["ISO20022Config"]
