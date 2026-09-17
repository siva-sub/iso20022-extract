"""Tokenizer for the ISO 20022 extraction model.

Trained rather than inherited. Needle ships an 8k vocab tuned for function
calling; ISO 20022 has a different token distribution -- ISO 4217 currency codes,
ISO 9362 BIC shapes, IBAN country prefixes and check digits, camel-case XML
element names, ISO 8601 dates -- so a general vocab spends many of its tokens on
text this task never sees and fragments the text it always sees.

Sequence format, which is the model's entire interface:

    <|doc|>
    Beneficiary: Acme GmbH
    IBAN: DE89370400440532013000
    ...
    <|json|>{"Cdtr_Nm": "Acme GmbH", ...}<|end|>

The document is context, the JSON is the target. Only tokens after `<|json|>` are
trained on, so the model is never rewarded for reproducing the input -- which is
the failure mode that turns an extraction model into a paraphraser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from iso20022_lab.paths import data_path

SPECIAL_TOKENS: list[str] = [
    "<pad>",  # 0
    "<bos>",  # 1
    "<eos>",  # 2
    "<unk>",  # 3
    "<|doc|>",  # 4
    "<|json|>",  # 5
    "<|end|>",  # 6
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
DOC_ID = 4
JSON_ID = 5
END_ID = 6

# The prefix the model must generate. Everything before it is input.
PROMPT_TEMPLATE = "<|doc|>\n{text}\n<|json|>"
ANSWER_PREFIX = "<|json|>"


def format_example(text: str, values: dict[str, str]) -> str:
    """Render one training sequence: document, then the JSON target."""
    ordered = {k: values[k] for k in sorted(values)}
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return f"<|doc|>\n{text}\n<|json|>{payload}<|end|>"


def prompt_for(text: str) -> str:
    """The inference-time prompt, with no target attached."""
    return PROMPT_TEMPLATE.format(text=text)


def train_tokenizer(
    sequences: list[str],
    vocab_size: int = 8192,
    out_dir: str | Path | None = None,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE tokenizer on extraction sequences.

    Byte-level so no input is ever unrepresentable: an unfamiliar currency
    symbol or a stray UTF-8 sequence becomes bytes rather than `<unk>`, and an
    extraction model that silently drops part of a document is worse than one
    that tokenises it oddly.
    """
    # Plain if/else rather than a ternary. The conditional expression ran to 84
    # characters and mixed two decisions -- where the default lives, and how a
    # caller's argument is honoured -- into one line where neither read clearly.
    if out_dir is None:
        out_path = data_path("tokenizer")
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # Digits and punctuation matter more than prose here: IBANs, amounts and
        # dates are almost entirely non-alphabetic, so showing them early stops
        # the BPE merges from being dominated by words.
        show_progress=False,
    )
    backend.train_from_iterator(sequences, trainer=trainer)
    backend.save(str(out_path / "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(out_path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<|doc|>", "<|json|>", "<|end|>"],
    )
    tokenizer.save_pretrained(str(out_path))
    return tokenizer


def build_sequences(documents: list[Any], limit: int | None = None) -> list[str]:
    """Turn corpus documents into training sequences."""
    sequences: list[str] = []
    for doc in documents[:limit] if limit else documents:
        sequences.append(format_example(doc.text, doc.truth))
    return sequences


def describe(tokenizer: PreTrainedTokenizerFast) -> str:
    """Report how well the vocab fits this domain."""
    probes = {
        "IBAN": "DE89370400440532013000",
        "BIC": "DEUTDEFF",
        "currency": "EUR",
        "date": "2026-09-20",
        "amount": "1,250.00",
        "json key": '"CdtrAcct_IBAN"',
    }
    lines = [f"vocab size: {tokenizer.vocab_size}"]
    for label, text in probes.items():
        pieces = tokenizer.tokenize(text)
        lines.append(f"  {label:<10} {len(pieces):>2} tok  {pieces}")
    return "\n".join(lines)


def main() -> int:
    import argparse

    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.synth import generate_many

    parser = argparse.ArgumentParser(description="train the ISO 20022 tokenizer")
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--docs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default=str(data_path("tokenizer")))
    args = parser.parse_args()

    values = generate_many(args.docs, seed=args.seed)
    corpus = build_corpus(values, per_seed=1, seed=args.seed)
    sequences = build_sequences(corpus.documents)
    print(f"training tokenizer on {len(sequences)} sequences")

    tokenizer = train_tokenizer(sequences, vocab_size=args.vocab, out_dir=args.out)
    print()
    print(describe(tokenizer))
    print()
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
