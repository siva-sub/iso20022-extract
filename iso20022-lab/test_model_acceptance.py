"""CLI wrapper for the model acceptance gate.

The gate itself lives in `iso20022_lab.acceptance`, not here. This file only
parses arguments, loads the artifact, runs the gate, and sets an exit code.

That split is deliberate, and it replaced a worse arrangement. The first version
defined the checks, the thresholds and the acceptance logic in this file, and the
recipes imported them from here -- which worked only while the test directory
happened to be on the type-checker's path, and which meant `acceptance.py` ended
up holding a second copy of `CANONICAL_KEYS`. Two definitions of the field set in
a gate whose entire purpose is to be unambiguous is a bug waiting for a chance to
happen. There is now one definition, in the library, and both this CLI and the
recipe tests import it.

    python -m test_model_acceptance --docs 30 --seed 31

Exit code is 0 only if every check passes. That is the point: this is meant to
run as a publish gate, so `&& python -m ... export --push` is a safe pipeline and
a silent pass is impossible.

Run it from `iso20022-lab/`, or from anywhere -- the default paths resolve against
this file rather than the working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from iso20022_lab.acceptance import evaluate, roundtrip_check

_HERE = Path(__file__).resolve().parent
XSD_PATH = _HERE / "schemas" / "pain.001.001.09.xsd"
DEFAULT_MODEL = _HERE / "data" / "model" / "final"
DEFAULT_TOKENIZER = _HERE / "data" / "tokenizer"


def main() -> int:
    parser = argparse.ArgumentParser(description="model acceptance gate")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--docs", type=int, default=30)
    parser.add_argument(
        "--seed",
        type=int,
        default=31,
        help="must differ from the training seed, or the documents are not held out",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="skip C7; useful while a checkpoint is still training",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"no model at {args.model}")
        print("train first, or export a checkpoint into a loadable artifact with")
        print("  python -m iso20022_lab.model.export --model <dir>")
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iso20022_lab.model.train import build_examples

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    mdl: Any = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        torch.nn.Module.cuda(mdl)
    mdl.eval()

    # Held out: a seed the training run never saw, so these layouts are new.
    examples = build_examples(args.docs, seed=args.seed)
    result = evaluate(mdl, tok, examples, device=device, limit=args.docs)

    # C7 runs through the workflow harness, which needs corpus Documents (it
    # scores against `truth` and builds a real message) rather than the bare
    # examples the other checks use. It also loads the model a second time.
    if not args.skip_roundtrip:
        from iso20022_lab.corpus import build_corpus
        from iso20022_lab.synth import generate_many

        corpus = build_corpus(
            generate_many(args.docs, seed=args.seed), per_seed=1, seed=args.seed
        )
        result.checks.append(
            roundtrip_check(corpus.documents, args.model, args.tokenizer, XSD_PATH)
        )

    print(result.render())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
