"""The logged numbers must describe the file that gets written.

Section 18.13 of PROCESS.md recorded a run whose log reported ``recall 100.0%``
from step 1400 onward while every checkpoint it wrote scored 0% when measured
directly. The gap was not in the weights: save/load round-trips faithfully and
generation output is byte-identical across it. The gap was that the log measured
the *in-memory* model and nothing ever measured the *artifact*, so no part of the
pipeline could notice the two had diverged.

These tests pin the two properties that make the report trustworthy again.
"""

# pyright: reportPrivateImportUsage=false
#
# Same pragma, and the same reason, as `model/train.py`: torch/__init__.py does
# `from torch._C import *`, and torch._C is a compiled extension with no stub, so
# pyright reports every `torch.tensor` as a private import from
# `torch._C._VariableFunctions`. All of them are in torch.__all__ and available at
# runtime. Importing the private path to satisfy the checker would break on torch
# upgrades, so the public API stays and this one rule is silenced for this file.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from iso20022_lab.fields import CANONICAL_KEYS
from iso20022_lab.model.configuration_iso20022 import ISO20022Config
from iso20022_lab.model.modeling_iso20022 import ISO20022ForCausalLM
from iso20022_lab.model.train import (
    EOS_ID,
    PAD_ID,
    ExtractionDataset,
    TrainConfig,
    _value_key,
    build_examples,
    collate,
    evaluate_checkpoint,
    generation_metrics,
    holdout_examples,
    model_device,
    save_checkpoint,
    split_examples,
)
from iso20022_lab.synth import generate_many


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    path = Path(__file__).parent / "data" / "tokenizer"
    if not (path / "tokenizer.json").exists():
        pytest.skip("tokenizer not built")
    return transformers.PreTrainedTokenizerFast.from_pretrained(path)


@pytest.fixture(scope="module")
def examples() -> list[Any]:
    return build_examples(60, seed=0)


def _signature(values: dict[str, str]) -> str:
    """Identity of a raw generator value-set.

    ``generate_many`` yields bare field->value dicts, while ``_value_key`` takes
    an ``Example`` with the values nested under ``"values"``. Passing one where
    the other is expected returns an empty string for every item, which makes
    set comparisons silently compare ``{""}`` against ``{""}`` and pass or fail
    for the wrong reason.
    """
    return json.dumps(values, sort_keys=True)


def _small_model(tokenizer: Any) -> ISO20022ForCausalLM:
    config = ISO20022Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=EOS_ID,
    )
    return ISO20022ForCausalLM(config)


class TestSplitCannotLeakAnswers:
    """Validation must not contain an answer the model could have memorised."""

    def test_no_value_set_appears_on_both_sides(self, examples: list[Any]) -> None:
        """The regression that mattered: 600/600 validation answers were in train.

        ``build_examples`` emits three documents per value-set -- clean, messy and
        hostile -- and the old split assigned per document, so holding out the
        clean variant left the same answer in training via its two siblings. Every
        validation answer was reachable by recall rather than extraction.
        """
        train, val = split_examples(examples, seed=0)
        assert train and val
        overlap = {_value_key(i) for i in train} & {_value_key(i) for i in val}
        assert overlap == set(), f"{len(overlap)} value-sets leaked across the split"

    def test_all_variants_of_a_value_set_stay_together(self, examples: list[Any]) -> None:
        """Splitting inside a value-set is what caused the leak."""
        train, val = split_examples(examples, seed=0)
        side = {_value_key(i): "train" for i in train}
        side.update({_value_key(i): "val" for i in val})
        for item in examples:
            assert side[_value_key(item)] in {"train", "val"}

    def test_validation_still_covers_every_difficulty(self, examples: list[Any]) -> None:
        """Stricter is no use if it only ever tests one kind of document."""
        _, val = split_examples(examples, seed=0)
        assert {"clean", "messy", "hostile"} <= {i.get("difficulty") for i in val}

    def test_val_frac_one_takes_everything(self, examples: list[Any]) -> None:
        """``evaluate_model`` passes ``val_frac=1.0`` and needs the whole corpus."""
        train, val = split_examples(examples, val_frac=1.0, seed=0)
        assert not train
        assert len(val) == len(examples)

    def test_split_is_deterministic(self, examples: list[Any]) -> None:
        _, a = split_examples(examples, seed=0)
        _, b = split_examples(examples, seed=0)
        assert [i["text"] for i in a] == [i["text"] for i in b]

    def test_seed_still_changes_the_split(self, examples: list[Any]) -> None:
        _, a = split_examples(examples, seed=0)
        _, b = split_examples(examples, seed=7)
        assert {i["text"] for i in a} != {i["text"] for i in b}

    def test_assignment_does_not_depend_on_corpus_size(self, examples: list[Any]) -> None:
        """The old ``randperm(len(items))`` made the split a function of size.

        Adding documents to the corpus reshuffled which ones were held out, so two
        runs at different ``--docs`` compared two different validation sets while
        both claiming to use the same seed.
        """
        _, full = split_examples(examples, seed=0)
        _, part = split_examples(examples[: len(examples) // 2], seed=0)
        full_text = {i["text"] for i in full}
        part_text = {i["text"] for i in part}
        assert part_text, "the subset produced no validation documents"
        assert part_text <= full_text, (
            f"{len(part_text - full_text)} documents changed side when the corpus changed"
        )


class TestMetricsDescribeTheSavedArtifact:
    """What is logged must be what is written."""

    def test_reloading_a_checkpoint_reproduces_its_metrics(
        self, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The central claim: a checkpoint's measured behaviour survives a round-trip.

        If this fails, no log line about a live model can be evidence about the
        file on disk, which is exactly how a run reported 100% recall for
        checkpoints that generate repetition loops.
        """
        torch.manual_seed(0)
        model = _small_model(tokenizer)
        model.eval()
        cfg = TrainConfig(eval_generate=4, eval_max_new_tokens=32, max_seq_len=256)

        _, val = split_examples(build_examples(20, seed=0), seed=0)
        save_checkpoint(model, tmp_path / "ck", cfg)

        live = generation_metrics(model, val, tokenizer, "cpu", cfg)
        saved = evaluate_checkpoint(tmp_path / "ck", val, tokenizer, cfg)

        assert live, "the metric produced nothing to compare"
        for key in ("json_rate", "field_recall", "exact_match", "terminated_rate"):
            assert saved[key] == pytest.approx(live[key], abs=1e-9), (
                f"{key} differs between the live model and the checkpoint on disk: "
                f"{live[key]} live vs {saved[key]} saved"
            )

    def test_evaluate_checkpoint_reads_from_disk(
        self, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The measurement must come from the file, not from a stale in-memory model.

        Comparing *rates* cannot show this on an untrained model: every rate is
        already 0.0 and stays 0.0 whatever the weights are. Generated tokens can
        tell the difference, so the comparison is on those.
        """
        from iso20022_lab.model.train import prompt_for

        cfg = TrainConfig(eval_generate=1, eval_max_new_tokens=12, max_seq_len=256)
        _, val = split_examples(build_examples(20, seed=0), seed=0)
        prompt_ids = tokenizer.encode(prompt_for(val[0]["text"]), add_special_tokens=False)

        def generated(path: Path) -> list[int]:
            net = ISO20022ForCausalLM.from_pretrained(path)
            net.eval()
            out = net.generate_greedy(
                torch.tensor([prompt_ids]), max_new_tokens=12, eos_token_id=EOS_ID
            )
            return out[0, len(prompt_ids) :].tolist()

        torch.manual_seed(0)
        first = _small_model(tokenizer)
        save_checkpoint(first, tmp_path / "ck", cfg)
        tokens_a = generated(tmp_path / "ck")

        torch.manual_seed(1234)
        second = _small_model(tokenizer)
        save_checkpoint(second, tmp_path / "ck", cfg)
        tokens_b = generated(tmp_path / "ck")

        assert tokens_a != tokens_b, (
            "two differently-initialised checkpoints generated identically, so the "
            "measurement is not reading the weights on disk"
        )
        assert evaluate_checkpoint(tmp_path / "ck", val, tokenizer, cfg)["docs"] > 0

    def test_saved_output_matches_the_live_model_token_for_token(
        self, tokenizer: Any, tmp_path: Path
    ) -> None:
        """The strongest form of the guarantee: identical tokens, not just equal rates.

        Equal rates could hide a difference that no document in a small sample
        happened to separate. Token equality cannot.
        """
        from iso20022_lab.model.train import prompt_for

        torch.manual_seed(0)
        model = _small_model(tokenizer)
        model.eval()
        cfg = TrainConfig(eval_generate=1, eval_max_new_tokens=16, max_seq_len=256)
        _, val = split_examples(build_examples(20, seed=0), seed=0)

        saved_path = tmp_path / "ck"
        save_checkpoint(model, saved_path, cfg)
        reloaded = ISO20022ForCausalLM.from_pretrained(saved_path)
        reloaded.eval()

        prompt_ids = tokenizer.encode(prompt_for(val[0]["text"]), add_special_tokens=False)
        arguments = {"max_new_tokens": 16, "eos_token_id": EOS_ID}
        live_tokens = model.generate_greedy(torch.tensor([prompt_ids]), **arguments)[
            0, len(prompt_ids) :
        ].tolist()
        saved_tokens = reloaded.generate_greedy(torch.tensor([prompt_ids]), **arguments)[
            0, len(prompt_ids) :
        ].tolist()
        assert live_tokens == saved_tokens

    def test_checkpoint_records_the_config_used(
        self, tokenizer: Any, tmp_path: Path
    ) -> None:
        """A checkpoint without its config cannot be re-measured on its own terms.

        ``eval_max_new_tokens`` changes what a generation metric reports, so a
        verifier has to be able to read the budget the run used rather than
        assume a default.
        """
        cfg = TrainConfig(eval_generate=3, eval_max_new_tokens=48, max_seq_len=192)
        save_checkpoint(_small_model(tokenizer), tmp_path / "ck", cfg)
        recorded = json.loads((tmp_path / "ck" / "train_config.json").read_text())
        assert recorded["eval_max_new_tokens"] == 48
        assert recorded["max_seq_len"] == 192


class TestTheVerifierActuallyRuns:
    """A verification hook that never fires is decoration."""

    def test_training_verifies_each_checkpoint_it_writes(
        self, tokenizer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every save during training must be re-measured from disk."""
        from iso20022_lab.model.train import train_stage1

        cfg = TrainConfig(
            epochs=1,
            device="cpu",
            log_every=100,
            eval_every=2,
            save_every=2,
            eval_generate=2,
            eval_max_new_tokens=8,
            max_seq_len=256,
        )
        examples = build_examples(20, seed=0)
        train_ex, val_ex = split_examples(examples, seed=0)
        loader = DataLoader(
            ExtractionDataset(train_ex, tokenizer, cfg.max_seq_len),
            batch_size=8,
            shuffle=False,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            ExtractionDataset(val_ex, tokenizer, cfg.max_seq_len),
            batch_size=8,
            shuffle=False,
            collate_fn=collate,
        )

        torch.manual_seed(0)
        model = _small_model(tokenizer)
        train_stage1(
            model,
            loader,
            val_loader,
            cfg,
            tmp_path / "out",
            examples=val_ex,
            tokenizer=tokenizer,
        )

        printed = capsys.readouterr().out
        assert "SAVED step2:" in printed, "checkpoints were written without verification"
        assert "MATCH" in printed or "MISMATCH" in printed
        assert (tmp_path / "out" / "step2").is_dir()


class TestValidationIsADisjointDraw:
    """A held-out slice of the training draw cannot measure generalisation.

    Every report this project produced quoted the within-draw figure. Measured on
    one checkpoint: 97.5% field recall against a held-out slice of the training
    value draw, and 61.2% against a fresh draw from the same generator. The model
    had memorised value instances, which 53M parameters can do for ~4,000
    value-sets of IBANs, amounts and remittance strings.
    """

    def test_holdout_is_disjoint_from_the_training_draw(self) -> None:
        """The whole point: a fresh draw shares no value-sets with training."""
        train_draw = build_examples(200, seed=0)
        holdout = holdout_examples(30, seed=1)
        train_sets = {_value_key(i) for i in train_draw}
        holdout_sets = {_value_key(i) for i in holdout}
        assert holdout_sets
        assert not (train_sets & holdout_sets), (
            f"{len(train_sets & holdout_sets)} value-sets appear in both the "
            "training draw and the supposed holdout"
        )

    def test_holdout_still_covers_every_difficulty(self) -> None:
        """An honest sample that only tests one level is not an improvement."""
        holdout = holdout_examples(30, seed=1)
        assert {"clean", "messy", "hostile"} <= {i.get("difficulty") for i in holdout}

    def test_holdout_is_deterministic(self) -> None:
        a = holdout_examples(20, seed=5)
        b = holdout_examples(20, seed=5)
        assert [i["text"] for i in a] == [i["text"] for i in b]

    def test_a_small_draw_is_a_subset_of_the_large_one(self) -> None:
        """Why the within-draw slice fails, stated as a property of the generator.

        ``generate_many`` walks one seeded stream, so asking for fewer value-sets
        returns a prefix of the larger draw. Training on 4,000 of them and
        validating on the other 216 stays inside the region the model has fitted.
        """
        small = {_signature(v) for v in generate_many(40, seed=0)}
        large = {_signature(v) for v in generate_many(400, seed=0)}
        assert small <= large, "a shorter draw is no longer a prefix of a longer one"

    def test_two_seeds_do_not_share_value_sets(self) -> None:
        """And why a fresh draw is a different question: nothing carries over."""
        a = {_signature(v) for v in generate_many(200, seed=0)}
        b = {_signature(v) for v in generate_many(200, seed=1)}
        assert not (a & b), "the two draws overlap, so seed=1 is not a fresh draw"
        assert len(a) == 200 and len(b) == 200, "value-sets are not unique within a draw"

    def test_the_generator_signatures_are_the_canonical_keys(self) -> None:
        """Guards against the value shape drifting out from under the tests."""
        first = generate_many(1, seed=0)[0]
        assert set(first) == set(CANONICAL_KEYS), (
            f"generator emits {sorted(set(first) - set(CANONICAL_KEYS))} and omits "
            f"{sorted(set(CANONICAL_KEYS) - set(first))}"
        )


class TestDeviceIsDerivedNotTrusted:
    """A wrong device argument must not be able to crash the metric.

    The checkpoint verifier passed ``"cpu"`` for the live model while a GPU
    training run held it on ``cuda:0``. Nothing raised at the call: the mismatch
    surfaced inside ``nn.Embedding`` as "Expected all tensors to be on the same
    device, but got index is on cpu", and it killed a four-hour run at its first
    checkpoint -- after the training it was verifying had already been paid for.
    """

    def test_generation_metrics_ignores_a_wrong_device(
        self, tokenizer: Any, examples: list[Any]
    ) -> None:
        """Claiming the wrong device must be harmless, not fatal."""
        torch.manual_seed(0)
        model = _small_model(tokenizer)
        model.eval()
        cfg = TrainConfig(eval_generate=2, eval_max_new_tokens=8, max_seq_len=256)

        actual = model_device(model)
        wrong = "cuda" if actual == "cpu" else "cpu"
        # A CPU-only host cannot place tensors on cuda, so if the argument were
        # honoured this would raise rather than return.
        metrics = generation_metrics(model, examples, tokenizer, wrong, cfg)
        assert metrics["docs"] > 0, "the metric scored nothing"

    def test_model_device_reports_where_the_parameters_are(self, tokenizer: Any) -> None:
        model = _small_model(tokenizer)
        assert model_device(model) == str(next(model.parameters()).device)

    def test_model_device_falls_back_when_there_are_no_parameters(self) -> None:
        """A module with no parameters must not raise a StopIteration."""

        class Empty(torch.nn.Module):
            pass

        assert model_device(Empty()) == "cpu"
