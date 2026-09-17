"""Repository-relative paths, resolved from this file and not from the CWD.

WHY THIS EXISTS

Every data path in this package used to be a bare relative string:

    DEFAULT_OUT = Path("iso20022-lab/data/model")
    tokenizer_path = "iso20022-lab/data/tokenizer"

Those are written as if the process always starts at the repository root. It does
not. Running any of these modules from `iso20022-lab/` -- which is the natural
thing to do, because that is where the package's own tests live -- resolves them
against the wrong directory, and the result depends on where you happened to be
standing.

The failures this caused were not crashes, which is what makes it worth fixing
properly rather than case by case:

- `model.export` looked for `train_report.json` at a path that did not exist,
  `_load_json` returned `{}` for the missing file, and the **model card was
  written claiming zero training documents and no validation history**. That card
  is what would have been uploaded to the Hub. A missing input silently became a
  confident wrong number in published documentation.
- `model.train` could not find the tokenizer and refused to start, which is at
  least honest, but it made the module unusable except from one directory.

Deriving the layout from `__file__` means every module agrees on where the data
is, from any working directory, with no environment variable and no install step.
This mirrors `fields.py`: a small module at the bottom of the dependency graph
that imports nothing from the package, so it cannot participate in a cycle.

LAYOUT

    <repo>/iso20022-lab/            LAB_DIR
        data/                       DATA_DIR
            model/                  trained checkpoints and the export
            tokenizer/              the trained vocabulary
            distill/  mt/  dual/    corpora
        schemas/                    XSDs
        src/iso20022_lab/           this package
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives at <root>/iso20022-lab/src/iso20022_lab/paths.py, so the package
# is parents[0], src is parents[1], and iso20022-lab is parents[2].
LAB_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = LAB_DIR / "data"
SCHEMA_DIR: Path = LAB_DIR / "schemas"
SRC_DIR: Path = LAB_DIR / "src"


def data_path(*parts: str) -> Path:
    """A path under `data/`, e.g. `data_path("model", "final")`."""
    return DATA_DIR.joinpath(*parts)


def schema_path(name: str = "pain.001.001.09.xsd") -> Path:
    """A path under `schemas/`."""
    return SCHEMA_DIR / name


def repo_root() -> Path:
    """The directory containing `iso20022-lab/`."""
    return LAB_DIR.parent


def describe() -> str:
    """Human-readable resolution, for the `--help` of anything that takes paths."""
    return (
        f"  package     {Path(__file__).resolve().parent}\n"
        f"  lab dir     {LAB_DIR}\n"
        f"  data dir    {DATA_DIR}\n"
        f"  schema dir  {SCHEMA_DIR}"
    )


def main() -> int:
    """Print the resolved layout, so a path problem is one command to diagnose."""
    print(describe())
    for label, path in (
        ("tokenizer", data_path("tokenizer")),
        ("model/final", data_path("model", "final")),
        ("train_report", data_path("model", "train_report.json")),
        ("schema", schema_path()),
    ):
        print(f"  {label:<14} {'OK ' if path.exists() else 'MISSING'}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
