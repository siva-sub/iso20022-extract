"""Document fixtures shared by the capture tests.

Deliberately *not* named `test_*.py`, so pytest does not collect it as a test
module -- it has no tests, and collecting it would report an empty module.

These live here rather than in one of the two test files because both need them.
`test_documents.py` uses them to assert route derivation; `test_recipes.py` uses
them so R12 can run the same code path the recipe book prints. Duplicating a PDF
writer into the second file is how the two would drift into disagreeing about
what a "scanned PDF" is, which is the one fixture the whole routing argument
rests on.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

DOC = (
    "Beneficiary: Orchid Pharma BV\n"
    "Beneficiary IBAN: DE80716202183220023878\n"
    "Beneficiary BIC: DEUTDEFF698\n"
    "Amount: EUR 1.250,00\n"
    "Payer: Acme GmbH\n"
    "Payer IBAN: NL9439110653356593\n"
)

# A text-layer PDF is a small object graph, and PDF is a text format, so the
# fixture is ~20 lines rather than a dependency. Helvetica is one of the base-14
# fonts every reader has, so nothing needs embedding.
_PDF_OBJECTS = 5


def render_doc(path: Path, text: str = DOC, size: tuple[int, int] = (1000, 360)) -> Path:
    """A raster image of a payment instruction, the way a fax arrives."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in text.split("\n"):
        draw.text((24, y), line, fill="black")
        y += 38
    image.save(path)
    return path


def scan_pdf(path: Path, text: str = DOC) -> Path:
    """An image-only PDF: pixels with no text layer.

    Written with PIL, which embeds the image and no font program, so the result
    is exactly the case the routing table calls "PDF, scanned". This is the
    fixture that is why the scanned branch was never exercised before.
    """
    image = Image.new("RGB", (1000, 360), "white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in text.split("\n"):
        draw.text((24, y), line, fill="black")
        y += 38
    image.save(path, "PDF")
    return path


def text_pdf(path: Path, text: str = DOC) -> Path:
    """A PDF with a real embedded text layer, hand-built."""
    lines = text.split("\n")
    ops = ["BT /F1 12 Tf 40 320 Td"]
    for i, line in enumerate(lines):
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"({safe}) Tj" if i == 0 else f"0 -20 Td ({safe}) Tj")
    ops.append("ET")
    body = " ".join(ops).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 400] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    blob = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(blob))
        blob += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(blob)
    blob += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        blob += f"{off:010d} 00000 n \n".encode()
    blob += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref).encode() + b"\n%%EOF\n"
    )
    path.write_bytes(blob)
    return path
