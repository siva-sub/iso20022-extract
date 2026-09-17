"""Tests for `documents.py`: bytes -> capture() inputs.

These exist because the module closes a hole that no test could previously see.
`ingest.py` routes on `has_text_layer`, and before `documents.py` that flag was
whatever the caller passed. A test of the router could set it either way and pass
in both cases, which proves the router works and says nothing about whether the
flag is ever true.

So the tests here assert the *derivation*: that a scanned PDF really does come
back needing recognition, that a text-layer PDF really does not, and that an
email's attachment is never concatenated into its body.

The type name in every assertion is `Path` rather than `str`. Earlier automation
reported four `str`-where-`Path`-expected errors in this file's helper; keeping
the helper's annotations honest is what makes that class of report checkable
rather than noise.
"""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from iso20022_lab.documents import (
    Extracted,
    channel_for,
    describe,
    extract_bytes,
    extract_email,
    extract_image,
    extract_pdf,
    ingest,
    sniff,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

DOC = (
    "Beneficiary: Orchid Pharma BV\n"
    "Beneficiary IBAN: DE80716202183220023878\n"
    "Beneficiary BIC: DEUTDEFF698\n"
    "Amount: EUR 1.250,00\n"
    "Payer: Acme GmbH\n"
    "Payer IBAN: NL9439110653356593\n"
)


def _render_doc(path: Path, text: str = DOC, size: tuple[int, int] = (1000, 360)) -> Path:
    """A raster image of a payment instruction, the way a fax arrives."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in text.split("\n"):
        draw.text((24, y), line, fill="black")
        y += 38
    image.save(path)
    return path


def _scan_pdf(path: Path, text: str = DOC) -> Path:
    """An image-only PDF: pixels with no text layer.

    Written with PIL, which embeds the image and no font program, so the result
    is exactly the case the routing table calls "PDF, scanned". This is the
    fixture that did not exist before and is the reason the scanned branch was
    never exercised.
    """
    image = Image.new("RGB", (1000, 360), "white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in text.split("\n"):
        draw.text((24, y), line, fill="black")
        y += 38
    image.save(path, "PDF")
    return path


def _text_pdf(path: Path, text: str = DOC) -> Path:
    """A PDF with a real embedded text layer, hand-built.

    PDF is a text format with a small object graph, so a fixture is ~20 lines
    rather than a dependency. Helvetica is one of the base-14 fonts every reader
    has, so nothing needs embedding.
    """
    lines = text.split("\n")
    ops = ["BT /F1 12 Tf 40 320 Td"]
    for i, line in enumerate(lines):
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"({safe}) Tj" if i == 0 else f"0 -20 Td ({safe}) Tj")
    ops.append("ET")
    body = " ".join(ops).encode("latin-1", errors="replace")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 400] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    blob = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(blob))
        blob += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(blob)
    blob += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        blob += f"{off:010d} 00000 n \n".encode()
    blob += (
        b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref).encode() + b"\n%%EOF\n"
    )
    path.write_bytes(blob)
    return path


@pytest.fixture
def work(tmp_path: Path) -> Path:
    return tmp_path


# --------------------------------------------------------------------------
# Sniffing: the declaration is not the evidence
# --------------------------------------------------------------------------


def test_magic_bytes_beat_a_lying_filename() -> None:
    """A PNG named `payment.pdf` must be typed as a PNG.

    The failure this prevents is specific and quiet: routing a raster file to a
    PDF text-layer extractor returns an empty string, which the router then reads
    as "blank document" instead of "wrong parser".
    """
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    assert sniff(png, "payment.pdf", "application/pdf") == "image/png"


def test_magic_bytes_beat_a_lying_mime_type() -> None:
    """Both declarations lie, and the bytes still win."""
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    assert sniff(png, "invoice.pdf", "application/pdf") == "image/png"


def test_real_pdf_is_still_recognised() -> None:
    assert sniff(b"%PDF-1.7\n...") == "application/pdf"


def test_declared_type_is_honoured_when_bytes_are_ambiguous() -> None:
    """Plain bytes with no magic get their declared type.

    The declaration is a fallback, not a competitor. A sniffer that ignored it
    entirely would fail on the formats that genuinely have no signature.
    """
    assert sniff(b"hello there", "note.eml") == "message/rfc822"
    assert sniff(b"plain body text", "", "text/plain") == "text/plain"


def test_html_email_body_is_typed_as_text() -> None:
    assert sniff(b"<html><body>Amount: 10</body></html>") == "text/plain"


def test_unknown_bytes_are_not_guessed_at() -> None:
    """Refusing to guess is the correct outcome, not a failure."""
    assert sniff(bytes(range(256))) == "application/octet-stream"


# --------------------------------------------------------------------------
# PDF: the branch that had no code path
# --------------------------------------------------------------------------


def test_scanned_pdf_reports_no_text_layer(work: Path) -> None:
    """The measured fact the whole routing table depends on."""
    pdf = _scan_pdf(work / "scan.pdf")
    ext = extract_pdf(pdf.read_bytes(), tmpdir=work)
    assert ext.has_text_layer is False
    assert ext.text.strip() == ""


def test_scanned_pdf_is_rasterised_so_ocr_can_run(work: Path) -> None:
    """This is the code path that did not exist.

    Before this module, "PDF scanned -> OCR" was documented and unreachable: the
    OCR backend takes an image path and nothing produced one from a PDF.
    """
    pdf = _scan_pdf(work / "scan.pdf")
    ext = extract_pdf(pdf.read_bytes(), tmpdir=work)
    assert ext.image_path is not None, "a scan must yield pixels for OCR"
    assert Path(ext.image_path).exists()
    with Image.open(ext.image_path) as rendered:
        assert rendered.width > 0 and rendered.height > 0


def test_text_layer_pdf_is_not_rasterised(work: Path) -> None:
    """A text-layer PDF must not be rendered.

    Rendering it would cost time and, more importantly, replace exact characters
    with recognised ones. PROCESS.md 18.11 measures what that substitution costs:
    a correct `NL94...` IBAN came back as `NI94...`.
    """
    pdf = _text_pdf(work / "letter.pdf")
    ext = extract_pdf(pdf.read_bytes(), tmpdir=work)
    assert ext.has_text_layer is True
    assert ext.image_path is None, "rendering an exact text layer is pure loss"
    assert "DE80716202183220023878" in ext.text


def test_text_layer_survives_with_the_iban_intact(work: Path) -> None:
    pdf = _text_pdf(work / "letter.pdf")
    ext = extract_pdf(pdf.read_bytes(), tmpdir=work)
    assert "NL9439110653356593" in ext.text
    assert "DEUTDEFF698" in ext.text


def test_pdf_without_a_renderer_says_so_rather_than_returning_empty(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent text is not the same as absent capability, and the notes say which."""
    pdf = _scan_pdf(work / "scan.pdf")
    monkeypatch.setattr("iso20022_lab.documents._rasterise_pdf", lambda *a, **k: None)
    ext = extract_pdf(pdf.read_bytes(), tmpdir=work)
    assert ext.image_path is None
    assert any("renderer" in n for n in ext.notes), ext.notes


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------


def _mail_with(
    body: str, *, attach: bool = True, name: str = "scan.pdf", payload: bytes | None = None
) -> bytes:
    msg = EmailMessage()
    msg["From"] = "ap@corp.example"
    msg["To"] = "payments@bank.example"
    msg["Subject"] = "Payment instruction"
    msg.set_content(body)
    if attach:
        data = (
            payload
            if payload is not None
            else _scan_pdf(Path("/tmp/_attach_tmp.pdf")).read_bytes()
        )
        msg.add_attachment(data, maintype="application", subtype="pdf", filename=name)
    return msg.as_bytes()


def test_email_body_is_extracted_as_text() -> None:
    ext = extract_email(_mail_with("Amount: EUR 1.250,00", attach=False))
    assert ext.media_type == "message/rfc822"
    assert "EUR 1.250,00" in ext.text


def test_email_attachment_is_not_concatenated_into_the_body() -> None:
    """The silent wrong-value path this prevents.

    A generous reading of "capture the email" is to concatenate body and
    attachment. Then a reference number in the covering note sits in the same
    string as the payment's fields, and extraction cannot tell them apart -- a
    wrong value rather than a failed parse.
    """
    ext = extract_email(_mail_with("Reference for our records: 999999"))
    assert len(ext.parts) == 1
    # The attachment's own text must not appear in the body field.
    assert "DE80716202183220023878" not in ext.text


def test_email_attachment_is_extracted_in_its_own_right() -> None:
    ext = extract_email(_mail_with("see attached"))
    (part,) = ext.parts
    assert part.media_type == "application/pdf"
    assert part.has_text_layer is False, "the payload is an image-only PDF"


def test_attachment_named_pdf_but_really_png_is_typed_from_bytes() -> None:
    png = _render_doc(Path("/tmp/_attach_png.png")).read_bytes()
    ext = extract_email(_mail_with("see attached", name="statement.pdf", payload=png))
    (part,) = ext.parts
    assert part.media_type == "image/png", (
        "the filename says pdf; the bytes say png, and the bytes are the evidence"
    )


def test_email_without_attachment_has_no_parts() -> None:
    ext = extract_email(_mail_with("nothing attached", attach=False))
    assert ext.parts == []


# --------------------------------------------------------------------------
# Raster
# --------------------------------------------------------------------------


def test_image_is_written_to_a_path_the_backend_can_open(work: Path) -> None:
    src = _render_doc(work / "in.png")
    ext = extract_image(src.read_bytes(), tmpdir=work, stem="scan")
    assert ext.image_path is not None
    assert Path(ext.image_path).exists()
    assert ext.has_text_layer is False


def test_image_suffix_follows_bytes_not_the_given_name(work: Path) -> None:
    """A PNG mislabelled `.tif` must still land on disk as a PNG."""
    src = _render_doc(work / "in.png")
    ext = extract_image(src.read_bytes(), tmpdir=work, stem="mislabelled")
    assert ext.image_path is not None
    assert Path(ext.image_path).suffix == ".png"


# --------------------------------------------------------------------------
# Dispatch and routing
# --------------------------------------------------------------------------


def test_structured_edi_is_never_recognised(work: Path) -> None:
    edi = b"{1:F01BANKBEBBAXXX0000000000}{2:I101...}"
    ext = extract_bytes(edi, filename="pain.edi", tmpdir=work)
    assert ext.media_type == "application/edi"
    assert ext.structured is not None
    assert ext.image_path is None


def test_channel_is_derived_from_media_type(work: Path) -> None:
    pdf = _text_pdf(work / "l.pdf")
    ext = extract_bytes(pdf.read_bytes(), filename="l.pdf", tmpdir=work)
    assert channel_for(ext) == channel_for(ext, from_email=False)


def test_attachment_channel_differs_from_root_channel() -> None:
    """An attachment routes as an attachment even when its type matches the root."""
    ext = Extracted(media_type="application/pdf")
    assert channel_for(ext, from_email=False) != channel_for(ext, from_email=True)


# --------------------------------------------------------------------------
# ingest(): one result per artifact, never one per message
# --------------------------------------------------------------------------


def test_ingest_returns_one_result_per_artifact() -> None:
    """A body plus one attachment is two results, not one."""
    results = ingest(_mail_with("cover note"))
    assert len(results) == 2, "collapsing them would merge two documents' fields"


def test_ingest_of_a_bare_text_file_is_a_single_free_result(work: Path) -> None:
    results = ingest(
        b"Beneficiary: Acme GmbH\nAmount: EUR 10,00", filename="n.txt", tmpdir=work
    )
    assert len(results) == 1
    assert results[0].used_ocr is False


def test_ingest_does_not_fabricate_text_when_ocr_is_absent(work: Path) -> None:
    """No backend means no text, and the result must say that explicitly.

    The alternative -- returning the empty string as if it were extracted -- is
    indistinguishable from a blank document at every later stage.
    """
    pdf = _scan_pdf(work / "scan.pdf")
    results = ingest(pdf.read_bytes(), filename="scan.pdf", tmpdir=work)
    assert len(results) == 1
    assert results[0].used_ocr is False
    assert results[0].text == ""
    joined = " ".join(results[0].notes).lower()
    assert "ocr" in joined or "text layer" in joined, results[0].notes


def test_describe_reports_the_route_for_each_artifact() -> None:
    results = ingest(b"Beneficiary: Acme GmbH\nAmount: EUR 10,00", filename="n.txt")
    out = describe(results)
    assert "1." in out
