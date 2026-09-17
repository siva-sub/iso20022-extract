"""Turning raw inbound bytes into `capture()` inputs.

`ingest.py` routes a document once somebody has already told it the channel, the
text, and whether a text layer exists. This module produces those facts from the
bytes, which is the layer that did not exist: the routing table in PROCESS.md
18 promised that a scanned PDF reaches OCR, and nothing turned a scanned PDF
into pixels.

WHAT THIS MEASURES THAT THE ROUTER CANNOT

The router trusts its caller. A caller that says "channel=PDF, has_text_layer=
True" gets the cheap path. That is fine as an interface and dangerous as a
default, because the two facts that decide the expensive path are exactly the two
facts a caller is most likely to guess. So this module derives both:

- **Type comes from magic bytes, never the filename.** A file named
  `payment.pdf` that is really a PNG is routed as a PNG. This is the input-layer
  form of the rule the rest of the project already follows at the output layer:
  shape is not evidence. See PROCESS.md 18.3.
- **`has_text_layer` comes from trying to read it**, not from a flag. A PDF with
  a text layer that extracts to nothing is a scan, whatever its MIME type says.

THE ONE THAT MATTERS

A PDF's text layer is often *present but wrong* -- and the decision to trust it
is where the money is. This module does not decide that. It extracts the layer
and hands it to `ingest.text_layer_trust`, which reuses the output validators
(IBAN mod-97, BIC shape) as an input-quality probe. Extraction and judgement stay
separate so the judgement can be tested on its own.

Measured on real PP-OCRv6 output (PROCESS.md 18.11), that split is load-bearing:

    truth  NL9439110653356593  -> OCR  NI9439110653356593   mod-97 catches it
    truth  DEUTDEFF698         -> OCR  DEUTDEEF698          nothing catches it

Recognition damaged `NL` to `NI` and `FF` to `F` on a *clean* render. The second
passes IBAN-shaped, BIC-shaped *and* country checks while being wrong, so a
checksum is not sufficient authority for a recognised field.
"""

from __future__ import annotations

import email
import io
import re
import tempfile
from dataclasses import dataclass, field
from email.message import Message
from email.policy import default as default_policy
from pathlib import Path

from iso20022_lab.ingest import (
    Channel,
    CaptureResult,
    OcrBackend,
    capture,
)

# --------------------------------------------------------------------------
# Type sniffing
# --------------------------------------------------------------------------

# Enough of each container to recognise it. Deliberately short: a sniffer that
# reads 8 KB to identify a file is doing work the router will redo.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_TEXT_SNIFF = re.compile(rb"^[\s\x20-\x7e\r\n\t]{32,}$")

# Real EDI markers only. This used to include a bare `<`, meant for XML forms like
# pain.001 -- but `<` also opens every HTML email, so an HTML body was sniffed as
# a structured SWIFT/EDI message and handed to a structured parser.
#
# The XML alternative is written as three explicit cases rather than one clever
# pattern, because the first attempt at it silently failed. It required
# `<?xml ...?>` to be followed directly by the element name, but the declaration
# is followed by `<Document` -- the opening angle of the document element. A
# pain.001 therefore fell through to `text/plain`, which is the misroute this
# regex exists to prevent. Spelling out both shapes is longer and correct.
_EDI_SNIFF = re.compile(
    rb"^(?:\{|\*|ISA|UNB|MSGID)"
    rb"|<\?xml[^>]*\?>\s*<(?:Document|Envelope|pain|pacs|camt|ISO20022)"
    rb"|<(?:Document|Envelope|pain|pacs|camt|ISO20022)[\s>]",
    re.MULTILINE,
)

# RFC 822 detection. Ordered before the ASCII-text fallback because a raw message
# is ASCII: without this check `sniff()` called an email `text/plain`, the MIME
# tree was never walked, and the *entire* message -- base64 attachment payload
# included -- was handed on as the document's text. That is a silent wrong-value
# path, not a parse error, and the test that caught it asserted a count.
_HEADER_LINE = re.compile(rb"^([A-Za-z][A-Za-z0-9-]*):[ \t]?", re.MULTILINE)
# Headers that appear in mail and essentially nowhere else. `Date` and `Subject`
# are too generic to count on their own; these are not.
_MAIL_HEADERS = (
    b"from",
    b"to",
    b"cc",
    b"bcc",
    b"subject",
    b"date",
    b"mime-version",
    b"content-type",
    b"content-transfer-encoding",
    b"received",
    b"message-id",
    b"reply-to",
    b"sender",
    b"return-path",
    b"delivered-to",
)


def _looks_like_email(data: bytes) -> bool:
    """True when this is an RFC 822 message rather than text that resembles one.

    Two conditions, because either alone misfires. There must be a header block
    terminated by a blank line -- otherwise body prose with a colon is a header --
    and at least two headers must be names that mail uses and ordinary documents
    do not. A payment instruction field like `Amount:` is a legitimate document
    line, so a single matching header is not evidence.
    """
    head = data[:4096]
    # mbox "From " separator, then the first blank line ending the headers.
    if head.startswith(b"From "):
        body_start = head.find(b"\n\n")
    else:
        body_start = head.find(b"\n\n")
    if body_start == -1 or body_start < 8:
        return False
    block = head[:body_start]
    names = {m.group(1).lower() for m in _HEADER_LINE.finditer(block)}
    matches = sum(1 for h in _MAIL_HEADERS if h in names)
    return matches >= 2


def sniff(data: bytes, filename: str = "", mime_type: str = "") -> str:
    """Best-effort media type, preferring bytes over any declared name.

    `filename` and `mime_type` are only consulted when the bytes are not
    self-identifying. When they are, the declaration is ignored -- including when
    it contradicts. Trusting the name is how a renamed executable becomes a
    "PDF", and in this pipeline it is how a scan gets handed to a text-layer
    parser that returns an empty string, which then looks like a blank document
    rather than a routing mistake.
    """
    head = data[:16]
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind

    # WebP needs a length check; the RIFF form is shared with wav/avi.
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    declared = (mime_type or "").split(";", 1)[0].strip().lower()
    if declared in ("message/rfc822", "application/vnd.ms-outlook"):
        return declared
    # A declared type is honoured only for types with no reliable magic bytes.
    if declared.startswith(("image/", "application/pdf", "text/", "message/")):
        return declared

    low = filename.lower()
    if low.endswith((".eml", ".msg")):
        return "message/rfc822"
    if low.endswith(".edi") or low.endswith(".mt101"):
        return "application/edi"

    # Before the ASCII fallback: a raw message is ASCII, and treating it as text
    # hands the MIME envelope on as document content.
    if _looks_like_email(data):
        return "message/rfc822"

    if _EDI_SNIFF.match(data[:512]):
        return "application/edi"
    if _TEXT_SNIFF.match(data[:512]):
        return "text/plain"
    return "application/octet-stream"


# --------------------------------------------------------------------------
# What extraction produces
# --------------------------------------------------------------------------


@dataclass
class Extracted:
    """One inbound artifact reduced to what `capture()` needs.

    `image_path` is a path rather than an array because that is the OCR backend's
    interface (`OcrBackend.text_from_image`). Rasterising eagerly is deliberate:
    it happens only when the type is raster or the text layer already failed, so
    the cost tracks the routing decision instead of preceding it.
    """

    text: str = ""
    has_text_layer: bool = False
    image_path: str | None = None
    media_type: str = ""
    structured: str | None = None
    notes: list[str] = field(default_factory=list)
    # Nested items: an email's attachments, each already `Extracted`.
    parts: list["Extracted"] = field(default_factory=list)
    filename: str = ""


def _rasterise_pdf(data: bytes, scale: int, tmpdir: Path, stem: str) -> str | None:
    """Render page 1 of a PDF to PNG, or None if no renderer is installed.

    Page 1 only. A payment instruction is a one-page artifact; rendering every
    page of a 60-page statement would pay OCR cost proportional to a document
    nobody asked about. Multi-page handling belongs in the caller, which knows
    whether it wants the rest.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(io.BytesIO(data))
        if len(doc) == 0:
            return None
        bitmap = doc[0].render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        out = tmpdir / f"{stem}.png"
        image.save(out)
        doc.close()
        return str(out)
    except Exception:
        return None


def _pdf_text(data: bytes) -> str:
    """Extract the embedded text layer, or "" when there is not one."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""
    try:
        doc = pdfium.PdfDocument(io.BytesIO(data))
        if len(doc) == 0:
            return ""
        page = doc[0]
        text = page.get_textpage().get_text_range()
        doc.close()
        return text
    except Exception:
        return ""


def extract_pdf(
    data: bytes,
    *,
    tmpdir: Path | None = None,
    scale: int = 2,
    stem: str = "page",
) -> Extracted:
    """A PDF reduced to text, pixels, or both.

    Rasterises **only** when the text layer is empty. A text-layer PDF is not
    rendered: rendering it would cost time and introduce recognition errors into
    a document whose characters were already exact. The measured cost of not
    making that distinction is in PROCESS.md 18.11 -- OCR changed a correct
    `NL...` IBAN to `NI...` and a correct `DEUTDEFF` BIC to `DEUTDEEF`.
    """
    notes: list[str] = []
    text = _pdf_text(data)
    if text.strip():
        notes.append("text layer extracted; nothing rendered")
        return Extracted(text=text, has_text_layer=True, media_type="application/pdf")

    notes.append("no text layer: this is a scan and needs recognition")
    workdir = tmpdir or Path(tempfile.mkdtemp(prefix="ingest-"))
    workdir.mkdir(parents=True, exist_ok=True)
    image = _rasterise_pdf(data, scale, workdir, stem)
    if image is None:
        notes.append("no PDF renderer available; text is unavailable, not empty")
    return Extracted(
        text="",
        has_text_layer=False,
        image_path=image,
        media_type="application/pdf",
        notes=notes,
    )


def extract_email(data: bytes, *, tmpdir: Path | None = None) -> Extracted:
    """An RFC 822 / MIME message split into body text and attachments.

    Uses the standard library, not `unstructured`. For this input -- a payment
    instruction arriving as mail -- `email.policy.default` decodes the transfer
    encoding, the charset and the MIME tree correctly, and it is the reference
    implementation of the format. A third-party document library buys breadth
    across formats this pipeline does not accept; it does not buy correctness on
    this one.

    Attachments are returned as `parts` and are *not* merged into the body. The
    body is a message somebody typed; an attachment is a file. Concatenating them
    lets a number in the covering note be mistaken for a field of the payment,
    which is a silent wrong-value path rather than a parse error.
    """
    msg: Message = email.message_from_bytes(data, policy=default_policy)
    notes: list[str] = []

    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            if ctype == "text/plain":
                try:
                    body_parts.append(part.get_content())
                except (LookupError, UnicodeDecodeError):
                    notes.append("a text part failed to decode and was skipped")
            elif ctype == "text/html":
                try:
                    body_parts.append(_html_to_text(part.get_content()))
                except (LookupError, UnicodeDecodeError):
                    notes.append("an html part failed to decode and was skipped")
    else:
        try:
            content = msg.get_content()
        except (LookupError, UnicodeDecodeError):
            content = ""
            notes.append("email body failed to decode")
        body_parts.append(
            _html_to_text(content) if msg.get_content_type() == "text/html" else content
        )

    parts: list[Extracted] = []
    for part in msg.iter_attachments():
        name = part.get_filename() or "attachment"
        payload = part.get_payload(decode=True)
        # `get_payload(decode=True)` returns bytes for any part with a transfer
        # encoding, and a str for one without. The stub types it loosely, and the
        # narrow is not only for the checker: a part that arrives as text would
        # otherwise reach `sniff()` as a str and be compared against byte magic
        # values, matching nothing and silently falling through to the
        # unrecognised-type branch.
        if not isinstance(payload, bytes):
            notes.append(f"attachment {name!r} is not byte payload; skipped")
            continue
        real_type = sniff(payload, name, part.get_content_type())
        parts.append(
            extract_bytes(payload, filename=name, mime_type=real_type, tmpdir=tmpdir)
        )
    if parts:
        notes.append(f"{len(parts)} attachment(s) captured separately from the body")

    text = "\n".join(p for p in body_parts if p.strip())
    return Extracted(
        text=text,
        has_text_layer=True,
        media_type="message/rfc822",
        notes=notes,
        parts=parts,
    )


_TAG = re.compile(r"<[^>]+>")
_BLOCKISH = re.compile(r"</(p|div|tr|li|h[1-6]|table|br)\s*>|<br\s*/?>", re.IGNORECASE)


def _html_to_text(html: str) -> str:
    """Flatten HTML to text without a parser dependency.

    `bs4` and `lxml` are installed, and are deliberately not used here. What is
    needed is the *text* of a payment instruction, and the structure that matters
    to this pipeline -- which label sits next to which value -- is carried by
    block boundaries and by whitespace. A DOM traversal returns the same string
    with more machinery. The regex is honest about what it is: a flattener, not a
    parser. It is not asked to handle malformed input, because a mail client that
    produced the message already fixed the markup.
    """
    text = _BLOCKISH.sub("\n", html)
    text = re.sub(r"<(script|style)\b.*?</\1>", "", text, flags=re.S | re.I)
    text = _TAG.sub("", text)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ):
        text = text.replace(entity, char)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines())


def extract_image(
    data: bytes, *, tmpdir: Path | None = None, stem: str = "scan"
) -> Extracted:
    """Write raster bytes to a path the OCR backend can open."""
    workdir = tmpdir or Path(tempfile.mkdtemp(prefix="ingest-"))
    workdir.mkdir(parents=True, exist_ok=True)
    # The suffix follows the sniffed type, not the supplied name: pypdfium2 and
    # PIL both behave differently on a mislabelled extension.
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tif",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
        "image/gif": ".png",
    }.get(sniff(data), ".png")
    out = workdir / f"{stem}{suffix}"
    out.write_bytes(data)
    # The sniffed type, not a placeholder. Returning "image" lost the distinction
    # between a scan and a photograph, which is the distinction the router needs
    # to choose a model and the audit needs to explain the choice.
    kind = sniff(data, filename=f"{stem}{suffix}")
    return Extracted(
        text="",
        has_text_layer=False,
        image_path=str(out),
        media_type=kind,
        filename=f"{stem}{suffix}",
        notes=["raster input: recognition required, no text layer exists"],
    )


def extract_bytes(
    data: bytes,
    *,
    filename: str = "",
    mime_type: str = "",
    tmpdir: Path | None = None,
    pdf_scale: int = 2,
) -> Extracted:
    """Dispatch on sniffed type. The single entry point for raw bytes."""
    kind = sniff(data, filename, mime_type)
    stem = Path(filename).stem if filename else "doc"

    if kind == "application/pdf":
        return extract_pdf(data, tmpdir=tmpdir, scale=pdf_scale, stem=stem)
    if kind.startswith("image/"):
        return extract_image(data, tmpdir=tmpdir, stem=stem)
    if kind in ("message/rfc822", "application/vnd.ms-outlook"):
        return extract_email(data, tmpdir=tmpdir)
    if kind == "application/edi":
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return Extracted(
            text=text,
            has_text_layer=True,
            structured=text,
            media_type=kind,
            notes=["structured message: parsed, never recognised"],
        )
    if kind.startswith("text/"):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return Extracted(text=text, has_text_layer=True, media_type=kind)

    return Extracted(
        media_type=kind,
        notes=[f"unrecognised type {kind!r}; refusing to guess a parser"],
    )


# --------------------------------------------------------------------------
# Bytes -> capture()
# --------------------------------------------------------------------------


def channel_for(ext: Extracted, *, from_email: bool = False) -> Channel:
    """The channel a sniffed artifact belongs to.

    Derived, not declared. `EMAIL_BODY` and `EMAIL_ATTACHMENT` are different
    channels because they route differently: a body is already text, while an
    attachment is a file that must be typed before anything else can be decided.
    """
    if ext.media_type == "message/rfc822":
        return Channel.EMAIL_BODY
    if from_email:
        return Channel.EMAIL_ATTACHMENT
    if ext.media_type == "application/pdf":
        return Channel.PDF
    if ext.media_type == "application/edi":
        return Channel.EDI
    if ext.media_type.startswith("image/"):
        return Channel.FAX
    return Channel.EMAIL_BODY


def ingest(
    data: bytes,
    *,
    filename: str = "",
    mime_type: str = "",
    ocr: OcrBackend | None = None,
    tmpdir: Path | None = None,
) -> list[CaptureResult]:
    """Bytes in, one `CaptureResult` per artifact out.

    A list rather than a single result because one email can carry three
    documents, and collapsing them into one result would concatenate values from
    different payments into a single extraction. Order is body first, then
    attachments in the order the message declared them.
    """
    root = extract_bytes(data, filename=filename, mime_type=mime_type, tmpdir=tmpdir)
    results = [_capture_one(root, ocr=ocr, from_email=False)]

    for part in root.parts:
        results.append(_capture_one(part, ocr=ocr, from_email=True))
    return results


def _capture_one(
    ext: Extracted, *, ocr: OcrBackend | None, from_email: bool
) -> CaptureResult:
    """Run one `Extracted` through the router and carry its notes through.

    The notes are concatenated rather than replaced. `capture()` reports why it
    chose a path, and this layer reports why the document looked the way it did;
    losing either makes an audit of "why did this cost OCR time" unanswerable.
    """
    channel = channel_for(ext, from_email=from_email)
    result = capture(
        channel,
        text=ext.text,
        image_path=ext.image_path,
        structured=ext.structured,
        has_text_layer=ext.has_text_layer,
        ocr=ocr,
    )
    if ext.notes:
        result.notes = list(ext.notes) + list(result.notes)
    return result


def describe(results: list[CaptureResult]) -> str:
    """A one-line-per-artifact audit of the routing decision."""
    lines = []
    for i, r in enumerate(results, 1):
        cost = "OCR" if r.used_ocr else "free"
        lines.append(f"  {i}. {r.route.channel.value:<18} {r.route.path.value:<11} {cost}")
        for note in r.notes:
            lines.append(f"       - {note}")
    return "\n".join(lines)


def main() -> int:
    """CLI: ingest one file and show the routing, for the recipe book."""
    import argparse

    parser = argparse.ArgumentParser(description="capture one inbound document")
    parser.add_argument("path", type=Path)
    parser.add_argument("--ocr", choices=["none", "tiny", "medium"], default="none")
    args = parser.parse_args()

    backend: OcrBackend | None = None
    if args.ocr != "none":
        from iso20022_lab.ocr import build_medium, build_tiny

        backend = build_tiny() if args.ocr == "tiny" else build_medium()

    data = args.path.read_bytes()
    results = ingest(data, filename=args.path.name, ocr=backend)
    print(f"{args.path.name}: {sniff(data, args.path.name)}")
    print(describe(results))
    for i, r in enumerate(results, 1):
        if r.text:
            preview = r.text[:160].replace("\n", " | ")
            print(f"  [{i}] text: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
