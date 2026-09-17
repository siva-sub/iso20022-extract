"""PP-OCRv6 via ONNX Runtime. Real recognition, no PaddlePaddle dependency.

PROCESS.md section 18 described the OCR path as an integration point that had
never executed. This module removes that caveat: it runs the actual PP-OCRv6
ONNX weights through `onnxruntime`, so the capture layer can be tested against
real pixels rather than a stub.

WHY ONNX AND NOT PADDLEOCR

The published PaddleOCR 3.x package pulls in PaddlePaddle, which is a large
dependency for a pipeline that only ever calls "image in, text out". The ONNX
export runs on `onnxruntime`, keeps the deployment footprint small, and is the
artefact already published for these models. For a capture service that runs
on-premises next to the payment system, that matters.

THE DICTIONARY IS LOAD-BEARING, AND A MISMATCH IS SILENT

Recognition output has one class per dictionary entry plus a CTC blank plus an
optional space. The counts must line up exactly:

    PP-OCRv6_tiny_rec   6906 classes = 6904 chars + blank + space
    PP-OCRv6_medium_rec 18710 classes = 18708 chars + blank + space

Feed a model the wrong dictionary and it does not error -- it produces confident
garbage, because every index still maps to *a* character. The v5 dictionary is
18383 entries and the v1 dictionary is 6623; either would decode nonsense here
without a single warning. `load_dictionary` checks the count against the model's
output dimension and refuses to run on a mismatch, because a silent failure at
this layer becomes a wrong account number downstream.

IMPLEMENTATION NOTE

Detection postprocessing (threshold, connected components, box expansion) is
written in numpy rather than OpenCV, because OpenCV is a large dependency for
four operations and the corpus pages are clean renders rather than photographs.
That is a real limitation: this is not a general document-unwarping pipeline, and
section 18's limitations carry that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from iso20022_lab.ingest import OcrBackend

# Normalisation used by PP-OCR for both detection and recognition.
_MEAN = 0.5
_STD = 0.5

# Detection postprocessing. These are the PP-OCR defaults and are exposed as
# parameters rather than constants because the right values depend on scan
# quality -- a fax needs a lower box threshold than a clean render.
DET_THRESH = 0.3
BOX_THRESH = 0.5

# UNCLIP_RATIO IS THE SINGLE MOST IMPORTANT KNOB HERE, and the default matters.
#
# The detector emits a probability map covering the CORE of each text region, so
# boxes must be grown before cropping or recognition is handed a thin horizontal
# slice of every line. Measured on a 32px render, sweeping the ratio:
#
#     ratio   crop height   exact   normalised
#     1.6     18-19px       0/8     3/8        <- published PP-OCR default
#     2.0     24-25px       0/8     3/8
#     2.5     30-31px       3/8     7/8        <- chosen
#     3.0     36-39px       2/8     8/8
#
# At 1.6 -- which is what PP-OCR ships as the default, and what this file used
# first -- crops came out at 18px for 32px text and recognition produced
# `PAYMENTTNSTRUCTTON` for `PAYMENT INSTRUCTION` and `EIR` for `EUR`. Those look
# like a weak model. They are a coordinate mistake, and no amount of extra
# training would have fixed them.
#
# 2.5 is chosen over 3.0 deliberately. 3.0 scored marginally better on this
# fixture (8/8 normalised) but expands boxes to 36-39px, which is close enough to
# the 54px line pitch here that a denser document would start merging adjacent
# lines -- and a merged line is a much worse failure than a misread glyph, since
# it corrupts two fields at once. 2.5 matches the source text height closely and
# leaves the safety margin.
#
# Normalised accuracy is the metric that matters, not exact: the extractor
# normalises IBAN spacing, amount separators and date order downstream, so
# `Debtor·AcmeGmbH` and `Debtor: Acme GmbH` are the same input to it.
UNCLIP_RATIO = 2.5
MIN_BOX_SIDE = 3

REC_HEIGHT = 48


@dataclass
class TextBox:
    """One detected region and what was read from it."""

    x0: int
    y0: int
    x1: int
    y1: int
    text: str = ""
    confidence: float = 0.0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def load_dictionary(path: Path, num_classes: int) -> list[str]:
    """Load a PP-OCR character dictionary, verifying it matches the model.

    The character list is `num_classes - 1` entries (CTC blank at index 0), with
    an optional space appended when the model was trained with `use_space_char`,
    which PP-OCRv6 is.

    Raises ValueError on a mismatch rather than returning a short list. A wrong
    dictionary does not crash a CTC decode; it produces fluent nonsense, and the
    whole point of this check is that the failure is loud.
    """
    chars = path.read_text(encoding="utf-8").splitlines()
    n = len(chars)
    for extra in (1, 2):
        if n + extra == num_classes:
            if extra == 2:
                chars = [*chars, " "]
            return chars
    raise ValueError(
        f"dictionary {path.name} has {n} entries but the model emits "
        f"{num_classes} classes; expected {n}+1 (blank) or {n}+2 (blank+space). "
        "A mismatched dictionary decodes to confident garbage, so this refuses "
        "to run. Use ppocrv6_tiny_dict.txt for tiny_rec and ppocrv6_dict.txt for "
        "medium_rec."
    )


# --------------------------------------------------------------------------
# Detection postprocessing (numpy)
# --------------------------------------------------------------------------


def _connected_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of connected regions in a boolean mask.

    Iterative flood fill with an explicit stack. Recursion would blow the stack
    on a full-page text region, which is the common case rather than the edge
    case.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for sy in range(h):
        row = mask[sy]
        for sx in range(w):
            if not row[sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            min_x = max_x = sx
            min_y = max_y = sy
            while stack:
                y, x = stack.pop()
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            boxes.append((min_x, min_y, max_x, max_y))
    return boxes


def _unclip(
    box: tuple[int, int, int, int], ratio: float, w: int, h: int
) -> tuple[int, int, int, int]:
    """Grow a box outward, the way PP-OCR's unclip does.

    The detector's probability map is a shrunk version of the text region, so
    boxes must be expanded before cropping or the first and last characters get
    cut off.
    """
    x0, y0, x1, y1 = box
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    pad_x = int(box_w * (ratio - 1) / 2)
    pad_y = int(box_h * (ratio - 1) / 2)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def _sort_reading_order(boxes: list[TextBox]) -> list[TextBox]:
    """Top-to-bottom, then left-to-right, with a line tolerance.

    Without the tolerance, boxes on the same visual line sort by their exact
    y-origin and a word that sits one pixel higher jumps ahead of its neighbours,
    scrambling the line.
    """
    if not boxes:
        return boxes
    heights = [b.height for b in boxes]
    tolerance = max(8, int(np.median(heights) * 0.6))
    return sorted(boxes, key=lambda b: (round(b.y0 / tolerance), b.x0))


# --------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------


class PaddleOcrOnnx(OcrBackend):
    """PP-OCRv6 detection + recognition through onnxruntime.

    Implements the same `OcrBackend` interface as the PaddleOCR 3.x wrapper in
    `ingest.py`, so it is a drop-in replacement for the capture layer.
    """

    name = "ppocrv6-onnx"

    def __init__(
        self,
        det_path: Path,
        rec_path: Path,
        dict_path: Path,
        *,
        det_thresh: float = DET_THRESH,
        box_thresh: float = BOX_THRESH,
        unclip_ratio: float = UNCLIP_RATIO,
    ) -> None:
        import onnxruntime as ort

        # One thread each: this runs alongside the extraction model on the same
        # box, and onnxruntime's default is to grab every core.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        self._det = ort.InferenceSession(
            str(det_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._rec = ort.InferenceSession(
            str(rec_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name

        num_classes = int(self._rec.get_outputs()[0].shape[-1])
        self.chars = load_dictionary(dict_path, num_classes)
        self.num_classes = num_classes

        self.det_thresh = det_thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    # -- detection ---------------------------------------------------------

    def _detect(self, image: np.ndarray) -> list[TextBox]:
        """Run detection, returning boxes in reading order.

        The detector wants dimensions divisible by 32, so the image is padded
        rather than resized: resizing would distort aspect ratio, and the
        recognition stage is aspect-sensitive.
        """
        h, w = image.shape[:2]
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="constant")

        x = padded.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))[None, ...]

        outputs = self._det.run(None, {self._det_input: x})
        # `run` is typed as returning a union that includes SparseTensor, but
        # these exports return dense ndarrays. np.asarray normalises it without
        # an annotation that would have to lie about the union.
        prob = np.asarray(outputs[0], dtype=np.float32)[0, 0]

        # THE DETECTOR DOWNSAMPLES. PP-OCR's detection head emits a probability
        # map at roughly a quarter of the input resolution, so every box below is
        # in MAP coordinates, not image coordinates. Treating them as image
        # coordinates was a real bug with a recognisable signature: boxes came
        # out 7 pixels tall for 28-pixel text, recognition was handed a thin
        # horizontal slice of each line, and it produced `Dabtor` for `Debtor`
        # and `Aeur` for `EUR` -- errors that look like a weak model but are
        # actually a coordinate mistake.
        #
        # The ratio is computed rather than hard-coded to 4, because it is a
        # property of the model export and hard-coding would silently break on a
        # different one.
        map_h, map_w = prob.shape[:2]
        scale_x = w / map_w
        scale_y = h / map_h

        mask = prob > self.det_thresh
        if not mask.any():
            return []

        boxes: list[TextBox] = []
        for raw in _connected_boxes(mask):
            # A region only counts if its peak probability is convincing. The
            # threshold above decides which pixels are candidates; this decides
            # whether the candidate is real, and it is what suppresses speckle
            # noise on a fax.
            mx0, my0, mx1, my1 = raw
            if prob[my0 : my1 + 1, mx0 : mx1 + 1].max() < self.box_thresh:
                continue

            # Scale to image coordinates BEFORE any size filtering or expansion,
            # so that MIN_BOX_SIDE and the unclip ratio are both measured in the
            # units the caller sees.
            x0 = int(mx0 * scale_x)
            y0 = int(my0 * scale_y)
            x1 = int(mx1 * scale_x)
            y1 = int(my1 * scale_y)

            if (x1 - x0) < MIN_BOX_SIDE or (y1 - y0) < MIN_BOX_SIDE:
                continue
            gx0, gy0, gx1, gy1 = _unclip((x0, y0, x1, y1), self.unclip_ratio, w, h)
            boxes.append(TextBox(gx0, gy0, gx1, gy1))

        return _sort_reading_order(boxes)

    # -- recognition -------------------------------------------------------

    def _recognise(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """CTC-decode a batch of crops."""
        if not crops:
            return []

        # Each crop is scaled to height 48 keeping aspect ratio, then padded to
        # the batch's widest member. Padding to a fixed width instead would
        # squash wide lines and lose the character spacing the model relies on.
        prepared: list[np.ndarray] = []
        for crop in crops:
            ch, cw = crop.shape[:2]
            if ch == 0 or cw == 0:
                prepared.append(np.zeros((3, REC_HEIGHT, 8), dtype=np.float32))
                continue
            new_w = max(8, int(round(cw * REC_HEIGHT / ch)))
            resized = np.array(
                Image.fromarray(crop).resize((new_w, REC_HEIGHT), Image.Resampling.BILINEAR)
            ).astype(np.float32)
            arr = (resized / 255.0 - _MEAN) / _STD
            prepared.append(np.transpose(arr, (2, 0, 1)))

        max_w = max(p.shape[2] for p in prepared)
        batch = np.zeros((len(prepared), 3, REC_HEIGHT, max_w), dtype=np.float32)
        for i, p in enumerate(prepared):
            batch[i, :, :, : p.shape[2]] = p

        logits = np.asarray(self._rec.run(None, {self._rec_input: batch})[0])
        return [self._ctc_decode(row) for row in logits]

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy CTC decode: collapse repeats, drop blanks, map to characters.

        Confidence is the mean probability of the kept characters. Averaging over
        every timestep would include the blanks and report high confidence for an
        empty read, which is the one answer that should score low.
        """
        idx = logits.argmax(axis=1)
        probs = logits.max(axis=1)

        chars: list[str] = []
        scores: list[float] = []
        prev = -1
        for t, k in enumerate(idx):
            k = int(k)
            if k != prev and k != 0:
                # Index 0 is the CTC blank, so character i is at dictionary i-1.
                if 1 <= k <= len(self.chars):
                    chars.append(self.chars[k - 1])
                    scores.append(float(probs[t]))
            prev = k

        text = "".join(chars)
        return text, (sum(scores) / len(scores) if scores else 0.0)

    # -- public API --------------------------------------------------------

    def read(self, image_path: str | Path) -> list[TextBox]:
        """Full pipeline: detect regions, recognise each, return in order."""
        image = np.array(Image.open(image_path).convert("RGB"))
        boxes = self._detect(image)
        if not boxes:
            return []

        crops = [image[b.y0 : b.y1, b.x0 : b.x1] for b in boxes]
        results = self._recognise(crops)
        for box, (text, conf) in zip(boxes, results, strict=True):
            box.text = text
            box.confidence = conf
        return boxes

    def text_from_image(self, image_path: str) -> tuple[str, float]:
        """`OcrBackend` interface: (text, mean confidence).

        Reads in detection order and joins on newlines, preserving the reading
        order the extractor depends on. Section 16.3 showed role disambiguation
        is the dominant failure mode, and a scrambled line order destroys the
        positional signal that partly resolves it.
        """
        boxes = self.read(image_path)
        lines = [b.text for b in boxes if b.text.strip()]
        if not lines:
            return "", 0.0
        confs = [b.confidence for b in boxes if b.text.strip()]
        return "\n".join(lines), sum(confs) / len(confs)


def default_model_dir() -> Path:
    """Where the PP-OCRv6 ONNX weights live on this machine.

    Not a hard-coded requirement: callers pass paths explicitly. This exists so
    the recipes can state a working default.
    """
    return Path("/home/siva/models/ppocrv6")


def build_tiny(model_dir: Path | None = None) -> PaddleOcrOnnx:
    """The tiny pair: 1.8 MB detection + 4.5 MB recognition.

    The right default for this pipeline. Payment instructions are clean,
    machine-printed, high-contrast text; the medium models cost 60x the download
    and 20x the compute for accuracy the workflow does not use. Reach for medium
    only when a reconciliation shows the tiny pair is the bottleneck.
    """
    root = model_dir or default_model_dir()
    return PaddleOcrOnnx(
        root / "PP-OCRv6_tiny_det_onnx" / "inference.onnx",
        root / "PP-OCRv6_tiny_rec_onnx" / "inference.onnx",
        root / "dict" / "ppocrv6_tiny_dict.txt",
    )


def build_medium(model_dir: Path | None = None) -> PaddleOcrOnnx:
    """The medium pair, for degraded scans where tiny struggles."""
    root = model_dir or default_model_dir()
    return PaddleOcrOnnx(
        root / "PP-OCRv6_medium_det_onnx" / "inference.onnx",
        root / "PP-OCRv6_medium_rec_onnx" / "inference.onnx",
        root / "dict" / "ppocrv6_dict.txt",
    )
