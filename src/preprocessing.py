"""Preprocessing for both dataset arrays and real-world user images.

Two pipelines live here and they must agree, otherwise the model sees
something at inference time that it never saw during training:

1. :func:`prepare_arrays` - the dataset pipeline (scale to [0, 1], add the
   channel axis).  EMNIST is already 28x28, centred and size-normalised by
   NIST, so no resizing or re-centring is applied to it.
2. :func:`preprocess_user_image` - the real-world pipeline.  A photo or
   scan is *not* EMNIST-like, so it is converted to grayscale, its polarity
   is detected, it is binarised, cropped to the ink, size-normalised into a
   20x20 box and centred by centre of mass inside 28x28 - which is exactly
   how NIST built MNIST/EMNIST in the first place.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: Model input geometry (height, width, channels).
IMAGE_SIZE = 28
N_CHANNELS = 1
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS)

#: Side of the box the character is scaled into before centring, matching
#: the NIST procedure that produced MNIST/EMNIST.
INNER_BOX = 20


# --------------------------------------------------------------------------
# Dataset pipeline
# --------------------------------------------------------------------------
def prepare_arrays(x: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """Scale ``uint8`` images to [0, 1] and append the channel axis.

    ``(N, 28, 28)`` uint8 -> ``(N, 28, 28, 1)`` float32.
    """
    x = np.asarray(x)
    if x.ndim == 2:  # a single image
        x = x[None, ...]
    if x.ndim == 3:
        x = x[..., None]
    return (x.astype(dtype) / 255.0).astype(dtype)


def to_uint8_images(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`prepare_arrays` for plotting: ``(N, 28, 28)`` uint8."""
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[..., 0]
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


# --------------------------------------------------------------------------
# Real-world image pipeline
# --------------------------------------------------------------------------
def _read_grayscale(source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    """Read any supported input into a 2-D ``uint8`` grayscale array."""
    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[..., 0]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    if isinstance(source, Image.Image):
        pil = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        # PIL handles PNG/JPG/BMP/TIFF and transparency consistently.
        pil = Image.open(path)

    if pil.mode in {"RGBA", "LA", "P"}:
        # Composite transparency onto white so a transparent PNG does not
        # decode as black ink on a black background.
        pil = pil.convert("RGBA")
        background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(background, pil)
    return np.array(pil.convert("L"), dtype=np.uint8)


def _binarise(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return an ink mask (255 = ink) and whether the source was inverted.

    Otsu's method picks the threshold from the image's own histogram, so it
    copes with photographs, scans and screenshots alike.  Polarity is then
    decided by area: handwriting covers a minority of the frame, so the
    smaller of the two regions is the ink.
    """
    # A light blur suppresses camera/JPEG noise before thresholding.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_ratio = float((mask > 0).mean())
    inverted = False
    if foreground_ratio > 0.5:
        # More "bright" pixels than dark ones => dark ink on light paper.
        mask = cv2.bitwise_not(mask)
        inverted = True
    return mask, inverted


def _largest_component(mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Keep only the largest connected blob, dropping specks and paper grain."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    if areas.max() < min_area:
        return mask
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def _center_by_mass(image20: np.ndarray) -> np.ndarray:
    """Paste a size-normalised character into 28x28 at its centre of mass."""
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    h, w = image20.shape
    top = (IMAGE_SIZE - h) // 2
    left = (IMAGE_SIZE - w) // 2
    canvas[top : top + h, left : left + w] = image20

    total = canvas.sum()
    if total <= 0:
        return canvas
    ys, xs = np.nonzero(canvas)
    weights = canvas[ys, xs]
    cy = float((ys * weights).sum() / total)
    cx = float((xs * weights).sum() / total)
    shift_y = int(round(IMAGE_SIZE / 2.0 - 0.5 - cy))
    shift_x = int(round(IMAGE_SIZE / 2.0 - 0.5 - cx))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        canvas, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR, borderValue=0.0
    )


def preprocess_user_image(
    source: str | Path | np.ndarray | Image.Image,
    keep_largest_component: bool = True,
    return_steps: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Turn an arbitrary handwritten-character image into a model input.

    Steps: grayscale -> denoise -> Otsu binarise -> polarity detection ->
    keep largest blob -> crop to ink bounding box -> scale longest side to
    20 px (aspect preserved) -> centre by centre of mass in 28x28 ->
    scale to [0, 1] -> shape ``(1, 28, 28, 1)``.

    Returns the tensor, or ``(tensor, steps)`` when ``return_steps`` is set,
    where ``steps`` holds the intermediate images for visualisation.
    """
    gray = _read_grayscale(source)
    mask, inverted = _binarise(gray)
    cleaned = _largest_component(mask) if keep_largest_component else mask

    ys, xs = np.nonzero(cleaned)
    if len(ys) == 0:
        raise ValueError("No foreground/ink detected in the image after thresholding.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Crop the grayscale intensities (not the binary mask) so stroke
    # darkness is preserved, but blank out everything outside the ink blob.
    ink = np.where(cleaned > 0, 255 - gray if inverted else gray, 0).astype(np.uint8)
    cropped = ink[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = INNER_BOX / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    # INTER_AREA is the correct choice when downsampling.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)

    centered = _center_by_mass(resized.astype(np.float32))
    if centered.max() > 0:
        centered = centered / centered.max() * 255.0

    tensor = prepare_arrays(centered.astype(np.uint8))
    if not return_steps:
        return tensor

    steps = {
        "grayscale": gray,
        "mask": cleaned,
        "inverted": inverted,
        "cropped": cropped,
        "resized": resized,
        "final_28x28": centered.astype(np.uint8),
        "bbox": (x0, y0, x1, y1),
    }
    return tensor, steps


def preprocessing_config() -> dict:
    """Serialisable description of the pipeline, saved next to the model."""
    return {
        "image_size": IMAGE_SIZE,
        "channels": N_CHANNELS,
        "input_shape": list(INPUT_SHAPE),
        "inner_box": INNER_BOX,
        "scaling": "divide by 255 -> [0, 1]",
        "dataset_orientation_fix": "transpose H and W axes of raw EMNIST IDX images",
        "user_image_pipeline": [
            "read as grayscale (transparency composited onto white)",
            "gaussian blur 3x3",
            "otsu threshold",
            "polarity detection (ink = minority region)",
            "keep largest connected component",
            "crop to ink bounding box",
            f"resize longest side to {INNER_BOX} px, aspect preserved",
            f"centre by centre of mass in {IMAGE_SIZE}x{IMAGE_SIZE}",
            "scale to [0, 1]",
        ],
        "foreground_convention": "white ink (high values) on black background",
    }
