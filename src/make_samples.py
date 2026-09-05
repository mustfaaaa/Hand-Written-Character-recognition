"""Generate realistic "user image" test cases for the inference pipeline.

These are **simulated** camera/scan images, not photographs of the user's own
handwriting - that distinction is stated wherever the results are reported.
Two families are produced, both with known ground truth:

``photo_*``
    Real EMNIST **test** images (never seen during training) rendered the way
    a phone photo or a scan would look: dark ink on light paper, uneven
    lighting, sensor noise, generous whitespace, an off-centre character, a
    small rotation and JPEG compression.  This exercises every branch of the
    real-world preprocessing pipeline while keeping a trustworthy label.

``font_*``
    Characters rendered from installed fonts.  These are genuinely
    out-of-distribution (printed glyphs, not handwriting) and exist to show
    honestly where the model's domain ends.

A handful of deliberate edge cases (light-on-dark, transparent PNG, tiny and
very large images) round out the Phase 17 coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_loader import load_emnist
from utils import SAMPLES_DIR, SEED, ensure_dirs, header, save_json


def _paper_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Off-white paper with a soft lighting gradient and sensor noise."""
    base = rng.integers(228, 250)
    background = np.full((height, width), base, dtype=np.float32)

    # Smooth diagonal illumination gradient.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = (xx / max(width - 1, 1)) * rng.uniform(-28, 28) + (
        yy / max(height - 1, 1)
    ) * rng.uniform(-28, 28)
    background += gradient
    background += rng.normal(0, 3.0, size=background.shape)
    return np.clip(background, 0, 255)


def render_photo_sample(
    glyph: np.ndarray,
    rng: np.random.Generator,
    out_size: int | None = None,
) -> np.ndarray:
    """Render a 28x28 EMNIST glyph as a realistic dark-on-light photo."""
    out_size = out_size or int(rng.integers(180, 420))

    # Upscale the glyph and give it a soft, pen-like edge.
    scale = int(rng.integers(int(out_size * 0.35), int(out_size * 0.62)))
    big = cv2.resize(glyph, (scale, scale), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (5, 5), 0)

    # Small rotation, well inside the label-preserving range.
    angle = rng.uniform(-9, 9)
    matrix = cv2.getRotationMatrix2D((scale / 2, scale / 2), angle, 1.0)
    big = cv2.warpAffine(big, matrix, (scale, scale), flags=cv2.INTER_LINEAR, borderValue=0)

    canvas = _paper_background(out_size, out_size, rng)

    # Place the character off-centre with generous surrounding whitespace.
    max_top = max(1, out_size - scale)
    top = int(rng.integers(0, max_top))
    left = int(rng.integers(0, max_top))

    ink_strength = rng.uniform(0.72, 0.95)
    alpha = (big.astype(np.float32) / 255.0) * ink_strength
    region = canvas[top : top + scale, left : left + scale]
    ink_colour = rng.uniform(15, 55)
    canvas[top : top + scale, left : left + scale] = region * (1 - alpha) + ink_colour * alpha

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _find_fonts(limit: int = 6) -> list[Path]:
    """Locate installed TrueType fonts, preferring handwriting-like ones."""
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")]
    preferred = ["inkfree", "segoesc", "comic", "bradhitc", "segoeprb", "gabriola", "arial", "calibri"]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in preferred:
            for path in root.glob(f"**/{name}*.ttf"):
                if path not in found:
                    found.append(path)
                if len(found) >= limit:
                    return found
    return found


def render_font_sample(char: str, font_path: Path, size: int = 256) -> np.ndarray | None:
    """Render ``char`` with ``font_path`` as dark text on a white page."""
    try:
        font = ImageFont.truetype(str(font_path), int(size * 0.62))
    except OSError:
        return None
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return None
    draw.text(
        ((size - width) / 2 - bbox[0], (size - height) / 2 - bbox[1]),
        char,
        font=font,
        fill=25,
    )
    return np.array(image, dtype=np.uint8)


def build_edge_cases(glyph: np.ndarray, char: str, rng: np.random.Generator) -> list[dict]:
    """Deliberate Phase 17 edge cases: polarity, transparency, extreme sizes."""
    cases = []

    # Light ink on a dark background (e.g. a photo of a whiteboard marker
    # inverted, or chalk on a blackboard).
    dark = np.clip(np.full((300, 300), 28, dtype=np.float32) + rng.normal(0, 4, (300, 300)), 0, 255)
    big = cv2.resize(glyph, (150, 150), interpolation=cv2.INTER_CUBIC)
    dark[70:220, 60:210] = np.maximum(dark[70:220, 60:210], big.astype(np.float32) * 0.9)
    cases.append({"suffix": "light_on_dark", "image": dark.astype(np.uint8), "mode": "L", "ext": "png"})

    # Transparent PNG (RGBA) - ink on an alpha background.
    rgba = np.zeros((240, 240, 4), dtype=np.uint8)
    glyph_big = cv2.resize(glyph, (160, 160), interpolation=cv2.INTER_CUBIC)
    rgba[40:200, 40:200, 3] = glyph_big  # alpha = ink
    rgba[..., :3] = 20  # near-black ink colour
    cases.append({"suffix": "transparent_png", "image": rgba, "mode": "RGBA", "ext": "png"})

    # Very small and very large renderings.
    cases.append(
        {"suffix": "small_56px", "image": render_photo_sample(glyph, rng, out_size=56), "mode": "L", "ext": "png"}
    )
    cases.append(
        {"suffix": "large_900px", "image": render_photo_sample(glyph, rng, out_size=900), "mode": "L", "ext": "jpg"}
    )

    # Lots of empty margin around a small character.
    wide = _paper_background(500, 500, rng)
    small = cv2.resize(glyph, (70, 70), interpolation=cv2.INTER_CUBIC)
    alpha = small.astype(np.float32) / 255.0
    wide[40:110, 380:450] = wide[40:110, 380:450] * (1 - alpha) + 30 * alpha
    cases.append({"suffix": "offcentre_whitespace", "image": wide.astype(np.uint8), "mode": "L", "ext": "png"})

    for case in cases:
        case["true_label"] = char
    return cases


def generate(n_photo: int = 24, n_font: int = 12, seed: int = SEED) -> dict:
    """Write every sample image to ``samples/`` and return their manifest."""
    ensure_dirs()
    rng = np.random.default_rng(seed)
    data = load_emnist("balanced")
    class_names = data["class_names"]
    x_test, y_test = data["x_test"], data["y_test"]

    manifest: list[dict] = []

    header("GENERATING SIMULATED USER IMAGES")

    # --- photo-style renderings of unseen EMNIST test glyphs -----------
    chosen = rng.choice(len(x_test), size=n_photo, replace=False)
    for n, idx in enumerate(chosen):
        char = class_names[y_test[idx]]
        image = render_photo_sample(x_test[idx], rng)
        safe = f"{ord(char):03d}"
        # JPEG for half of them so compression artefacts are represented.
        ext = "jpg" if n % 2 == 0 else "png"
        path = SAMPLES_DIR / f"photo_{n:02d}_{safe}.{ext}"
        Image.fromarray(image).save(path, quality=72 if ext == "jpg" else None)
        manifest.append(
            {
                "path": str(path),
                "true_label": char,
                "family": "photo",
                "source": f"EMNIST test index {int(idx)} (unseen during training)",
            }
        )

    # --- edge cases ----------------------------------------------------
    idx = int(rng.choice(len(x_test)))
    char = class_names[y_test[idx]]
    for case in build_edge_cases(x_test[idx], char, rng):
        path = SAMPLES_DIR / f"edge_{case['suffix']}_{ord(char):03d}.{case['ext']}"
        Image.fromarray(case["image"], mode=case["mode"]).save(path)
        manifest.append(
            {
                "path": str(path),
                "true_label": char,
                "family": "edge_case",
                "source": f"EMNIST test index {idx}, rendered as {case['suffix']}",
            }
        )

    # --- font renderings (out of distribution) -------------------------
    fonts = _find_fonts()
    if fonts:
        letters = [c for c in class_names]
        picks = rng.choice(letters, size=min(n_font, len(letters)), replace=False)
        for n, char in enumerate(picks):
            font_path = fonts[n % len(fonts)]
            image = render_font_sample(str(char), font_path)
            if image is None:
                continue
            path = SAMPLES_DIR / f"font_{n:02d}_{ord(str(char)):03d}.png"
            Image.fromarray(image).save(path)
            manifest.append(
                {
                    "path": str(path),
                    "true_label": str(char),
                    "family": "font",
                    "source": f"rendered with {font_path.name} (printed glyph, out of distribution)",
                }
            )
    else:
        print("  no TrueType fonts found; skipping the font family")

    save_json(manifest, SAMPLES_DIR / "manifest.json")
    counts: dict[str, int] = {}
    for entry in manifest:
        counts[entry["family"]] = counts.get(entry["family"], 0) + 1
    print(f"  wrote {len(manifest)} images to {SAMPLES_DIR}")
    for family, count in sorted(counts.items()):
        print(f"    {family:12s} {count}")
    return {"manifest": manifest, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated user images.")
    parser.add_argument("--n-photo", type=int, default=24)
    parser.add_argument("--n-font", type=int, default=12)
    args = parser.parse_args()
    generate(n_photo=args.n_photo, n_font=args.n_font)


if __name__ == "__main__":
    main()
