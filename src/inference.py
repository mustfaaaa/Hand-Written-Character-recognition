"""Inference on new, unseen handwritten-character images (Phases 16-19).

Exposes :class:`CharacterRecognizer`, which loads the exported model plus its
label mapping and preprocessing configuration, and returns a prediction with
calibrated-as-trained softmax probabilities and Top-K alternatives.

Command line
------------
    python src/inference.py --image samples/my_letter.png --top-k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocessing import preprocess_user_image, preprocessing_config
from utils import FIGURES_DIR, FINAL_MODELS_DIR, PREDICTIONS_DIR, load_json, save_json

DEFAULT_MODEL_PATH = FINAL_MODELS_DIR / "handwritten_character_cnn.keras"
DEFAULT_LABELS_PATH = FINAL_MODELS_DIR / "labels.json"
DEFAULT_CONFIG_PATH = FINAL_MODELS_DIR / "preprocessing_config.json"


# --------------------------------------------------------------------------
# Export (Phase 19)
# --------------------------------------------------------------------------
def export_model(
    source_model_path: str | Path,
    class_names: list[str],
    dataset: str,
    metrics: dict | None = None,
    destination: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Copy the winning model to its final name and write its sidecar files.

    Writes ``handwritten_character_cnn.keras``, ``labels.json`` and
    ``preprocessing_config.json`` into ``models/final/`` so that inference
    never has to guess the label order or the preprocessing.
    """
    model = tf.keras.models.load_model(source_model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)

    labels = {
        "dataset": dataset,
        "n_classes": len(class_names),
        "class_names": class_names,
        "index_to_char": {str(i): c for i, c in enumerate(class_names)},
        "char_to_index": {c: i for i, c in enumerate(class_names)},
        "source_model": str(source_model_path),
    }
    if metrics:
        labels["test_metrics"] = metrics
    save_json(labels, DEFAULT_LABELS_PATH)
    save_json(preprocessing_config(), DEFAULT_CONFIG_PATH)
    return {
        "model": str(destination),
        "labels": str(DEFAULT_LABELS_PATH),
        "config": str(DEFAULT_CONFIG_PATH),
    }


# --------------------------------------------------------------------------
# Recognizer
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    """A single prediction with its Top-K alternatives."""

    character: str
    class_index: int
    confidence: float
    top_k: list[dict]
    probabilities: np.ndarray

    def format(self, k: int | None = None) -> str:
        """Human-readable rendering used by the CLI and the demo."""
        lines = [
            f"Predicted Character: {self.character}",
            f"Confidence: {self.confidence * 100:.1f}%",
            "",
            "Top predictions:",
        ]
        for rank, entry in enumerate(self.top_k[: k or len(self.top_k)], start=1):
            # ASCII bar: Windows consoles default to cp1252, which cannot
            # encode block-drawing characters.
            bar = "#" * max(1, int(round(entry["probability"] * 30)))
            lines.append(
                f"  {rank}. {entry['character']:2s} - {entry['probability'] * 100:6.2f}%  {bar}"
            )
        return "\n".join(lines)


class CharacterRecognizer:
    """Loads the exported model and predicts characters from raw images."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        labels_path: str | Path = DEFAULT_LABELS_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No exported model at {model_path}. Run src/finalize.py first."
            )
        self.model = tf.keras.models.load_model(model_path)
        self.labels = load_json(labels_path)
        self.class_names: list[str] = self.labels["class_names"]
        self.config = load_json(config_path) if Path(config_path).exists() else preprocessing_config()
        self.model_path = model_path

    # -- core -----------------------------------------------------------
    def predict_tensor(self, tensor: np.ndarray, top_k: int = 5) -> Prediction:
        """Predict from an already-preprocessed ``(1, 28, 28, 1)`` tensor."""
        probabilities = self.model.predict(tensor, verbose=0)[0]
        order = np.argsort(probabilities)[::-1]
        top = [
            {
                "rank": rank,
                "character": self.class_names[int(i)],
                "class_index": int(i),
                "probability": float(probabilities[i]),
            }
            for rank, i in enumerate(order[:top_k], start=1)
        ]
        best = int(order[0])
        return Prediction(
            character=self.class_names[best],
            class_index=best,
            confidence=float(probabilities[best]),
            top_k=top,
            probabilities=probabilities,
        )

    def predict_image(
        self, source, top_k: int = 5, return_steps: bool = False
    ) -> Prediction | tuple[Prediction, dict]:
        """Predict from an image file path, PIL image or NumPy array."""
        if return_steps:
            tensor, steps = preprocess_user_image(source, return_steps=True)
            return self.predict_tensor(tensor, top_k=top_k), steps
        tensor = preprocess_user_image(source)
        return self.predict_tensor(tensor, top_k=top_k)

    # -- reporting ------------------------------------------------------
    def explain(self, source, top_k: int = 5, save_path: str | Path | None = None) -> tuple[Prediction, Path | None]:
        """Predict and render the preprocessing steps plus the Top-K bars."""
        prediction, steps = self.predict_image(source, top_k=top_k, return_steps=True)
        if save_path is None:
            return prediction, None

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
        axes[0].imshow(steps["grayscale"], cmap="gray")
        axes[0].set_title(f"1. grayscale\n{steps['grayscale'].shape[1]}x{steps['grayscale'].shape[0]}")
        axes[1].imshow(steps["mask"], cmap="gray")
        axes[1].set_title(f"2. ink mask\n(inverted={steps['inverted']})")
        axes[2].imshow(steps["cropped"], cmap="gray")
        axes[2].set_title("3. cropped to ink")
        axes[3].imshow(steps["final_28x28"], cmap="gray")
        axes[3].set_title("4. centred 28x28")
        for ax in axes[:4]:
            ax.axis("off")

        chars = [e["character"] for e in prediction.top_k][::-1]
        probs = [e["probability"] * 100 for e in prediction.top_k][::-1]
        axes[4].barh(chars, probs, color="#1f4e79")
        axes[4].set_xlim(0, 100)
        axes[4].set_xlabel("probability (%)")
        axes[4].set_title(f"5. prediction: {prediction.character} ({prediction.confidence * 100:.1f}%)")
        for i, p in enumerate(probs):
            axes[4].text(p + 1, i, f"{p:.1f}", va="center", fontsize=8)

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        return prediction, save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognise a handwritten character.")
    parser.add_argument("--image", required=True, help="path to a PNG/JPG image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="also save a figure of the preprocessing steps and Top-K bars",
    )
    args = parser.parse_args()

    recognizer = CharacterRecognizer(model_path=args.model)
    figure_path = (
        FIGURES_DIR / f"inference_{Path(args.image).stem}.png" if args.explain else None
    )
    prediction, saved = recognizer.explain(args.image, top_k=args.top_k, save_path=figure_path)

    print(f"\nImage: {args.image}")
    print(prediction.format())
    if saved:
        print(f"\nExplanation figure: {saved}")

    save_json(
        {
            "image": str(args.image),
            "model": str(recognizer.model_path),
            "predicted_character": prediction.character,
            "confidence": prediction.confidence,
            "top_k": prediction.top_k,
        },
        PREDICTIONS_DIR / f"prediction_{Path(args.image).stem}.json",
    )


if __name__ == "__main__":
    main()
