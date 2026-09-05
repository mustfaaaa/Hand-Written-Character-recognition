"""Web application for handwritten character recognition.

A small Flask backend in front of the *validated final model*. Every number
the UI displays — predictions, confidences, accuracy, confusion pairs — is
read from the trained model or from the metrics files that `finalize.py` and
`evaluate.py` wrote. Nothing is mocked.

Endpoints
---------
``GET  /``                 the single-page UI
``GET  /api/health``       liveness + which model is loaded
``GET  /api/model-info``   real dataset/architecture/test metrics
``GET  /api/figure/<key>`` whitelisted evaluation figures (e.g. confusion matrix)
``POST /api/predict``      canvas data-URL **or** multipart upload -> prediction

Run:
    python src/app.py            # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from inference import DEFAULT_MODEL_PATH, CharacterRecognizer
from utils import FIGURES_DIR, METRICS_DIR, PROJECT_ROOT, load_json

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
WEB_DIR = PROJECT_ROOT / "web"

#: Reject uploads larger than this before they reach the model.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

#: Reject absurdly large images (decompression-bomb guard).
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 8000

#: Formats we accept. Validated by *decoding*, never by file extension.
ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "BMP", "WEBP", "GIF"}

#: Below this softmax probability the UI tells the user the model is unsure.
LOW_CONFIDENCE_THRESHOLD = 0.60

#: Evaluation figures the UI is allowed to request.
FIGURE_WHITELIST = {
    "confusion_matrix": "final_confusion_matrix_norm.png",
    "top_confusions": "final_top_confusions.png",
    "training_curves": "final_training_curves.png",
    "confidence": "final_confidence_analysis.png",
    "errors": "final_errors_most_confident.png",
}

log = logging.getLogger("hcr.app")

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_recognizer: CharacterRecognizer | None = None
_model_path = DEFAULT_MODEL_PATH


class ApiError(Exception):
    """An error that is safe and useful to show the user."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


def get_recognizer() -> CharacterRecognizer:
    """Load the model once and reuse it for every request."""
    global _recognizer
    if _recognizer is None:
        log.info("Loading model from %s", _model_path)
        _recognizer = CharacterRecognizer(model_path=_model_path)
        log.info("Model ready: %d classes", len(_recognizer.class_names))
    return _recognizer


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def _decode_image(raw: bytes) -> Image.Image:
    """Decode and validate arbitrary uploaded bytes into a PIL image."""
    if not raw:
        raise ApiError("That file is empty. Please choose a different image.", 400, "empty_file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError("That image is larger than 8 MB. Please use a smaller file.", 413, "too_large")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()  # cheap structural check
        image = Image.open(io.BytesIO(raw))  # verify() exhausts the file
    except (UnidentifiedImageError, OSError, ValueError):
        raise ApiError(
            "That file could not be read as an image. Please upload a PNG or JPG.",
            415,
            "unreadable",
        )

    fmt = (image.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(
            f"{fmt or 'That file type'} is not supported. Please upload a PNG or JPG.",
            415,
            "unsupported_format",
        )

    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS or max(width, height) > MAX_IMAGE_SIDE:
        raise ApiError(
            "That image's resolution is too large. Please use an image under 8000 px per side.",
            413,
            "too_large",
        )
    return image


def _image_from_request() -> Image.Image:
    """Pull an image out of either a multipart upload or a canvas data URL."""
    upload = request.files.get("file")
    if upload is not None and upload.filename:
        return _decode_image(upload.read())

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image", "")
    if not data_url:
        raise ApiError("No image was received. Draw a character or choose a file.", 400, "no_image")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        return _decode_image(base64.b64decode(data_url, validate=True))
    except (binascii.Error, ValueError):
        raise ApiError("The drawing could not be read. Please try again.", 400, "bad_data_url")


def _thumbnail_data_url(array, size: int = 112) -> str:
    """PNG data URL of the 28x28 the model actually saw, upscaled crisply."""
    buffer = io.BytesIO()
    Image.fromarray(array).resize((size, size), Image.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.errorhandler(ApiError)
def _handle_api_error(error: ApiError):
    return jsonify({"error": {"message": error.message, "code": error.code}}), error.status


@app.errorhandler(413)
def _handle_too_large(_):
    return (
        jsonify({"error": {"message": "That image is larger than 8 MB.", "code": "too_large"}}),
        413,
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        recognizer = get_recognizer()
    except FileNotFoundError as exc:
        return jsonify({"status": "model_missing", "detail": str(exc)}), 503
    return jsonify(
        {
            "status": "ok",
            "model": Path(recognizer.model_path).name,
            "n_classes": len(recognizer.class_names),
        }
    )


@app.get("/api/model-info")
def model_info():
    """Real dataset, architecture and test metrics, read from disk.

    Anything that has not been generated yet is returned as ``null`` so the
    UI can hide that panel rather than invent a number.
    """
    try:
        recognizer = get_recognizer()
        class_names = recognizer.class_names
        model_name = Path(recognizer.model_path).name
    except FileNotFoundError:
        class_names, model_name = [], None

    def maybe(path: Path):
        return load_json(path) if path.exists() else None

    evaluation = maybe(METRICS_DIR / "evaluation_final.json")
    summary = maybe(METRICS_DIR / "final_model_summary.json")
    dataset = maybe(METRICS_DIR / "dataset_summary.json")
    baselines = maybe(METRICS_DIR / "baselines_emnist_balanced.json")
    experiments = maybe(METRICS_DIR / "experiment_table_emnist_balanced.json")

    info: dict = {
        "model_file": model_name,
        "class_names": class_names,
        "n_classes": len(class_names) or None,
        "input_shape": [28, 28, 1],
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "metrics": None,
        "dataset": None,
        "architecture": None,
        "top_confusions": None,
        "hardest_classes": None,
        "confidence": None,
        "baselines": None,
        "experiments": experiments,
        "figures": {},
    }

    if evaluation:
        m = evaluation["metrics"]
        info["metrics"] = {
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "n_test": evaluation["n_test"],
            "n_errors": evaluation["n_errors"],
            "error_rate": evaluation["error_rate"],
        }
        info["top_confusions"] = evaluation["top_confusions"][:10]
        info["hardest_classes"] = evaluation["hardest_classes"][:8]
        info["confidence"] = evaluation["confidence"]

    if dataset:
        e = dataset["emnist_summary"]
        info["dataset"] = {
            "name": "EMNIST Balanced",
            "n_train": e["n_train"],
            "n_test": e["n_test"],
            "n_classes": e["n_classes"],
            "image_shape": e["image_shape"],
            "per_class_train": e["train_count_min"],
        }

    if summary:
        info["architecture"] = {
            "variant": summary["architecture_variant"],
            "parameters": summary["parameters"]["total"],
            "selected_experiment": summary["selected_experiment"],
            "optimizer": summary["hyperparameters"]["optimizer"],
            "learning_rate": summary["hyperparameters"]["learning_rate"],
            "batch_size": summary["hyperparameters"]["batch_size"],
            "epochs_run": summary["hyperparameters"]["epochs_run"],
            "augmented": summary["augmentation"] is not None,
        }

    if baselines:
        info["baselines"] = {
            name: {"accuracy": r["test"]["accuracy"], "f1_macro": r["test"]["f1_macro"]}
            for name, r in baselines.items()
        }

    info["figures"] = {
        key: f"/api/figure/{key}"
        for key, filename in FIGURE_WHITELIST.items()
        if (FIGURES_DIR / filename).exists()
    }
    return jsonify(info)


@app.get("/api/figure/<key>")
def figure(key: str):
    """Serve a whitelisted evaluation figure."""
    filename = FIGURE_WHITELIST.get(key)
    if not filename:
        raise ApiError("Unknown figure.", 404, "not_found")
    path = FIGURES_DIR / filename
    if not path.exists():
        raise ApiError("That figure has not been generated yet.", 404, "not_found")
    return send_file(path, mimetype="image/png")


@app.post("/api/predict")
def predict():
    """Run the real model over a drawn or uploaded character."""
    try:
        recognizer = get_recognizer()
    except FileNotFoundError:
        raise ApiError(
            "The recognition model is not available. Run src/finalize.py to export it.",
            503,
            "model_missing",
        )

    image = _image_from_request()
    top_k = max(1, min(int(request.args.get("top_k", 5)), len(recognizer.class_names)))

    try:
        prediction, steps = recognizer.predict_image(image, top_k=top_k, return_steps=True)
    except ValueError:
        # Raised when thresholding finds no ink at all.
        raise ApiError(
            "No character was found in that image. Try drawing thicker strokes, "
            "or use a photo with clearer contrast.",
            422,
            "no_ink",
        )
    except Exception:  # pragma: no cover - unexpected model failure
        log.exception("Prediction failed")
        raise ApiError("Something went wrong while analysing that image.", 500, "inference_failed")

    return jsonify(
        {
            "predicted_character": prediction.character,
            "confidence": prediction.confidence,
            "is_uncertain": prediction.confidence < LOW_CONFIDENCE_THRESHOLD,
            "top_k": prediction.top_k,
            "preprocessed_png": _thumbnail_data_url(steps["final_28x28"]),
            "inverted": bool(steps["inverted"]),
        }
    )


def main() -> None:
    global _model_path
    parser = argparse.ArgumentParser(description="Run the recognition web app.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _model_path = Path(args.model)

    # Fail fast and clearly if the model has not been exported yet.
    try:
        get_recognizer()
    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    print(f"\n  Handwritten Character Recognition")
    print(f"  model : {_model_path}")
    print(f"  open  : http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
