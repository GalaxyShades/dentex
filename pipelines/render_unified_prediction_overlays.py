"""Render unified DENTEX predictions as image overlays."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_ROOT = PROJECT_ROOT / "data" / "dentex" / "test_data" / "input"
DEFAULT_PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_checkpoint_test_inference"
    / "predictions"
    / "clean_review_predictions_unified_test.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "best_checkpoint_test_inference"
    / "annotated_images"
    / "clean_review_fused"
)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DIAGNOSIS_NAMES = {
    0: "impacted",
    1: "caries",
    2: "periapical_lesion",
    3: "deep_caries",
}
COLOURS = [
    (230, 57, 70),
    (42, 157, 143),
    (69, 123, 157),
    (255, 183, 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw fused prediction overlays from unified JSON predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="Unified prediction JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered overlays.",
    )
    return parser.parse_args()


def parse_image_name(image_id: int) -> str:
    return f"test_{image_id}.png"


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    colour: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    x, y = position
    text_box = draw.textbbox((x, y), text, font=font)
    padding = 4
    draw.rectangle(
        (
            text_box[0] - padding,
            text_box[1] - padding,
            text_box[2] + padding,
            text_box[3] + padding,
        ),
        fill=colour,
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def render_overlay(
    image_id: int,
    predictions: list[dict],
    output_dir: Path,
    font: ImageFont.FreeTypeFont,
) -> None:
    image_name = parse_image_name(image_id)
    image = Image.open(TEST_IMAGE_ROOT / image_name).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for prediction in sorted(
        predictions, key=lambda item: float(item["score"]), reverse=True
    ):
        x, y, width, height = [float(value) for value in prediction["bbox"]]
        diagnosis = int(prediction["diagnosis"])
        colour = COLOURS[diagnosis % len(COLOURS)]
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=(*colour, 255),
            width=5,
        )
        label = (
            f"{DIAGNOSIS_NAMES.get(diagnosis, str(diagnosis))} "
            f"q{int(prediction['quadrant']) + 1} "
            f"t{int(prediction['enumeration']) + 1} "
            f"{float(prediction['score']):.2f}"
        )
        draw_label(draw, (x, max(0.0, y - 28.0)), label, colour, font)

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / image_name)


def main() -> None:
    args = parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_by_image = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    font = ImageFont.truetype(str(FONT_PATH), size=24)
    for image_id, image_predictions in sorted(predictions_by_image.items()):
        render_overlay(image_id, image_predictions, args.output_dir, font)

    print(f"Rendered {len(predictions_by_image)} overlays: {args.output_dir}")


if __name__ == "__main__":
    main()
