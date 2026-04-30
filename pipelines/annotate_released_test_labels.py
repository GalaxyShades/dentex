"""Draw released DENTEX test labels for model-overlay comparison."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "data" / "dentex" / "test_data"
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "results"
    / "best_checkpoint_test_inference"
    / "annotated_images"
    / "pathology"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "best_checkpoint_test_inference"
    / "annotated_images"
    / "ground_truth"
)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
COLOURS = [
    (230, 57, 70),
    (42, 157, 143),
    (38, 70, 83),
    (244, 162, 97),
    (69, 123, 157),
    (131, 56, 236),
    (255, 183, 3),
    (0, 150, 199),
]
TEST_DIAGNOSIS_TO_MODEL_LABEL = {
    1: "caries",
    6: "impacted",
    7: "periapical_lesion",
}
TASKS = ("pathology", "tooth", "quadrant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate released test images with ground-truth LabelMe polygons."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory whose image filenames define the comparison subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ground-truth annotated images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to annotate from the reference directory.",
    )
    return parser.parse_args()


def task_colour(task_name: str, label: str) -> tuple[int, int, int]:
    task_offset = TASKS.index(task_name) * 3
    return COLOURS[(task_offset + sum(label.encode("utf-8"))) % len(COLOURS)]


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
    background = (
        text_box[0] - padding,
        text_box[1] - padding,
        text_box[2] + padding,
        text_box[3] + padding,
    )
    draw.rectangle(background, fill=colour)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def parse_model_labels(raw_label: str) -> dict[str, str] | None:
    parts = raw_label.split("-")
    if len(parts) < 3:
        return None

    raw_diagnosis_id = int(parts[0])
    pathology_label = TEST_DIAGNOSIS_TO_MODEL_LABEL.get(raw_diagnosis_id)
    if pathology_label is None:
        return None

    tooth_code = parts[2]
    if len(tooth_code) != 2 or not tooth_code.isdigit():
        return None

    quadrant = int(tooth_code[0])
    tooth = int(tooth_code[1])
    if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
        return None

    return {
        "pathology": pathology_label,
        "tooth": f"tooth_{tooth}",
        "quadrant": f"quadrant_{quadrant}",
    }


def draw_polygon(
    image: Image.Image,
    shape: dict,
    task_name: str,
    label: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    points = [(float(x), float(y)) for x, y in shape["points"]]
    colour = task_colour(task_name, label)
    fill = (*colour, 45)
    outline = (*colour, 255)
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=5)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    draw_label(draw, (min_x, max(0.0, min_y - 28.0)), label, colour, font)


def annotate_image(
    image_name: str, output_dir: Path, font: ImageFont.FreeTypeFont
) -> None:
    image_path = TEST_ROOT / "input" / image_name
    label_path = TEST_ROOT / "label" / f"{Path(image_name).stem}.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))

    images = {
        task_name: Image.open(image_path).convert("RGB") for task_name in TASKS
    }

    for shape in payload["shapes"]:
        model_labels = parse_model_labels(str(shape["label"]))
        if model_labels is None:
            continue

        for task_name, label in model_labels.items():
            draw_polygon(images[task_name], shape, task_name, label, font)

    for task_name, image in images.items():
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)
        image.save(task_output_dir / image_name)


def main() -> None:
    args = parse_args()
    image_names = sorted(path.name for path in args.reference_dir.glob("*"))[
        : args.max_images
    ]
    font = ImageFont.truetype(str(FONT_PATH), size=24)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    for image_name in image_names:
        annotate_image(image_name, args.output_dir, font)

    print(
        f"Saved {len(image_names)} ground-truth annotated images per task: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
