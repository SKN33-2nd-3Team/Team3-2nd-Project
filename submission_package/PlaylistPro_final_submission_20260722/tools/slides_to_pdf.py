"""Render the final slide PNGs into a 16:9 PDF without changing slide content."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas


def natural_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits or 0), path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()

    images = sorted(args.input_dir.glob("*.png"), key=natural_key)
    if not images:
        raise FileNotFoundError(f"No PNG slides found: {args.input_dir}")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_size = (12.8 * 72, 7.2 * 72)
    pdf = canvas.Canvas(str(args.output_pdf), pagesize=page_size)
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        scale = min(page_size[0] / width, page_size[1] / height)
        draw_width, draw_height = width * scale, height * scale
        left = (page_size[0] - draw_width) / 2
        bottom = (page_size[1] - draw_height) / 2
        pdf.drawImage(str(image_path), left, bottom, draw_width, draw_height)
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
