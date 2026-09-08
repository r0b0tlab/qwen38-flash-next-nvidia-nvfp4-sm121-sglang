#!/usr/bin/env python3
"""Deterministic synthetic vision fixtures for the qualification harness.

CPU/Pillow/FFmpeg only. No PII: every pixel is programmatic geometry or a
4-character alphanumeric rendered from a system font. 8 images (512x512) and
2 videos (512x512, 8 frames @ 2 fps = 4 s).

Counterfactual pairs (deliberate confusables):
- red_square_on_blue  vs blue_square_on_red   (colors swapped)
- circles_3           vs circles_5            (count changed)
- left_green_right_yellow vs left_yellow_right_green (sides swapped)
- red_bar_tall_blue_short: single-object color/height compound
- ocr_text: four alphanumeric glyphs "R7K9"

Prompts ask ONLY about the visual property under test. They never contain the
expected answer, the filename, or any identifying hint. Unit-test assertions
enforce this (see tests/test_fixture_generation.py).

This module also documents the benchmark prompts used by scripts/vision_bench.py
so fixtures and requests cannot drift apart. Generated PNG/MP4 bytes are test
data only and never constitute model-output evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"

SIZE = 512

# exact palette (RGB) used for geometric fixtures and video frames
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 200, 0),
    "blue": (30, 60, 200),
    "yellow": (250, 220, 20),
    "white": (255, 255, 255),
    "black": (10, 10, 10),
}

# video scripts: 8 frames @ 2 fps = 4 s
VIDEO_FRAME_COLORS: Dict[str, List[str]] = {
    "colors_first_red": ["red", "red", "green", "green", "blue", "blue", "yellow", "yellow"],
    "colors_first_yellow": ["yellow", "yellow", "blue", "blue", "green", "green", "red", "red"],
}

VIDEO_FPS = 2.0

# ---------------------------------------------------------------------------
# Benchmark prompts — property-only, answer-blind (enforced by tests)
# ---------------------------------------------------------------------------

PROMPT_SQUARE = "What color is the large center square, and what color surrounds it? Answer with only two color words, square first and background second."
PROMPT_CIRCLES = "How many complete green circles appear in this image? Answer with the number only."
PROMPT_OCR = "Transcribe the characters shown in this image. Answer with the characters only."
PROMPT_SIDES = "In this image, is the green region on the left half or the right half? Answer with one word."
PROMPT_BARS = "Two vertical bars are shown side by side. Which color is the taller bar? Answer with one word."
PROMPT_VIDEO = "Watch the video. What is the color of the first frame, and what is the color of the last frame? Answer with only two color words in that order."

PROMPTS: Dict[str, str] = {
    "red_square_on_blue": PROMPT_SQUARE,
    "blue_square_on_red": PROMPT_SQUARE,
    "circles_3": PROMPT_CIRCLES,
    "circles_5": PROMPT_CIRCLES,
    "ocr_text": PROMPT_OCR,
    "left_green_right_yellow": PROMPT_SIDES,
    "left_yellow_right_green": PROMPT_SIDES,
    "red_bar_tall_blue_short": PROMPT_BARS,
    "colors_first_red": PROMPT_VIDEO,
    "colors_first_yellow": PROMPT_VIDEO,
}

EXPECTED: Dict[str, str] = {
    "red_square_on_blue": "red blue",
    "blue_square_on_red": "blue red",
    "circles_3": "3",
    "circles_5": "5",
    "ocr_text": "R7K9",
    "left_green_right_yellow": "left",
    "left_yellow_right_green": "right",
    "red_bar_tall_blue_short": "red",
    "colors_first_red": "red yellow",
    "colors_first_yellow": "yellow red",
}

# EXPECTED is intentionally NOT exported into the manifest: requests and
# evidence must stay answer-blind. It exists only for local sanity checking.

# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------


def img_red_square_on_blue() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["blue"])
    d = ImageDraw.Draw(img)
    d.rectangle([156, 156, 356, 356], fill=PALETTE["red"])
    return img


def img_blue_square_on_red() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["red"])
    d = ImageDraw.Draw(img)
    d.rectangle([156, 156, 356, 356], fill=PALETTE["blue"])
    return img


def _green_circles(count: int) -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["black"])
    d = ImageDraw.Draw(img)
    r = 40
    centers = {
        3: [(128, 256), (256, 384), (384, 256)],
        5: [(96, 128), (256, 96), (416, 128), (160, 320), (352, 320)],
    }[count]
    for cx, cy in centers:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE["green"])
    return img


def img_circles_3() -> Image.Image:
    return _green_circles(3)


def img_circles_5() -> Image.Image:
    return _green_circles(5)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()  # bitmap fallback; tests assert a TTF was used


def img_ocr_text() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["white"])
    d = ImageDraw.Draw(img)
    text = "R7K9"
    font = _font(140)
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((SIZE - w) / 2 - bbox[0], (SIZE - h) / 2 - bbox[1]), text, font=font, fill=(0, 0, 0))
    return img


def img_left_green_right_yellow() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["white"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE // 2 - 1, SIZE - 1], fill=PALETTE["green"])
    d.rectangle([SIZE // 2, 0, SIZE - 1, SIZE - 1], fill=PALETTE["yellow"])
    return img


def img_left_yellow_right_green() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["white"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE // 2 - 1, SIZE - 1], fill=PALETTE["yellow"])
    d.rectangle([SIZE // 2, 0, SIZE - 1, SIZE - 1], fill=PALETTE["green"])
    return img


def img_red_bar_tall_blue_short() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PALETTE["white"])
    d = ImageDraw.Draw(img)
    d.rectangle([140, 60, 200, 440], fill=PALETTE["red"])    # tall
    d.rectangle([312, 350, 372, 440], fill=PALETTE["blue"])  # short
    return img


IMAGE_BUILDERS = {
    "red_square_on_blue": img_red_square_on_blue,
    "blue_square_on_red": img_blue_square_on_red,
    "circles_3": img_circles_3,
    "circles_5": img_circles_5,
    "ocr_text": img_ocr_text,
    "left_green_right_yellow": img_left_green_right_yellow,
    "left_yellow_right_green": img_left_yellow_right_green,
    "red_bar_tall_blue_short": img_red_bar_tall_blue_short,
}

# pixel probes recorded in the manifest for local verification (label -> xy)
IMAGE_PROBES: Dict[str, Dict[str, List[int]]] = {
    "red_square_on_blue": {"center": [256, 256], "corner": [20, 20]},
    "blue_square_on_red": {"center": [256, 256], "corner": [20, 20]},
    "circles_3": {"circle_center": [128, 256], "background": [20, 20], "empty_mid": [256, 256]},
    "circles_5": {"circle_center": [96, 128], "background": [20, 20]},
    "ocr_text": {"glyph_area": [256, 256], "background": [20, 20]},
    "left_green_right_yellow": {"left": [128, 256], "right": [384, 256]},
    "left_yellow_right_green": {"left": [128, 256], "right": [384, 256]},
    "red_bar_tall_blue_short": {"red_bar": [170, 200], "blue_bar": [342, 400], "blue_bar_top_gap": [342, 200], "background": [20, 20]},
}


def _video_color_frame(name: str) -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), PALETTE[name])


# ---------------------------------------------------------------------------
# Video construction (FFmpeg)
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required to build video fixtures but was not found")


def write_video(colors: List[str], out_path: Path) -> None:
    """Encode one solid-color MP4 per scripted frame list (8 frames @ 2 fps)."""
    _require_ffmpeg()
    assert len(colors) == 8, "video scripts are exactly 8 frames (4 s @ 2 fps)"
    with tempfile.TemporaryDirectory(prefix="vidfix") as td:
        frame_paths: List[str] = []
        for i, cname in enumerate(colors):
            p = Path(td) / f"frame_{i:02d}.png"
            _video_color_frame(cname).save(p, format="PNG")
            frame_paths.append(str(p))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(VIDEO_FPS),
            "-i", str(Path(td) / "frame_%02d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rgb_at(img: Image.Image, xy: List[int]) -> List[int]:
    return list(img.getpixel((xy[0], xy[1])))


def probe_video_frames(path: Path) -> Tuple[int, float, float, List[List[int]]]:
    """Return (frame_count, fps, duration_s, per-frame mean RGB) via ffprobe/ffmpeg."""
    _require_ffmpeg()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(stream["nb_read_frames"])
    duration = float(stream["duration"])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    fw = SIZE * SIZE * 3
    means: List[List[int]] = []
    for i in range(n):
        frame = raw[i * fw : (i + 1) * fw]
        if not frame:
            break
        means.append([
            round(sum(frame[c::3]) / (SIZE * SIZE)) for c in range(3)
        ])
    return n, fps, duration, means


def generate_all(outdir: Path) -> Dict[str, Any]:
    """Write all fixtures + manifest.json under outdir; return the manifest."""
    _require_ffmpeg()
    imgdir = outdir / "images"
    viddir = outdir / "videos"
    imgdir.mkdir(parents=True, exist_ok=True)
    viddir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "kind": "qualification-vision-fixtures",
        "model_id": MODEL_ID,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pillow_version": Image.__version__,
        "synthetic_note": "all pixels programmatic; no photography, no PII; test data only",
        "images": {},
        "videos": {},
        "prompts": dict(PROMPTS),
    }

    for name, builder in IMAGE_BUILDERS.items():
        img = builder()
        assert img.size == (SIZE, SIZE)
        out = imgdir / f"{name}.png"
        img.save(out, format="PNG")
        probes = {
            label: _rgb_at(img, xy) for label, xy in IMAGE_PROBES[name].items()
        }
        manifest["images"][name] = {
            "file": f"images/{name}.png",
            "sha256": _sha256(out),
            "width": SIZE,
            "height": SIZE,
            "mode": "RGB",
            "format": "PNG",
            "probes": probes,
        }

    for name, colors in VIDEO_FRAME_COLORS.items():
        out = viddir / f"{name}.mp4"
        write_video(colors, out)
        n, fps, duration, means = probe_video_frames(out)
        assert n == 8 and fps == VIDEO_FPS and duration == 4.0, (n, fps, duration)
        manifest["videos"][name] = {
            "file": f"videos/{name}.mp4",
            "sha256": _sha256(out),
            "width": SIZE,
            "height": SIZE,
            "fps": fps,
            "frames": n,
            "duration_s": duration,
            "frame_colors_scripted": colors,
            "frame_mean_rgb": means,
        }

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return manifest


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="generate synthetic vision fixtures + manifest")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "fixtures"))
    args = p.parse_args(argv)
    manifest = generate_all(Path(args.outdir))
    print(f"wrote {len(manifest['images'])} images, {len(manifest['videos'])} videos under {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
