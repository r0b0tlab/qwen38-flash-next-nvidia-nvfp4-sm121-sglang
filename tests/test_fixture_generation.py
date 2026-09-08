"""Local verification of synthetic vision fixtures.

These tests validate fixture GENERATION (geometry, color probes, hashes,
ffprobe frame counts). A passing fixture test is NOT a model-vision result:
no model ever participates here. Requests built from these fixtures are unit
fixtures, never evidence of real model behavior.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_vision_fixtures as mvf  # noqa: E402

pytest.importorskip("PIL")
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg/ffprobe required for vision fixture tests", allow_module_level=True)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("vision-fixtures")
    manifest = mvf.generate_all(out)
    return out, manifest


# ---------------------------------------------------------------------------
# prompt hygiene — property-only, answer-blind
# ---------------------------------------------------------------------------


class TestPromptHygiene:
    def test_all_ten_have_prompts(self):
        assert len(mvf.IMAGE_BUILDERS) == 8
        assert len(mvf.VIDEO_FRAME_COLORS) == 2
        assert set(mvf.PROMPTS) == set(mvf.IMAGE_BUILDERS) | set(mvf.VIDEO_FRAME_COLORS)

    def test_prompts_do_not_leak_answers(self):
        """No prompt may contain its expected answer or discriminating terms.

        Option words that a well-posed question must name symmetrically
        (left/right, taller) are not leaks; the anti-leak property for those
        is that both variants share the identical prompt (see
        test_counterfactual_pairs_share_prompts) and that no asymmetric
        hint is present.
        """
        leak_terms = {
            "red_square_on_blue": ["red", "blue"],
            "blue_square_on_red": ["red", "blue"],
            "circles_3": ["3", "three"],
            "circles_5": ["5", "five"],
            # the OCR answer must never appear verbatim (any case/mixed)
            "ocr_text": ["r7k9"],
            "left_green_right_yellow": ["green is on the left", "green is on the right", "yellow is on the"],
            "left_yellow_right_green": ["green is on the left", "green is on the right", "yellow is on the"],
            "red_bar_tall_blue_short": ["red", "blue", "shorter"],
            "colors_first_red": ["red", "yellow"],
            "colors_first_yellow": ["red", "yellow"],
        }
        for name, terms in leak_terms.items():
            prompt = mvf.PROMPTS[name].lower()
            for term in terms:
                assert term not in prompt, f"{name}: prompt leaks answer term {term!r}"

    def test_side_prompt_names_both_options_symmetrically(self):
        prompt = mvf.PROMPTS["left_green_right_yellow"].lower()
        assert "left" in prompt and "right" in prompt  # symmetric options
        assert mvf.PROMPTS["left_green_right_yellow"] == mvf.PROMPTS["left_yellow_right_green"]

    def test_prompts_do_not_leak_filenames(self):
        """The fixture identifier (variant name) must never appear in a prompt.

        Generic shape words that are genuinely part of the question (e.g.
        "square", "bars") are allowed; variant-discriminating words are
        covered by test_prompts_do_not_leak_answers.
        """
        for name, prompt in mvf.PROMPTS.items():
            assert name not in prompt, f"{name}: prompt contains fixture name"

    def test_variant_discriminating_words_absent(self):
        """Words that would identify the variant are absent from prompts."""
        for name in ("red_square_on_blue", "blue_square_on_red"):
            for w in ("center square is", "background is"):
                assert w not in mvf.PROMPTS[name].lower()

    def test_no_expected_answers_in_manifest(self, fixtures):
        out, manifest = fixtures
        text = json.dumps(manifest)
        assert "expected" not in text.lower().replace("expected_", "")  # no answer key at all
        assert mvf.EXPECTED["circles_3"] == "3"  # sanity on the guard map itself

    def test_counterfactual_pairs_share_prompts(self):
        """Swapped variants must ask the identical question."""
        assert mvf.PROMPTS["red_square_on_blue"] == mvf.PROMPTS["blue_square_on_red"]
        assert mvf.PROMPTS["circles_3"] == mvf.PROMPTS["circles_5"]
        assert mvf.PROMPTS["left_green_right_yellow"] == mvf.PROMPTS["left_yellow_right_green"]
        assert mvf.PROMPTS["colors_first_red"] == mvf.PROMPTS["colors_first_yellow"]


# ---------------------------------------------------------------------------
# image geometry + probes
# ---------------------------------------------------------------------------


class TestImages:
    def test_dimensions_512(self, fixtures):
        out, manifest = fixtures
        for name, entry in manifest["images"].items():
            assert (entry["width"], entry["height"]) == (512, 512)
            assert entry["mode"] == "RGB" and entry["format"] == "PNG"

    def test_sha256_matches_files(self, fixtures):
        out, manifest = fixtures
        for entry in manifest["images"].values():
            p = out / entry["file"]
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            assert h == entry["sha256"]

    def test_square_counterfactual_probes(self, fixtures):
        _, manifest = fixtures
        a = manifest["images"]["red_square_on_blue"]["probes"]
        b = manifest["images"]["blue_square_on_red"]["probes"]
        assert a["center"] == list(mvf.PALETTE["red"]) and a["corner"] == list(mvf.PALETTE["blue"])
        assert b["center"] == list(mvf.PALETTE["blue"]) and b["corner"] == list(mvf.PALETTE["red"])

    def test_circle_counts_differ(self, fixtures):
        _, manifest = fixtures
        m3 = manifest["images"]["circles_3"]["probes"]
        m5 = manifest["images"]["circles_5"]["probes"]
        assert m3["circle_center"] == list(mvf.PALETTE["green"])
        assert m5["circle_center"] == list(mvf.PALETTE["green"])
        # circles_3 has no circle at its own center; circles_5's center is a circle
        assert m3["empty_mid"] == list(mvf.PALETTE["black"])

    def test_side_halves_swapped(self, fixtures):
        _, manifest = fixtures
        a = manifest["images"]["left_green_right_yellow"]["probes"]
        b = manifest["images"]["left_yellow_right_green"]["probes"]
        assert a["left"] == list(mvf.PALETTE["green"]) and a["right"] == list(mvf.PALETTE["yellow"])
        assert b["left"] == list(mvf.PALETTE["yellow"]) and b["right"] == list(mvf.PALETTE["green"])

    def test_bar_heights(self, fixtures):
        _, manifest = fixtures
        p = manifest["images"]["red_bar_tall_blue_short"]["probes"]
        assert p["red_bar"] == list(mvf.PALETTE["red"])
        assert p["blue_bar"] == list(mvf.PALETTE["blue"])
        # at the blue bar's x, high up is background (blue bar is short)
        assert p["blue_bar_top_gap"] == list(mvf.PALETTE["white"])
        assert p["background"] == list(mvf.PALETTE["white"])

    def test_ocr_text_pixels_and_font(self, fixtures):
        from PIL import ImageFont

        out, manifest = fixtures
        p = manifest["images"]["ocr_text"]["probes"]
        assert p["background"] == list(mvf.PALETTE["white"])
        # a real scalable TTF was used (bitmap default cannot render 140px glyphs)
        assert isinstance(mvf._font(140), ImageFont.FreeTypeFont)

    def test_ocr_pixels_probe_black_glyph(self, fixtures):
        img = mvf.img_ocr_text()
        # inside the first glyph stroke there must be black pixels
        found_black = any(
            img.getpixel((x, y)) == (0, 0, 0)
            for x in range(100, 420, 4)
            for y in range(150, 380, 4)
        )
        assert found_black

    def test_manifest_videos_match_scripts(self, fixtures):
        _, manifest = fixtures
        means = manifest["videos"]["colors_first_red"]["frame_mean_rgb"]
        for mean, cname in zip(means, mvf.VIDEO_FRAME_COLORS["colors_first_red"]):
            target = mvf.PALETTE[cname]
            assert all(abs(m - t) <= 6 for m, t in zip(mean, target)), (mean, target)


# ---------------------------------------------------------------------------
# video container facts
# ---------------------------------------------------------------------------


class TestVideos:
    def test_frames_fps_duration(self, fixtures):
        _, manifest = fixtures
        for entry in manifest["videos"].values():
            assert entry["frames"] == 8
            assert entry["fps"] == pytest.approx(2.0)
            assert entry["duration_s"] == pytest.approx(4.0)
            assert (entry["width"], entry["height"]) == (512, 512)

    def test_video_sha256_matches(self, fixtures):
        out, manifest = fixtures
        for entry in manifest["videos"].values():
            assert hashlib.sha256((out / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]

    def test_first_last_frame_colors(self, fixtures):
        _, manifest = fixtures
        for name, entry in manifest["videos"].items():
            script = entry["frame_colors_scripted"]
            first_mean = entry["frame_mean_rgb"][0]
            last_mean = entry["frame_mean_rgb"][-1]
            first_target = mvf.PALETTE[script[0]]
            last_target = mvf.PALETTE[script[-1]]
            assert all(abs(a - b) <= 6 for a, b in zip(first_mean, first_target))
            assert all(abs(a - b) <= 6 for a, b in zip(last_mean, last_target))

    def test_video_pairs_are_reverses(self, fixtures):
        _, manifest = fixtures
        a = mvf.VIDEO_FRAME_COLORS["colors_first_red"]
        b = mvf.VIDEO_FRAME_COLORS["colors_first_yellow"]
        assert a == list(reversed(b))

    def test_videos_are_h264_yuv420(self, fixtures):
        out, _ = fixtures
        for vid in out.glob("videos/*.mp4"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(vid)],
                check=True, capture_output=True, text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"


# ---------------------------------------------------------------------------
# determinism / reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_png_bytes_deterministic(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        ma = mvf.generate_all(a)
        mb = mvf.generate_all(b)
        for name, entry in ma["images"].items():
            assert entry["sha256"] == mb["images"][name]["sha256"]
        # PNG encoding is deterministic; video may embed no timestamps by
        # default flags but compare sizes as a weak check too
        for name, entry in ma["videos"].items():
            assert (a / entry["file"]).stat().st_size == (b / entry["file"]).stat().st_size

    def test_manifest_no_pii_strings(self, fixtures):
        out, manifest = fixtures
        text = json.dumps(manifest)
        for banned in ("person", "user", "name:", "email", "@"):
            assert banned not in text
