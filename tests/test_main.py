import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import main


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "pages"

    def video(self, frames):
        path = self.root / "synthetic pages.avi"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"),
                                 20, (width, height))
        self.assertTrue(writer.isOpened(), "FFV1 codec unavailable")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return path

    def frame(self, value, size=(64, 96)):
        return np.full((*size, 3), value, dtype=np.uint8)

    def read(self, paths):
        return [cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                for p in paths]

    def test_stable_first_last_and_revisits(self):
        a, b = self.frame(0), self.frame(255)
        fade = [self.frame(n) for n in range(30, 255, 30)]
        paths = main.video_book2pages(self.video([a] * 8 + fade + [b] * 8
                                                + list(reversed(fade)) + [a] * 5), self.out)
        self.assertEqual([p.name for p in paths], ["0001.png", "0002.png", "0003.png"])
        for got, expected in zip(self.read(paths), [a, b, a]):
            np.testing.assert_array_equal(got, expected)

    def test_static_noise_and_transient_duplicate(self):
        a = self.frame(100)
        frames = [a] * 8 + [self.frame(250)] * 2 + [self.frame(105)] * 8
        self.assertEqual(len(main.video_book2pages(self.video(frames), self.out)), 1)

    def test_slow_fade_is_not_saved(self):
        frames = [self.frame(n) for n in range(0, 241, 10)] + [self.frame(255)] * 5
        paths = main.video_book2pages(self.video(frames), self.out)
        self.assertEqual(len(paths), 1)
        np.testing.assert_array_equal(self.read(paths)[0], self.frame(255))

    def test_small_page_edit(self):
        a = self.frame(255)
        b = a.copy()
        b[20:24, 20:24] = 0
        paths = main.video_book2pages(self.video([a] * 5 + [b] * 5), self.out)
        self.assertEqual(len(paths), 2)

    def test_single_tiny_frame_unicode_output(self):
        paths = main.video_book2pages(self.video([self.frame(42, (2, 2))]),
                                     self.root / "ページ画像")
        self.assertEqual(len(paths), 1)
        self.assertEqual(self.read(paths)[0].shape, (2, 2, 3))

    def test_short_final_transition_is_omitted(self):
        paths = main.video_book2pages(self.video([self.frame(0)] * 5
                                                + [self.frame(255)] * 2), self.out)
        self.assertEqual(len(paths), 1)

    def test_existing_output_unchanged(self):
        source = self.video([self.frame(0)] * 5)
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            main.video_book2pages(source, self.out)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_missing_and_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            main.video_book2pages(self.root / "missing.mp4", self.out)
        invalid = self.root / "invalid.avi"
        invalid.write_bytes(b"not a video")
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            main.video_book2pages(invalid, self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_options(self):
        for options in [{"stable_seconds": 0}, {"stable_seconds": float("nan")},
                        {"pixel_threshold": 256}, {"pixel_threshold": float("nan")},
                        {"changed_fraction": -1}, {"changed_fraction": 1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                main.video_book2pages("missing", self.out, **options)

    def test_no_stable_pages(self):
        with self.assertRaisesRegex(ValueError, "No stable pages"):
            main.video_book2pages(self.video([self.frame(0), self.frame(255)]), self.out)

    def test_encode_failure_releases_capture(self):
        source = self.video([self.frame(0)] * 5)
        capture = cv2.VideoCapture(str(source))
        with patch.object(main.cv2, "VideoCapture", return_value=capture), \
                patch.object(main.cv2, "imencode", return_value=(False, None)):
            with self.assertRaisesRegex(OSError, "Cannot encode"):
                main.video_book2pages(source, self.out)
        self.assertFalse(capture.isOpened())

    def test_pdf_and_cli(self):
        source = self.video([self.frame(0)] * 5 + [self.frame(255)] * 5)
        pdf = self.root / "pages.pdf"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main.main([str(source), "-o", str(self.out), "--pdf", str(pdf)])
        self.assertEqual(code, 0)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        self.assertEqual(pdf.read_bytes().count(b"/Type /Page\n"), 2)
        original = pdf.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main.main([str(source), "-o", str(self.root / "other"),
                                        "--pdf", str(pdf)]), 1)
        self.assertEqual(original, pdf.read_bytes())
        self.assertFalse((self.root / "other").exists())
        with Image.open(self.out / "0001.png") as image:
            self.assertEqual(image.size, (96, 64))


if __name__ == "__main__":
    unittest.main()
