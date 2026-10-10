"""Extract stable, distinct page images from a local video, without a GUI."""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np


def signature(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    return cv2.resize(frame, (max(1, round(width * scale)),
                              max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def changed(a, b, pixel_threshold, changed_fraction):
    if a.shape != b.shape:
        return True
    delta = cv2.absdiff(a, b).max(axis=2)
    return np.count_nonzero(delta > pixel_threshold) / delta.size > changed_fraction


def video_book2pages(video_path, output_dir, *, stable_seconds=0.25,
                     pixel_threshold=20, changed_fraction=0.001):
    """Save stable runs as PNGs in a new directory and return their paths.

    An anchored comparison detects cumulative changes during slow fades.
    Consecutive duplicates are suppressed, but later revisits are preserved.
    """
    if not math.isfinite(stable_seconds) or stable_seconds <= 0:
        raise ValueError("stable_seconds must be positive and finite")
    if not math.isfinite(pixel_threshold) or not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be between 0 and 255")
    if not math.isfinite(changed_fraction) or not 0 <= changed_fraction < 1:
        raise ValueError("changed_fraction must be between 0 (inclusive) and 1")
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Input video does not exist: {source}")
    if destination.exists():
        raise ValueError(f"Output already exists; choose a new directory: {destination}")
    cap = cv2.VideoCapture(str(source))
    pages = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Video contains no readable frames: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video frame rate is unavailable")
        required = max(2, math.ceil(stable_seconds * fps))
        destination.mkdir(parents=True, exist_ok=False)
        anchor, candidate = signature(frame), frame
        run_length = total_frames = 1
        last_saved = None

        def save_stable():
            nonlocal last_saved
            if last_saved is not None and not changed(
                    last_saved, anchor, pixel_threshold, changed_fraction):
                return
            path = destination / f"{len(pages) + 1:04d}.png"
            success, encoded = cv2.imencode(".png", candidate)
            if not success:
                raise OSError(f"Cannot encode image: {path}")
            with path.open("xb") as stream:
                stream.write(encoded.tobytes())
            pages.append(path)
            last_saved = anchor

        while True:
            if run_length == required:
                save_stable()
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            current = signature(frame)
            if changed(anchor, current, pixel_threshold, changed_fraction):
                anchor, candidate, run_length = current, frame, 1
            else:
                candidate = frame
                run_length += 1
        if total_frames == 1:
            save_stable()
        if not pages:
            raise ValueError("No stable pages found; try a smaller --stable-seconds. "
                             f"The empty output directory remains at {destination}")
        return pages
    finally:
        cap.release()


def images2pdf(image_paths, output_filename):
    """Create a PDF from this run's pages without overwriting any file."""
    from PIL import Image

    if not image_paths:
        raise ValueError("No images to convert")
    images = []
    try:
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        with Path(output_filename).open("xb") as stream:
            images[0].save(stream, format="PDF", save_all=True,
                           append_images=images[1:])
    finally:
        for image in images:
            image.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Local video file")
    parser.add_argument("-o", "--output", type=Path,
                        help="New output directory (default: out/<video filename>)")
    parser.add_argument("--stable-seconds", type=float, default=0.25,
                        help="Minimum stable hold duration (default: 0.25)")
    parser.add_argument("--pixel-threshold", type=float, default=20,
                        help="Ignore per-channel differences up to this level (0-255)")
    parser.add_argument("--changed-fraction", type=float, default=0.001,
                        help="Allowed changed pixel fraction (default: 0.001)")
    parser.add_argument("--pdf", type=Path, help="Also create a PDF at a new file path")
    args = parser.parse_args(argv)
    try:
        if args.pdf and args.pdf.expanduser().exists():
            raise ValueError(f"PDF already exists: {args.pdf}")
        if args.pdf and not args.pdf.expanduser().parent.is_dir():
            raise ValueError(f"PDF parent directory does not exist: {args.pdf.parent}")
        pages = video_book2pages(
            args.video, args.output or Path("out") / args.video.name,
            stable_seconds=args.stable_seconds,
            pixel_threshold=args.pixel_threshold,
            changed_fraction=args.changed_fraction)
        if args.pdf:
            images2pdf(pages, args.pdf.expanduser())
        print(f"Saved {len(pages)} pages to {pages[0].parent}")
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
