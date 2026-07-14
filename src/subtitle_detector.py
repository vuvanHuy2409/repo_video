"""
subtitle_detector.py — OCR-based subtitle zone detector.

Samples frames from a video, runs OCR on the bottom 35% (where Chinese
subtitles typically appear), and returns the bounding box of the subtitle
zone so the merger can cover + replace it.
"""
import os
import cv2
import numpy as np
from src.utils import setup_logging

logger = setup_logging("subtitle_detector")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_COUNT   = 12     # number of frames to sample
SCAN_ZONE_TOP  = 0.65   # only scan bottom 35% of frame height
PADDING_H      = 6      # extra vertical padding around detected text (px)
PADDING_W      = 20     # extra horizontal padding
MIN_CONFIDENCE = 0.4    # minimum OCR confidence to count as subtitle text
# ──────────────────────────────────────────────────────────────────────────────


def _get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def _sample_frames(video_path: str, count: int) -> list[np.ndarray]:
    """Extract `count` frames spread evenly across the video."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    indices = [int(total * i / count) for i in range(count)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _crop_scan_zone(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (cropped_region, y_offset) for the bottom portion."""
    h      = frame.shape[0]
    y_off  = int(h * SCAN_ZONE_TOP)
    return frame[y_off:, :], y_off


def detect_subtitle_zone(video_path: str) -> dict | None:
    """
    Detect the global subtitle bounding box (in full-frame coordinates).
    Useful as a fallback.
    """
    try:
        import easyocr
    except ImportError:
        return None

    frames = _sample_frames(video_path, SAMPLE_COUNT)
    if not frames:
        return None

    frame_h, frame_w = frames[0].shape[:2]

    import ssl
    try:
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
    finally:
        ssl._create_default_https_context = _orig_ctx

    all_boxes: list[tuple[int, int, int, int]] = []

    for fi, frame in enumerate(frames):
        crop, y_off = _crop_scan_zone(frame)
        results     = reader.readtext(crop, detail=1)
        for (bbox, text, conf) in results:
            if conf < MIN_CONFIDENCE or not text.strip():
                continue
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1 = int(min(xs))
            y1 = int(min(ys)) + y_off
            x2 = int(max(xs))
            y2 = int(max(ys)) + y_off
            all_boxes.append((x1, y1, x2, y2))

    if not all_boxes:
        return None

    x1 = max(0,        min(b[0] for b in all_boxes) - PADDING_W)
    y1 = max(0,        min(b[1] for b in all_boxes) - PADDING_H)
    x2 = min(frame_w,  max(b[2] for b in all_boxes) + PADDING_W)
    y2 = min(frame_h,  max(b[3] for b in all_boxes) + PADDING_H)

    return {
        "x":       x1,
        "y":       y1,
        "width":   x2 - x1,
        "height":  y2 - y1,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def detect_subtitle_positions_for_chunks(video_path: str, chunks: list[dict]) -> list[dict]:
    """
    For each chunk, extract the middle frame, run OCR to find the exact y-position
    and height of the Chinese subtitle. Updates each chunk dict with "zone": {x, y, w, h} or None.
    """
    try:
        import easyocr
    except ImportError:
        logger.error("easyocr is not installed. Skipping precise subtitle position detection.")
        for chunk in chunks:
            chunk["zone"] = None
        return chunks

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize EasyOCR
    import ssl
    try:
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
    finally:
        ssl._create_default_https_context = _orig_ctx

    logger.info(f"Scanning {len(chunks)} chunks for precise vertical subtitle coordinates...")

    for idx, chunk in enumerate(chunks):
        t_mid = (chunk["start"] + chunk["end"]) / 2.0
        frame_idx = int(t_mid * fps)
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            chunk["zone"] = None
            continue

        # Crop to the bottom subtitle area
        y_off = int(frame_h * SCAN_ZONE_TOP)
        crop = frame[y_off:, :]

        results = reader.readtext(crop, detail=1)
        candidates = []
        for (bbox, text, conf) in results:
            if conf < MIN_CONFIDENCE or not text.strip():
                continue
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1 = int(min(xs))
            y1 = int(min(ys)) + y_off
            x2 = int(max(xs))
            y2 = int(max(ys)) + y_off

            # Center distance calculation
            text_center = (x1 + x2) / 2.0
            screen_center = frame_w / 2.0
            center_dist = abs(text_center - screen_center)

            candidates.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "height": y2 - y1,
                "width": x2 - x1,
                "center_dist": center_dist,
                "text": text,
            })

        if candidates:
            # We filter out very small noise boxes, and prioritize centered text
            candidates = [c for c in candidates if c["height"] > 10]
            if candidates:
                # Sort primarily by how centered they are horizontally
                candidates.sort(key=lambda c: (c["center_dist"], -c["height"]))
                best = candidates[0]
                
                # Add horizontal and vertical padding
                x1_pad = max(0, best["x1"] - PADDING_W)
                y1_pad = max(0, best["y1"] - PADDING_H)
                x2_pad = min(frame_w, best["x2"] + PADDING_W)
                y2_pad = min(frame_h, best["y2"] + PADDING_H)

                chunk["zone"] = {
                    "x": x1_pad,
                    "y": y1_pad,
                    "width": x2_pad - x1_pad,
                    "height": y2_pad - y1_pad,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                }
                logger.info(
                    f"  Chunk {idx+1} ({chunk['start']:.1f}s-{chunk['end']:.1f}s): "
                    f"detected Chinese subtitle at y={chunk['zone']['y']}, h={chunk['zone']['height']} (text: '{best['text']}')"
                )
            else:
                chunk["zone"] = None
        else:
            chunk["zone"] = None

    cap.release()
    return chunks
