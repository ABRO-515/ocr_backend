from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    logger.info("image.load_bytes.start", bytes=len(data))
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("image.load_bytes.failed")
        raise ValueError("Invalid image data")
    logger.info("image.load_bytes.success", shape=list(img.shape), dtype=str(img.dtype))
    return img


def preprocess_for_ocr(image: np.ndarray, max_size: int = 640, min_height: int = 120) -> np.ndarray:
    if image is None:
        logger.warning("preprocess.input.none")
        raise ValueError("Input image is None")

    h, w = image.shape[:2]

    # Only downscale if too large; never go below min_height
    scale = min(1.0, max_size / max(h, w))
    new_h = max(min_height, int(h * scale))
    new_w = int(w * scale * new_h / h)  # maintain aspect ratio

    logger.info(
        "preprocess.start",
        input_shape=[h, w, 3],
        scale=round(scale, 3),
        new_shape=[new_h, new_w, 3],
    )

    # High-quality resize
    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    logger.debug("preprocess.resized_color", shape=list(resized.shape))

    # --- Optional: Very light denoise (only if needed) ---
    # Comment this out first to test
    # denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=30, sigmaSpace=30)
    # logger.debug("preprocess.denoised", shape=list(denoised.shape))
    # enhanced = resized  # skip CLAHE for now

    # --- Mild CLAHE (safe) ---
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    logger.debug("preprocess.enhanced", shape=list(enhanced.shape))

    logger.info("preprocess.done", output_shape=list(enhanced.shape), dtype=str(enhanced.dtype))
    return enhanced




def save_image(image: np.ndarray, dest_path: Path) -> None:
    logger.info("image.save.start", path=str(dest_path))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(dest_path), image)
    if not success:
        logger.error("image.save.failed", path=str(dest_path))
        raise RuntimeError(f"Failed to write image to {dest_path}")
    logger.info("image.save.success", path=str(dest_path))


