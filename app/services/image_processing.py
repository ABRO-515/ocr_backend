from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image_from_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def preprocess_for_ocr(
    image: np.ndarray, target_size: Tuple[int, int] = (1000, 600)
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    filtered = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thresh


def save_image(image: np.ndarray, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(dest_path), image)
    if not success:
        raise RuntimeError(f"Failed to write image to {dest_path}")


