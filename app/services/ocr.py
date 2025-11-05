from functools import lru_cache
from typing import List, Tuple

import numpy as np
from paddleocr import PaddleOCR


@lru_cache(maxsize=1)
def get_ocr() -> PaddleOCR:
    # Enable angle classification for better plate/CNIC reading
    return PaddleOCR(use_angle_cls=True, lang="en")


def ocr_text(image: np.ndarray) -> List[Tuple[str, float]]:
    ocr = get_ocr()
    result = ocr.ocr(image, cls=True)
    texts: List[Tuple[str, float]] = []
    for page in result:
        for line in page:
            text, conf = line[1][0], float(line[1][1])
            texts.append((text, conf))
    return texts


