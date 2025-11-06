from typing import List, Tuple
import os
from pathlib import Path
from threading import Lock

import numpy as np
from paddleocr import PaddleOCR

from app.core.logging import get_logger

logger = get_logger(__name__)


# Thread-safe singleton instance
_ocr_instance: PaddleOCR | None = None
_ocr_lock: Lock = Lock()


def _resolve_model_paths() -> dict:
    r"""Resolve explicit model directories if provided via env or conventional paths.

    Priority:
    1) Environment variables: OCR_DET_MODEL_DIR, OCR_REC_MODEL_DIR, OCR_CLS_MODEL_DIR
    2) Default local paths under C:\Users\hp\.paddlex\official_models
    """
    base_models_dir = Path(os.getenv("OCR_MODELS_DIR", r"C:\Users\hp\.paddlex\official_models")).resolve()

    det_dir_env = os.getenv("OCR_DET_MODEL_DIR")
    rec_dir_env = os.getenv("OCR_REC_MODEL_DIR")
    cls_dir_env = os.getenv("OCR_CLS_MODEL_DIR")

    # Use actual folder names from your system
    det_model_dir = Path(det_dir_env) if det_dir_env else base_models_dir / "PP-OCRv5_mobile_det"
    # rec_model_dir = Path(rec_dir_env) if rec_dir_env else base_models_dir / "en_PP-OCRv5_mobile_rec"  # Fixed
    rec_model_dir = Path(rec_dir_env) if rec_dir_env else base_models_dir / "PP-OCRv5_mobile_rec"  # Fixed
    cls_model_dir = Path(cls_dir_env) if cls_dir_env else None  # Optional

    paths: dict = {}

    if det_model_dir.exists():
        paths["det_model_dir"] = str(det_model_dir)
    if rec_model_dir.exists():
        paths["rec_model_dir"] = str(rec_model_dir)
    if cls_model_dir and cls_model_dir.exists():
        paths["cls_model_dir"] = str(cls_model_dir)

    logger.info(
        "ocr.model_paths",
        has_det=det_model_dir.exists(),
        has_rec=rec_model_dir.exists(),
        has_cls=cls_model_dir.exists() if cls_dir_env else False,
        base=str(base_models_dir),
        det_path=str(det_model_dir),
        rec_path=str(rec_model_dir),
    )

    return paths


def _create_ocr_instance() -> PaddleOCR:
    model_paths = _resolve_model_paths()

    logger.info("ocr.init.start", use_gpu=False, enable_mkldnn=True, lang="en")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        # use_gpu=False,
        enable_mkldnn=True,
        **model_paths,
    )
    logger.info("ocr.init.success")
    _prewarm_ocr(ocr)
    return ocr


def _prewarm_ocr(ocr: PaddleOCR) -> None:
    """Run a tiny inference once to load weights into memory.
    This reduces first-request latency and surfaces init errors early.
    """
    try:
        dummy = np.zeros((16, 64, 3), dtype=np.uint8)
        logger.info("ocr.prewarm.start", shape=list(dummy.shape))
        _ = ocr.ocr(dummy)
        logger.info("ocr.prewarm.ok")
    except Exception as e:
        logger.error("ocr.prewarm.failed", error=str(e))


def get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance
    with _ocr_lock:
        if _ocr_instance is None:
            _ocr_instance = _create_ocr_instance()
    return _ocr_instance


def _normalize_ocr_result(result):
    # If already PaddleOCR-like: [[box, (text, score)], ...]
    if isinstance(result, (list, tuple)):
        if len(result) > 0 and isinstance(result[0], (list, tuple)):
            if len(result[0]) >= 2 and isinstance(result[0][1], (list, tuple)):
                # filter out entries with empty text
                page = []
                for line in result[0]:
                    try:
                        ti = line[1]
                        text = ti[0] if isinstance(ti, (list, tuple)) and len(ti) > 0 else None
                        if text is None or str(text).strip() == "":
                            continue
                        page.append(line)
                    except Exception:
                        continue
                return [page] if page else [[]]
        # List of dicts case
        if len(result) > 0 and isinstance(result[0], dict):
            page = []
            for item in result:
                # Try multiple key aliases seen in PaddleX variants
                box = (
                    item.get("bbox") or item.get("box") or item.get("poly") or item.get("polygon") or
                    item.get("points") or item.get("quad")
                )
                text = (
                    item.get("text") or item.get("label") or item.get("rec_text") or
                    item.get("transcription") or item.get("value")
                )
                score = item.get("score") or item.get("rec_score") or item.get("prob") or item.get("confidence")
                if text is None or str(text).strip() == "":
                    continue
                page.append([box, (text, score)])
            return [page]
    # Dict of arrays case
    if isinstance(result, dict):
        boxes = (
            result.get("boxes") or result.get("dt_polys") or result.get("polys") or
            result.get("dt_boxes") or result.get("bboxes")
        )
        texts = result.get("texts") or result.get("rec_text") or result.get("text") or result.get("transcriptions")
        scores = result.get("scores") or result.get("rec_score") or result.get("score") or result.get("confidences")
        if boxes is not None and texts is not None:
            page = []
            n = min(len(boxes) if hasattr(boxes, "__len__") else 0, len(texts) if hasattr(texts, "__len__") else 0)
            for i in range(n):
                text = texts[i]
                if text is None or str(text).strip() == "":
                    continue
                box = boxes[i] if hasattr(boxes, "__len__") and i < len(boxes) else None
                sc = None
                if isinstance(scores, (list, tuple)) and i < len(scores):
                    sc = scores[i]
                elif isinstance(scores, (int, float)):
                    sc = scores
                page.append([box, (text, sc)])
            return [page]
    return result


def ocr_text(image: np.ndarray) -> List[Tuple[str, float]]:
    if image is None:
        logger.warning("ocr.input.none")
        return []

    if not isinstance(image, np.ndarray):
        logger.error("ocr.input.invalid_type", received_type=str(type(image)))
        return []

    if image.size == 0:
        logger.warning("ocr.input.empty", shape=list(image.shape))
        return []

    logger.info("ocr.run.start", shape=list(image.shape), dtype=str(image.dtype))

    ocr = get_ocr()
    try:
        result = ocr.ocr(image)
        result = _normalize_ocr_result(result)

        # === DEBUG: Dump raw OCR result ===
        try:
            logger.debug("ocr.raw_result.structure", 
                         type=str(type(result)), 
                         is_list=isinstance(result, (list, tuple)))
            
            if isinstance(result, (list, tuple)) and len(result) > 0:
                first_page = result[0]
                logger.debug("ocr.raw_result.first_page", 
                             type=str(type(first_page)), 
                             length=len(first_page) if hasattr(first_page, '__len__') else None)
                
                for idx, raw_line in enumerate(first_page[:3]):
                    logger.debug("ocr.raw_result.item", 
                                 index=idx, 
                                 raw_value=raw_line, 
                                 raw_type=str(type(raw_line)),
                                 raw_len=len(raw_line) if hasattr(raw_line, '__len__') else None)
        except Exception as e:
            logger.error("ocr.raw_dump.failed", error=str(e))
        # ==================================

    except Exception as e:
        logger.error("ocr.run.failed", error=str(e))
        return []

    # Summarize the raw result structure for debugging
    try:
        pages = len(result) if isinstance(result, (list, tuple)) else 0
        first_page_len = len(result[0]) if pages and isinstance(result[0], (list, tuple)) else 0
        logger.info("ocr.result.summary", pages=pages, first_page_items=first_page_len)
        for i, line in enumerate(result[0][:5] if pages else []):
            line_len = len(line) if isinstance(line, (list, tuple)) else None
            logger.debug(
                "ocr.result.sample_line",
                index=i,
                type=str(type(line)),
                len=line_len,
            )

        for page_index, page in enumerate(result if isinstance(result, (list, tuple)) else []):
            for line_index, line in enumerate(page if isinstance(page, (list, tuple)) else []):
                try:
                    box = line[0] if isinstance(line, (list, tuple)) and len(line) > 0 else None
                    ti = line[1] if isinstance(line, (list, tuple)) and len(line) > 1 else None
                    text = ti[0] if isinstance(ti, (list, tuple)) and len(ti) > 0 else None
                    confidence = ti[1] if isinstance(ti, (list, tuple)) and len(ti) > 1 else None
                    logger.info(
                        "ocr.result.item",
                        page_index=page_index,
                        line_index=line_index,
                        box=box,
                        text=text,
                        confidence=confidence,
                    )
                except Exception as e:
                    logger.debug(
                        "ocr.result.item.parse_error",
                        page_index=page_index,
                        line_index=line_index,
                        error=str(e),
                    )
    except Exception:
        pass

    if not result:
        logger.info("ocr.run.no_pages")
        return []

    texts: List[Tuple[str, float]] = []

    for page_index, page in enumerate(result):
        if not page:
            logger.debug("ocr.page.empty", page_index=page_index)
            continue

        for line_index, line in enumerate(page):
            # Expected shape: [box, (text, conf)]
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                logger.debug("ocr.line.malformed", page_index=page_index, line_index=line_index)
                continue

            text_info = line[1]
            if not isinstance(text_info, (tuple, list)) or len(text_info) < 2:
                logger.debug("ocr.text_info.malformed", page_index=page_index, line_index=line_index)
                continue

            text_raw = text_info[0]
            conf_raw = text_info[1]

            text = ("" if text_raw is None else str(text_raw)).strip()
            try:
                conf = float(conf_raw)
            except (ValueError, TypeError):
                conf = 0.0

            if not text:
                logger.debug("ocr.text.empty", page_index=page_index, line_index=line_index)
                continue

            texts.append((text, conf))

    if not texts:
        logger.info("ocr.run.no_texts")
    else:
        logger.info("ocr.run.success", count=len(texts))

    return texts