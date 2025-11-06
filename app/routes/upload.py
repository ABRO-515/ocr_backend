from datetime import datetime
import re
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.user import UserDetails
from app.models.vehicle import VehicleDetails
from app.schemas.user import UserDetailsOut
from app.schemas.vehicle import VehicleDetailsOut
from app.services.image_processing import (
    load_image_from_bytes,
    preprocess_for_ocr,
    save_image,
)
from app.services.ocr import ocr_text


router = APIRouter()
logger = get_logger()


def _clean_ocr_text(text: str) -> str:
    """Lightweight cleaning for OCR lines to stabilize downstream parsing.
    - Trim, collapse whitespace
    - Normalize common lookalikes and punctuation
    - Keep letters, digits, and common separators
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    # Normalize unicode quotes/dashes and separators
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash → hyphen
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # curly quotes → straight
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("|", "I")  # pipe → I
    s = s.replace("$", "S")  # dollar → S
    s = s.replace(",", ".")  # commas often used as dot
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Keep only reasonable characters for CNIC parsing
    s = re.sub(r"[^A-Za-z0-9 .\-/:]", "", s)
    return s.strip()


def _store_upload(file: UploadFile, subdir: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    ext = Path(file.filename or "upload").suffix or ".jpg"
    dest = Path(settings.upload_dir) / subdir / f"{ts}{ext}"
    logger.info("upload.store.start", filename=file.filename, subdir=subdir, dest=str(dest))
    data = file.file.read()
    logger.debug("upload.store.read", bytes=len(data))
    img = load_image_from_bytes(data)
    logger.info("upload.store.decoded", shape=list(img.shape))
    save_image(img, dest)
    logger.info("upload.store.success", path=str(dest))
    return dest


def _parse_cnic_fields(texts: Dict[int, str]) -> Dict[str, str]:
    # Clean and preprocess all lines
    cleaned_lines = {i: _clean_ocr_text(line) for i, line in texts.items()}
    full_text = " ".join(cleaned_lines.values()).upper()
    lines = [line.upper() for line in cleaned_lines.values()]

    result = {
        "name": "Unknown",
        "father_name": "Unknown",
        "gender": "Unknown",
        "country_of_stay": "Pakistan",
        "identity_number": "",
        "date_of_birth": "1970-01-01",
        "date_of_issue": "1970-01-01",
        "date_of_expiry": "1970-01-01",
    }

    # === 1. IDENTITY NUMBER (MOST RELIABLE) ===
    # Pattern: 38403-9346396-1  (5-7-1 format)
    cnic_patterns = [
        r'\b(\d{5}[- ]?\d{7}[- ]?\d)\b',           # Standard with hyphens/spaces
        r'\b(\d{5}\.\d{7}\.\d)\b',                 # Dots
        r'\b(\d{13})\b',                           # Flat 13 digits
        r'\b(\d{5}[- ]?\d{3}[- ]?\d{3}[- ]?\d{4}[- ]?\d)\b',  # Broken
    ]
    for pattern in cnic_patterns:
        match = re.search(pattern, full_text)
        if match:
            cnic = match.group(1).replace(" ", "").replace(".", "-")
            if len(cnic) == 15 and cnic.count("-") == 2:
                result["identity_number"] = cnic
                break
            elif len(cnic) == 13:
                result["identity_number"] = f"{cnic[:5]}-{cnic[5:12]}-{cnic[12]}"
                break

    # === 2. DATES - Very flexible matching ===
    date_pattern = r'\b(\d{2}[-/.]\d{2}[-/.]\d{2,4})\b'
    raw_dates = re.findall(date_pattern, full_text)

    # Normalize and filter valid dates (1980–2030)
    valid_dates = []
    for d in raw_dates:
        d = d.replace("/", ".").replace("-", ".")
        parts = d.split(".")
        if len(parts) != 3:
            continue
        day, month, year = parts
        day = day.zfill(2)
        month = month.zfill(2)
        year = year.zfill(4)
        if year.startswith("19") or year.startswith("20"):
            normalized = f"{day}.{month}.{year}"
            if normalized not in valid_dates:
                valid_dates.append(normalized)

    # Assign in order: DOB → DOI → DOE (based on typical card order and values)
    dob_candidates = [d for d in valid_dates if "70" <= d[6:] <= "05"]  # 1970–2005
    doi_candidates = [d for d in valid_dates if d.startswith(("12.06.", "13.06.", "11.06."))]
    doe_candidates = [d for d in valid_dates if "202" in d or "203" in d]

    if dob_candidates:
        result["date_of_birth"] = dob_candidates[0].replace(".", "-")
    elif valid_dates:
        result["date_of_birth"] = valid_dates[0].replace(".", "-")

    if doi_candidates:
        result["date_of_issue"] = doi_candidates[0].replace(".", "-")
    elif len(valid_dates) > 1:
        result["date_of_issue"] = valid_dates[1].replace(".", "-")

    if doe_candidates:
        result["date_of_expiry"] = doe_candidates[0].replace(".", "-")
    elif len(valid_dates) > 2:
        result["date_of_expiry"] = valid_dates[2].replace(".", "-")

    # === 3. NAME & FATHER NAME (Urdu + English handling) ===
    name_keywords = ["NAME"]
    father_keywords = ["FATHER", "S/O", "D/O", "SON", "DAUGHTER"]
    stopwords = {
        "OF", "THE", "AND", "IS", "CARD", "IDENTITY", "NATIONAL", "ISLAMIC", "REPUBLIC",
        "PAKISTAN", "COUNTRY", "GENDER", "DATE", "BIRTH", "ISSUE", "EXPIRY", "HOLDER",
        "SIGNATURE", "NAME", "FATHER"
    }

    name_found = False
    father_found = False

    for line in lines:
        line_clean = re.sub(r'[^A-Z\s]', ' ', line)

        # Look for Name
        if not name_found and any(k in line_clean for k in name_keywords):
            words = line_clean.split()
            # Take words after keyword on same line if present
            name_parts = []
            started = False
            for w in words:
                if any(k in w for k in name_keywords):
                    started = True
                    continue
                if started and w.isalpha() and len(w) >= 2 and w not in stopwords:
                    name_parts.append(w)
                elif started and len(name_parts) > 0:
                    break
            if name_parts:
                result["name"] = " ".join(name_parts)
                name_found = True

        # Look for Father Name
        if not father_found and any(k in line_clean for k in father_keywords):
            words = line_clean.split()
            father_parts = []
            started = False
            for w in words:
                if any(k in w for k in father_keywords):
                    started = True
                    continue
                if started and w.isalpha() and len(w) >= 3 and w not in stopwords:
                    father_parts.append(w)
                elif started and len(father_parts) > 0:
                    break
            if father_parts:
                result["father_name"] = " ".join(father_parts)
                father_found = True

    # Fallback: if not found, look at the next lines following the keyword line
    if not name_found:
        for idx, line in enumerate(lines):
            if "NAME" in re.sub(r'[^A-Z\s]', ' ', line):
                # find next non-empty line(s)
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    nxt = re.sub(r'[^A-Z\s]', ' ', lines[j]).strip()
                    if not nxt:
                        continue
                    tokens = [w for w in nxt.split() if w.isalpha() and len(w) >= 2 and w not in stopwords]
                    if len(tokens) >= 1:
                        result["name"] = " ".join(tokens[:4])
                        name_found = True
                        break
            if name_found:
                break

    if not father_found:
        for idx, line in enumerate(lines):
            line_u = re.sub(r'[^A-Z\s]', ' ', line)
            if any(k in line_u for k in father_keywords):
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    nxt = re.sub(r'[^A-Z\s]', ' ', lines[j]).strip()
                    if not nxt:
                        continue
                    tokens = [w for w in nxt.split() if w.isalpha() and len(w) >= 3 and w not in stopwords]
                    if len(tokens) >= 1:
                        result["father_name"] = " ".join(tokens[:4])
                        father_found = True
                        break
            if father_found:
                break

    # === 4. GENDER ===
    if "MALE" in full_text or "M " in full_text:
        result["gender"] = "Male"
    elif "FEMALE" in full_text or "F " in full_text:
        result["gender"] = "Female"

    # === 5. COUNTRY (almost always Pakistan) ===
    if "PAKISTAN" in full_text or "PAK" in full_text:
        result["country_of_stay"] = "Pakistan"

    # === FINAL FALLBACK: Extract from known positions if still missing ===
    if result["identity_number"] == "":
        # Try to reconstruct from partial matches
        digits = re.sub(r'\D', '', full_text)
        if len(digits) >= 13:
            d13 = digits[:13]
            result["identity_number"] = f"{d13[:5]}-{d13[5:12]}-{d13[12]}"

    return result


def _best_plate_candidate(pairs) -> Optional[str]:
    """
    Extract Sindh vehicle registration plate in format: XXX-YYY
    Where:
      - XXX = 3 uppercase letters (e.g., AFR)
      - YYY = 3 digits (e.g., 123)
    Province name 'SINDH' appears below (used as a hint, not required in output).
    
    Returns cleaned plate as 'AFR123' (hyphen removed for DB storage).
    """
    # Pattern: 3 letters, optional hyphen/dot/space, 3 digits
    plate_pattern = re.compile(r"""
        ^\s*                  # optional leading space
        ([A-Z]{3})            # exactly 3 letters
        \s*[-.\s]?\s*         # optional separator: hyphen, dot, or space
        (\d{3})               # exactly 3 digits
        \s*$                  # optional trailing space
    """, re.IGNORECASE | re.VERBOSE)

    best_match = None
    best_conf = 0.0

    for text, conf in pairs:
        # Normalize EasyOCR (0–1) to percentage if needed
        conf_pct = (conf * 100.0) if (isinstance(conf, (int, float)) and conf <= 1.0) else conf
        # Confidence threshold: skip very low quality OCR
        if conf_pct < 40:
            continue

        match = plate_pattern.search(text)
        if not match:
            continue

        letters, digits = match.groups()
        candidate = f"{letters.upper()}{digits}"

        # Optional: boost confidence if "SINDH" appears anywhere in OCR results
        full_text = " ".join(t for t, _ in pairs).upper()
        if "SINDH" in full_text:
            conf_boost = conf_pct + 10  # small boost for context confirmation
        else:
            conf_boost = conf_pct

        if conf_boost > best_conf:
            best_conf = conf_boost
            best_match = candidate

    return best_match


@router.post("/upload-cnic", response_model=UserDetailsOut)
async def upload_cnic(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        logger.info("cnic.request.start", filename=file.filename)
        path = _store_upload(file, "cnic")
        logger.info("cnic.image.stored", path=str(path))
        original = load_image_from_bytes(path.read_bytes())
        logger.info("cnic.image.loaded", shape=list(original.shape))
        pre = preprocess_for_ocr(original)
        logger.info("cnic.preprocessed", shape=list(pre.shape))
        pre_path = path.with_name(f"{path.stem}_pre{path.suffix}")
        save_image(pre, pre_path)
        logger.info("cnic.preprocessed.saved", path=str(pre_path))
        # pairs = ocr_text(pre)
        pairs = ocr_text(original)
        logger.info("cnic.ocr.done", pair_count=len(pairs))
        lines = {i: t for i, (t, _c) in enumerate(pairs)}
        fields = _parse_cnic_fields(lines)

        # Parse dates to ISO YYYY-MM-DD
        def _to_iso(s: str) -> str:
            s = s.replace("/", "-").replace(".", "-")
            try:
                return datetime.strptime(s, "%d-%m-%Y").date().isoformat()
            except Exception:
                return "1970-01-01"

        user = UserDetails(
            name=fields["name"],
            father_name=fields["father_name"],
            gender=fields["gender"],
            country_of_stay=fields["country_of_stay"],
            identity_number=fields["identity_number"],
            date_of_birth=_to_iso(fields["date_of_birth"]),
            date_of_issue=_to_iso(fields["date_of_issue"]),
            date_of_expiry=_to_iso(fields["date_of_expiry"]),
            cnic_image_path=str(path),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("cnic.processed", id=user.id, identity=user.identity_number)
        return user
    except Exception as e:
        logger.error("cnic.error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to process CNIC: {e}")


@router.post("/upload-vehicle", response_model=VehicleDetailsOut)
async def upload_vehicle(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        logger.info("vehicle.request.start", filename=file.filename)
        path = _store_upload(file, "vehicle")
        logger.info("vehicle.image.stored", path=str(path))
        original = load_image_from_bytes(path.read_bytes())
        logger.info("vehicle.image.loaded", shape=list(original.shape))
        pre = preprocess_for_ocr(original)
        logger.info("vehicle.preprocessed", shape=list(pre.shape))
        pre_path = path.with_name(f"{path.stem}_pre{path.suffix}")
        save_image(pre, pre_path)
        logger.info("vehicle.preprocessed.saved", path=str(pre_path))
        # pairs = ocr_text(pre)
        pairs = ocr_text(original)
        logger.info("vehicle.ocr.done", pair_count=len(pairs))
        candidate = _best_plate_candidate(pairs)
        if not candidate:
            raise ValueError("Failed to detect a plausible vehicle number")

        vehicle = VehicleDetails(vehicle_number=candidate, vehicle_image_path=str(path))
        db.add(vehicle)
        db.commit()
        db.refresh(vehicle)
        logger.info("vehicle.processed", id=vehicle.id, number=vehicle.vehicle_number)
        return vehicle
    except Exception as e:
        logger.error("vehicle.error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to process vehicle: {e}")


