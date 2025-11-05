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


def _store_upload(file: UploadFile, subdir: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    ext = Path(file.filename or "upload").suffix or ".jpg"
    dest = Path(settings.upload_dir) / subdir / f"{ts}{ext}"
    data = file.file.read()
    img = load_image_from_bytes(data)
    save_image(img, dest)
    return dest


def _parse_cnic_fields(texts: Dict[int, str]) -> Dict[str, str]:
    # Simple heuristics; real-world parsing would be locale-specific and more robust
    joined = " ".join(texts.values())

    identity_number = None
    id_match = re.search(r"(\d{5}-\d{7}-\d)", joined)
    if id_match:
        identity_number = id_match.group(1)

    dob = None
    doi = None
    doe = None
    for token in re.findall(r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b", joined):
        if dob is None:
            dob = token
        elif doi is None:
            doi = token
        elif doe is None:
            doe = token

    name = None
    father = None
    gender = None
    country = None

    # Heuristic mapping by keywords
    for _, line in texts.items():
        low = line.lower()
        if "name" in low and name is None:
            name = line.split(":")[-1].strip()
        if ("father" in low or "s/o" in low or "d/o" in low) and father is None:
            father = line.split(":")[-1].strip()
        if "male" in low or "female" in low:
            gender = "Male" if "male" in low else "Female"
        if "pakistan" in low:
            country = "Pakistan"

    return {
        "name": name or "Unknown",
        "father_name": father or "Unknown",
        "gender": gender or "Unknown",
        "country_of_stay": country or "Unknown",
        "identity_number": identity_number or "",
        "date_of_birth": dob or "1970-01-01",
        "date_of_issue": doi or "1970-01-01",
        "date_of_expiry": doe or "1970-01-01",
    }


def _best_plate_candidate(pairs) -> Optional[str]:
    # Pick the string with mix of letters/numbers and length 5-10
    best = None
    for text, conf in pairs:
        t = re.sub(r"[^A-Za-z0-9]", "", text)
        if 5 <= len(t) <= 10 and re.search(r"[A-Za-z]", t) and re.search(r"\d", t):
            if best is None or conf > best[1]:
                best = (t.upper(), conf)
    return best[0] if best else None


@router.post("/upload-cnic", response_model=UserDetailsOut)
async def upload_cnic(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        path = _store_upload(file, "cnic")
        original = load_image_from_bytes(path.read_bytes())
        pre = preprocess_for_ocr(original)
        pairs = ocr_text(pre)
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
        path = _store_upload(file, "vehicle")
        original = load_image_from_bytes(path.read_bytes())
        pre = preprocess_for_ocr(original, target_size=(800, 300))
        pairs = ocr_text(pre)
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


