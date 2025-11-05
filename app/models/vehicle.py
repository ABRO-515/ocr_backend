from sqlalchemy import Column, Integer, String

from app.db.session import Base


class VehicleDetails(Base):
    __tablename__ = "vehicle_details"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_number = Column(String(40), nullable=False, unique=True, index=True)
    vehicle_image_path = Column(String(500), nullable=False)


