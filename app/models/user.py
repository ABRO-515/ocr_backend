from sqlalchemy import Column, Integer, String, Date

from app.db.session import Base


class UserDetails(Base):
    __tablename__ = "user_details"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    father_name = Column(String(200), nullable=False)
    gender = Column(String(20), nullable=False)
    country_of_stay = Column(String(100), nullable=False)
    identity_number = Column(String(50), nullable=False, unique=True, index=True)
    date_of_birth = Column(Date, nullable=False)
    date_of_issue = Column(Date, nullable=False)
    date_of_expiry = Column(Date, nullable=False)
    cnic_image_path = Column(String(500), nullable=False)


