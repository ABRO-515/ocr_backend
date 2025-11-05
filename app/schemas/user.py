from datetime import date
from pydantic import BaseModel, Field


class UserDetailsCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    father_name: str = Field(min_length=1, max_length=200)
    gender: str = Field(min_length=1, max_length=20)
    country_of_stay: str = Field(min_length=1, max_length=100)
    identity_number: str = Field(min_length=3, max_length=50)
    date_of_birth: date
    date_of_issue: date
    date_of_expiry: date
    cnic_image_path: str


class UserDetailsOut(BaseModel):
    id: int
    name: str
    father_name: str
    gender: str
    country_of_stay: str
    identity_number: str
    date_of_birth: date
    date_of_issue: date
    date_of_expiry: date
    cnic_image_path: str

    class Config:
        from_attributes = True


