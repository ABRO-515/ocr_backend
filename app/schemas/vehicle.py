from pydantic import BaseModel, Field


class VehicleDetailsCreate(BaseModel):
    vehicle_number: str = Field(min_length=1, max_length=40)
    vehicle_image_path: str


class VehicleDetailsOut(BaseModel):
    id: int
    vehicle_number: str
    vehicle_image_path: str

    class Config:
        from_attributes = True


