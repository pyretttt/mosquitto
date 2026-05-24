"""Pydantic v2 request/response schemas.

Separating these from ORM models is good practice:
- ORM model = how data lives in the DB.
- Pydantic model = the API contract (what clients see).
Changing one should not silently break the other.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    owner: str = Field(min_length=1, max_length=60)


class ModelVersionIn(BaseModel):
    version: str = Field(min_length=1, max_length=40)
    metric_name: str = Field(min_length=1, max_length=40)
    metric_value: float
    artifact_uri: str | None = None


class ModelVersionOut(ModelVersionIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    created_at: datetime


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    owner: str
    created_at: datetime
    versions: list[ModelVersionOut] = []
