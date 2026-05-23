from pydantic import BaseModel, Field


class TopK(BaseModel):
    class_id: int
    label: str
    score: float


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    cache_hit: bool
    latency_ms: float = Field(description="End-to-end time excluding network in/out")
    top_class: TopK
    top_k: list[TopK]


class HealthResponse(BaseModel):
    api: bool
    triton: bool
    redis: bool
