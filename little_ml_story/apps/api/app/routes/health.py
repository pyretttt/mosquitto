from fastapi import APIRouter, Request

from apps.api.app.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    cache = request.app.state.cache
    triton = request.app.state.triton
    return HealthResponse(
        api=True,
        triton=triton.ready(),
        redis=await cache.ping(),
    )


@router.get("/livez")
async def livez() -> dict[str, bool]:
    return {"ok": True}
