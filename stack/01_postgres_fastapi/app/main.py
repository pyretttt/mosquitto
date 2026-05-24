"""FastAPI app: a minimal model registry.

Endpoints:
  POST /experiments                  create an experiment
  GET  /experiments/{id}             fetch experiment WITH its versions (no N+1)
  POST /experiments/{id}/versions    register a new model version (unique per experiment)
"""

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .db import get_session
from .models import Experiment, ModelVersion
from .schemas import (
    ExperimentIn,
    ExperimentOut,
    ModelVersionIn,
    ModelVersionOut,
)

app = FastAPI(title="MLOps Model Registry")


@app.post("/experiments", response_model=ExperimentOut, status_code=201)
async def create_experiment(
    payload: ExperimentIn,
    session: AsyncSession = Depends(get_session),
) -> Experiment:
    exp = Experiment(**payload.model_dump())
    session.add(exp)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "experiment name already exists")
    await session.refresh(exp)
    # No versions yet, so empty list is fine.
    return exp


@app.get("/experiments/{exp_id}", response_model=ExperimentOut)
async def get_experiment(
    exp_id: int,
    session: AsyncSession = Depends(get_session),
) -> Experiment:
    # selectinload => "fetch versions in a SECOND query using IN(...)".
    # Without it, accessing exp.versions would trigger a lazy load PER row
    # if you ever loop over many experiments — the N+1 problem.
    stmt = (
        select(Experiment)
        .where(Experiment.id == exp_id)
        .options(selectinload(Experiment.versions))
    )
    exp = (await session.execute(stmt)).scalar_one_or_none()
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return exp


@app.post(
    "/experiments/{exp_id}/versions",
    response_model=ModelVersionOut,
    status_code=201,
)
async def register_version(
    exp_id: int,
    payload: ModelVersionIn,
    session: AsyncSession = Depends(get_session),
) -> ModelVersion:
    exists = await session.get(Experiment, exp_id)
    if exists is None:
        raise HTTPException(404, "experiment not found")

    mv = ModelVersion(experiment_id=exp_id, **payload.model_dump())
    session.add(mv)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        # The unique (experiment_id, version) constraint was hit.
        raise HTTPException(
            409, f"version {payload.version!r} already exists for this experiment"
        )
    await session.refresh(mv)
    return mv
