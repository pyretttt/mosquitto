from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db.models import Prediction


class PredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, prediction: Prediction) -> None:
        self._session.add(prediction)
        await self._session.commit()
