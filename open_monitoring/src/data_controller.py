from abc import ABC, abstractmethod
from src.condition import ConditionEvaluationResult
from src.notification import Notification

class DataController[Ctx, Data](ABC):
    @abstractmethod
    async def pull(self, ctx: Ctx) -> Data:
        pass


    @abstractmethod
    def evaluate_conditions(self, data: Data, ctx: Ctx) -> dict[str, ConditionEvaluationResult]:
        pass


    @abstractmethod
    def get_notifications(self, data: Data, ctx: Ctx, conditions: dict[str, ConditionEvaluationResult]) -> list[Notification]:
        pass


    async def pull_and_get_notifications(self, ctx: Ctx) -> list[Notification]:
        data = await self.pull(ctx)
        conditions = self.evaluate_conditions(data, ctx)
        return self.get_notifications(data=data, ctx=ctx, conditions=conditions)
