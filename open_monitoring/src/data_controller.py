from abc import ABC, abstractmethod
from src.condition import ConditionEvaluationResult


class DataController[Ctx, Data](ABC):
    @abstractmethod
    def pull(self, ctx: Ctx) -> Data:
        pass


    @abstractmethod
    def evaluate_conditions(self, data: Data, ctx: Ctx) -> dict[str, ConditionEvaluationResult]:
        pass


    @abstractmethod
    def notify(self, data: Data, ctx: Ctx, conditions: dict[str, ConditionEvaluationResult]) -> bool:
        pass


    def pull_and_notify(self, ctx: Ctx) -> bool:
        data = self.pull(ctx)
        conditions = self.evaluate_conditions(data, ctx)
        return self.notify(ctx=ctx, conditions=conditions)
