from abc import ABC, abstractmethod
from src.condition import Condition


class DataController[Ctx, Data](ABC):
    @abstractmethod
    def pull(self, ctx: Ctx) -> Data:
        pass

    @abstractmethod
    def find_conditions(self, data: Data) -> list[Condition]:
        pass

    @abstractmethod
    def notify(self, ctx: Ctx, conditions: list[Condition]) -> bool:
        pass

    def pull_and_notify(self, ctx: Ctx) -> bool:
        data = self.pull(ctx)
        conditions = self.find_conditions(data, ctx)
        return self.notify(ctx=ctx, conditions=conditions)
