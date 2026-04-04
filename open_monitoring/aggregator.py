from abc import ABC, abstractmethod
from condition import Condition


class Aggregator[Ctx, Data, Config](ABC):
    @abstractmethod
    def pull(self, ctx: Ctx) -> Data:
        pass

    @abstractmethod
    def find_conditions(self, data: Data, cfg: Config) -> list[Condition]:
        pass

    @abstractmethod
    def notify(self, cfg: Config, conditions: list[Condition]) -> bool:
        pass

    def pull_and_notify(self, ctx: Ctx, cfg: Config) -> bool:
        data = self.pull(ctx)
        conditions = self.find_conditions(data, cfg)
        return self.notify(cfg=cfg, conditions=conditions)
