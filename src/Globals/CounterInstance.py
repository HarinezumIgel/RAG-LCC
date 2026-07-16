from Commons.SingletonMixin import SingletonMixin


class _CounterBase(SingletonMixin):
    """Base singleton counter – each subclass gets its own independent instance."""

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.value: int = 0

    def increment(self) -> None:
        self.value += 1

    def get(self) -> int:
        return self.value


class FailedCount(_CounterBase):
    pass


class HumanReviewCount(_CounterBase):
    pass


class ExclusionsCount(_CounterBase):
    pass


class ProcessedCount(_CounterBase):
    pass


class IgnoredCount(_CounterBase):
    pass
