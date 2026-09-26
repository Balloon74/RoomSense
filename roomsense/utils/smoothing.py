"""Small reusable smoothers for noisy frame-by-frame measurements."""

from __future__ import annotations

from typing import Iterable


class ExponentialSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, measurement: float) -> float:
        if self._value is None:
            self._value = measurement
        else:
            self._value += self.alpha * (measurement - self._value)
        return self._value

    def reset(self) -> None:
        self._value = None


class VectorSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self._components: list[ExponentialSmoother] = []
        self.alpha = alpha

    def update(self, measurement: Iterable[float]) -> tuple[float, ...]:
        values = tuple(measurement)
        if not self._components:
            self._components = [ExponentialSmoother(self.alpha) for _ in values]
        if len(values) != len(self._components):
            raise ValueError("measurement dimension changed")
        return tuple(smoother.update(value) for smoother, value in zip(self._components, values))

    def reset(self) -> None:
        self._components.clear()
