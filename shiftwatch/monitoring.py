from dataclasses import dataclass


@dataclass(frozen=True)
class Alarm:
    index: int
    method: str
    statistic: float


def cusum(values: list[float], target: float, drift: float = 0.02, threshold: float = 0.3) -> list[Alarm]:
    statistic = 0.0
    alarms = []
    for index, value in enumerate(values):
        statistic = max(0.0, statistic + value - target - drift)
        if statistic >= threshold:
            alarms.append(Alarm(index, "cusum", statistic))
            statistic = 0.0
    return alarms


def ewma(values: list[float], target: float, alpha: float = 0.3, threshold: float = 0.15) -> list[Alarm]:
    statistic = target
    alarms = []
    for index, value in enumerate(values):
        statistic = alpha * value + (1 - alpha) * statistic
        if statistic - target >= threshold:
            alarms.append(Alarm(index, "ewma", statistic))
    return alarms
