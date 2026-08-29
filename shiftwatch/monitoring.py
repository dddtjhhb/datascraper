from dataclasses import dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class Alarm:
    index: int
    method: str
    statistic: float


@dataclass(frozen=True)
class BatchMetric:
    batch_id: str
    error_rate: float

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id is required")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")


def load_batch_metrics(path: str | Path) -> list[BatchMetric]:
    """Load an ordered error-rate series from a CSV file."""
    metrics = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"batch_id", "error_rate"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain batch_id and error_rate columns")
        for line_number, row in enumerate(reader, 2):
            metric = BatchMetric(row["batch_id"].strip(), float(row["error_rate"]))
            if metric.batch_id in seen:
                raise ValueError(f"duplicate batch_id {metric.batch_id!r} on line {line_number}")
            seen.add(metric.batch_id)
            metrics.append(metric)
    if not metrics:
        raise ValueError("batch metric series is empty")
    return metrics


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


def monitor(
    metrics: list[BatchMetric],
    target: float,
    cusum_drift: float = 0.02,
    cusum_threshold: float = 0.3,
    ewma_alpha: float = 0.3,
    ewma_threshold: float = 0.15,
) -> list[dict]:
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be between 0 and 1")
    values = [metric.error_rate for metric in metrics]
    alarms = cusum(values, target, cusum_drift, cusum_threshold)
    alarms += ewma(values, target, ewma_alpha, ewma_threshold)
    return [
        {
            "batch_id": metrics[alarm.index].batch_id,
            "index": alarm.index,
            "method": alarm.method,
            "statistic": round(alarm.statistic, 6),
            "error_rate": metrics[alarm.index].error_rate,
        }
        for alarm in sorted(alarms, key=lambda alarm: (alarm.index, alarm.method))
    ]


def write_alarms(alarms: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["batch_id", "index", "method", "statistic", "error_rate"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(alarms)
