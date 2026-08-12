"""Typed, file-based configuration for the forecasting pipeline.

All paths in the YAML file are interpreted relative to the project root
(the directory containing ``config.yaml``), so the pipeline is portable —
no hardcoded absolute paths anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class SarimaConfig:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 5)


@dataclass
class GarchConfig:
    vol: str = "GARCH"  # GARCH | ARCH | EGARCH — passed straight to arch_model
    p: int = 1
    q: int = 1
    dist: str = "normal"


@dataclass
class PipelineConfig:
    symbol: str = "TCS"
    horizon: int = 20
    resample_rule: str = "D"
    train_window: int | None = 750
    confidence_level: float = 0.95
    data_dir: Path = PROJECT_ROOT / "data" / "stocks_data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    sarima: SarimaConfig = field(default_factory=SarimaConfig)
    garch: GarchConfig = field(default_factory=GarchConfig)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH, **overrides) -> PipelineConfig:
        """Load config from YAML; keyword overrides (e.g. from CLI) win."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        root = path.resolve().parent

        raw.update({k: v for k, v in overrides.items() if v is not None})

        sarima = SarimaConfig(
            order=tuple(raw.get("sarima", {}).get("order", (1, 1, 1))),
            seasonal_order=tuple(raw.get("sarima", {}).get("seasonal_order", (1, 1, 1, 5))),
        )
        garch_raw = raw.get("garch", {})
        garch = GarchConfig(
            vol=str(garch_raw.get("vol", "GARCH")),
            p=int(garch_raw.get("p", 1)),
            q=int(garch_raw.get("q", 1)),
            dist=str(garch_raw.get("dist", "normal")),
        )

        def _resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else root / p

        train_window = raw.get("train_window")
        return cls(
            symbol=str(raw.get("symbol", "TCS")),
            horizon=int(raw.get("horizon", 20)),
            resample_rule=str(raw.get("resample_rule", "D")),
            train_window=None if train_window is None else int(train_window),
            confidence_level=float(raw.get("confidence_level", 0.95)),
            data_dir=_resolve(raw.get("data_dir", "data/stocks_data")),
            output_dir=_resolve(raw.get("output_dir", "outputs")),
            sarima=sarima,
            garch=garch,
        )
