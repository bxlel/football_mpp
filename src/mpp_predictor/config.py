"""Chargement et validation de la configuration du modèle.

La config vit dans ``config/params.yaml``. Ce module la charge une fois,
valide les invariants critiques (les poids somment à 1) et expose un objet
typé pratique à manipuler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "params.yaml"
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class IndexWeights:
    """Pondérations des 3 critères d'un Super-Index."""

    dynamics: float
    freshness: float
    context: float

    def __post_init__(self) -> None:
        total = self.dynamics + self.freshness + self.context
        if abs(total - 1.0) > _TOLERANCE:
            raise ValueError(
                f"Les poids d'un index doivent sommer à 1.0, obtenu {total:.4f}. "
                "Corrige config/params.yaml."
            )


@dataclass(frozen=True)
class Config:
    """Vue typée de l'ensemble de la configuration."""

    raw: dict[str, Any]

    @property
    def attack_weights(self) -> IndexWeights:
        w = self.raw["attack_index"]["weights"]
        return IndexWeights(w["dynamics"], w["freshness"], w["context"])

    @property
    def defense_weights(self) -> IndexWeights:
        w = self.raw["defense_index"]["weights"]
        return IndexWeights(w["dynamics"], w["freshness"], w["context"])

    def section(self, *keys: str) -> Any:
        """Accès défensif à une sous-section, ex: cfg.section('poisson', 'max_goals')."""
        node: Any = self.raw
        for key in keys:
            node = node[key]
        return node


def load_config(path: Path | None = None) -> Config:
    """Charge la config depuis le YAML (défaut : config/params.yaml)."""
    target = path or _CONFIG_PATH
    with open(target, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw)
    # Force la validation des poids dès le chargement.
    _ = cfg.attack_weights
    _ = cfg.defense_weights
    return cfg
