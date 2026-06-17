"""Modèles de domaine : les objets métier manipulés par tout le pipeline.

On garde ces structures volontairement minces. Elles représentent l'état
*consolidé* d'une équipe à l'instant d'un match donné — c'est l'entrée de la
couche `features` qui produit les Super-Index.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlayedMatch:
    """Un match passé d'une sélection, vu du côté de l'équipe analysée."""

    goals_for: int
    goals_against: int
    opponent_elo: float
    days_ago: int  # ancienneté en jours par rapport au match à prédire


@dataclass(frozen=True)
class KeyPlayer:
    """Un cadre de la sélection et sa charge récente en club."""

    name: str
    is_offensive: bool
    club_matches_last_year: int  # volume sur une année glissante


@dataclass(frozen=True)
class MatchContext:
    """Conditions entourant le match à prédire."""

    rest_days: int                 # jours de repos de l'équipe
    opponent_rest_days: int        # jours de repos de l'adversaire
    temperature_celsius: float | None = None
    altitude_meters: float | None = None
    is_dead_rubber: bool = False   # match sans enjeu (déjà qualifié/éliminé)
    mutual_draw_qualifies: bool = False  # le nul arrange les deux équipes


@dataclass(frozen=True)
class TeamSnapshot:
    """État complet d'une équipe à l'instant d'un match.

    C'est l'unité d'entrée des calculateurs d'index.
    """

    name: str
    elo: float
    recent_matches: list[PlayedMatch] = field(default_factory=list)
    key_players: list[KeyPlayer] = field(default_factory=list)
    context: MatchContext | None = None

    @property
    def offensive_players(self) -> list[KeyPlayer]:
        return [p for p in self.key_players if p.is_offensive]

    @property
    def defensive_players(self) -> list[KeyPlayer]:
        return [p for p in self.key_players if not p.is_offensive]
