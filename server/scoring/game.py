"""
Game mechanics profiles (``game/<name>.yaml``): how far each class can notice other players.

The profiles live in the repository, in plain YAML, so players who know the game can read and correct them. A
profile may ``extends:`` another and override only what differs, which is how modded servers get their own. The
operator picks a profile per server in config.yaml (``game_profile``); ``evrima`` is the default.

Validation is strict (unknown keys are errors), so a typo in a suggested change fails loudly instead of being
silently ignored.
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DEFAULT_PROFILE = "evrima"
GAME_DIR = Path(__file__).resolve().parents[2] / "game"

Metres = Annotated[float, Field(ge=0)]
ProfileName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")]


class Senses(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sight_m: Metres
    scent_m: Metres
    hearing_m: Metres
    flyer: bool = False

    @property
    def awareness_m(self) -> float:
        """
        How far this class can plausibly notice another player: the larger of sight and scent.

        Hearing is not included: calls are momentary events, handled separately once they are observable.

        :return: metres
        """
        return max(self.sight_m, self.scent_m)


class SensesOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sight_m: Metres | None = None
    scent_m: Metres | None = None
    hearing_m: Metres | None = None
    flyer: bool | None = None


class Calls(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: bool = False
    relevance_s: Annotated[float, Field(gt=0)] = 60.0


class Groups(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: bool = False
    shared_awareness_s: Annotated[float, Field(gt=0)] = 120.0
    mixed_species: bool = False
    max_group_size: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)


class GameProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ProfileName
    description: str = ""
    extends: ProfileName | None = None
    defaults: Senses
    classes: dict[str, SensesOverride] = Field(default_factory=dict)
    calls: Calls = Field(default_factory=Calls)
    groups: Groups = Field(default_factory=Groups)

    def senses(self, dino_class: str | None) -> Senses:
        """
        A class's senses: its overrides on top of the defaults. Unknown or missing classes get the defaults.

        :param dino_class: class name as the server reports it (without ``BP_``/``_C``), or None
        :return: the senses
        """
        override = self.classes.get(dino_class) if dino_class else None
        if override is None:
            return self.defaults
        return self.defaults.model_copy(update=override.model_dump(exclude_none=True))

    def awareness_m(self, dino_class: str | None) -> float:
        """
        Shorthand for ``senses(dino_class).awareness_m``.

        :param dino_class: class name, or None
        :return: metres
        """
        return self.senses(dino_class).awareness_m


def load_profile(name: str = DEFAULT_PROFILE, directory: Path = GAME_DIR) -> GameProfile:
    """
    Load a profile, resolving ``extends`` (the child's values win, section by section and class by class).

    :param name: profile name (the file is ``<directory>/<name>.yaml``)
    :param directory: where profiles live
    :return: the resolved profile
    """
    return _load(name, directory.resolve())


@cache
def _load(name: str, directory: Path) -> GameProfile:
    data = _resolve(name, directory, seen=())
    return GameProfile.model_validate(data)


def _resolve(name: str, directory: Path, seen: tuple[str, ...]) -> dict[str, Any]:
    if name in seen:
        raise ValueError(f"game profile cycle: {' -> '.join((*seen, name))}")
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"game profile not found: {path}")
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("name") != name:
        raise ValueError(f"{path}: `name` must be {name!r}")
    parent_name = data.get("extends")
    if parent_name is None:
        return data
    return _merge(_resolve(parent_name, directory, (*seen, name)), data)


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings; the child wins. Lists and scalars are replaced, not merged."""
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(parent.get(key), dict):
            merged[key] = _merge(parent[key], value)
        else:
            merged[key] = value
    return merged
