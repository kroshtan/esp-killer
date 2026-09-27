from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from server.orgconfig import OrgConfig
from server.scoring.config import ScoringConfig
from server.scoring.features import associates, beeline_evidence
from server.scoring.game import load_profile
from server.scoring.trajectories import from_frame
from tests.helpers import killed, track
from tests.unit.test_scoring_features import CFG

BASE = """
name: base
defaults: {sight_m: 300, scent_m: 150, hearing_m: 800}
classes:
  Pteranodon: {sight_m: 900, flyer: true}
groups: {mixed_species: false, shared_awareness_s: 120}
"""


def write(directory: Path, name: str, text: str) -> None:
    (directory / f"{name}.yaml").write_text(text)


def test_the_repository_profile() -> None:
    profile = load_profile()
    assert profile.name == "evrima"
    assert profile.awareness_m("Pteranodon") == 900
    assert profile.senses("Pteranodon").flyer
    assert profile.awareness_m("Tyrannosaurus") == profile.awareness_m(None) == 300
    assert profile.awareness_m("NotARealDino") == 300
    assert not profile.groups.mixed_species
    assert not profile.calls.observed


def test_extends_overrides_only_what_differs(tmp_path: Path) -> None:
    write(tmp_path, "base", BASE)
    write(
        tmp_path,
        "modded",
        "name: modded\nextends: base\nclasses:\n  Pteranodon: {sight_m: 1500}\n  Troodon: {scent_m: 400}\n"
        "groups: {mixed_species: true}\n",
    )
    profile = load_profile("modded", tmp_path)
    assert profile.awareness_m("Pteranodon") == 1500
    assert profile.senses("Pteranodon").flyer  # inherited from the parent's class entry
    assert profile.awareness_m("Troodon") == 400
    assert profile.groups.mixed_species
    assert profile.groups.shared_awareness_s == 120  # inherited


@pytest.mark.parametrize(
    ("files", "error"),
    [
        ({"a": "name: a\nextends: b\n", "b": "name: b\nextends: a\n"}, "cycle"),
        ({"a": "name: a\nextends: missing\n"}, "not found"),
        ({"a": "name: something-else\ndefaults: {sight_m: 1, scent_m: 1, hearing_m: 1}\n"}, "must be"),
        ({"a": "name: a\ndefaults: {sight_m: 1, scent_m: 1, hearing_m: 1, sihgt_m: 2}\n"}, "Extra inputs"),
    ],
)
def test_broken_profiles_fail_loudly(tmp_path: Path, files: dict[str, str], error: str) -> None:
    for name, text in files.items():
        write(tmp_path, name, text)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        load_profile("a", tmp_path)


def test_awareness_follows_the_class_at_each_moment() -> None:
    frame = pd.concat(
        [
            track("p", [(0, 0, 0), (300, 0, 0)], dino_class="Pteranodon"),
            track("p", [(305, 0, 0), (600, 0, 0)], dino_class="Stegosaurus"),  # respawned as another class
        ]
    )
    tr = from_frame(frame, "srv", CFG, load_profile())
    assert tr.awareness[0, 0] == 900
    assert tr.awareness[-1, 0] == 300
    assert np.all(from_frame(frame, "srv", ScoringConfig(awareness_m=123)).awareness == 123)  # no profile


@pytest.mark.parametrize(("dino_class", "episodes"), [("Pteranodon", 0), ("Carnotaurus", 1)])
def test_a_flyer_heading_for_someone_it_could_see_is_no_evidence(dino_class: str, episodes: int) -> None:
    # Straight to a resting player 800 m away, then off again. From the air that player was in sight.
    resting = killed(track("resting", [(0, 800, 0), (1200, 800, 0)]), at=150)  # killed on arrival
    mover = track("mover", [(0, 0, 0), (140, 800, 0), (200, 800, 0), (400, 800, 1200), (1200, 800, 1200)], dino_class)
    tr = from_frame(pd.concat([mover, resting]), "srv", CFG, load_profile())
    evidence = beeline_evidence(tr, CFG, associates(tr, CFG))["mover"]
    assert evidence.episodes == episodes


def test_config_rejects_unknown_profiles() -> None:
    config = {"orgs": {"org": {"servers": {"s1": {"game_profile": "evrima"}, "s2": {}}}}}
    profiles = OrgConfig.model_validate(config).game_profiles("org")
    assert {name: p.name for name, p in profiles.items()} == {"s1": "evrima", "s2": "evrima"}
    config["orgs"]["org"]["servers"]["s1"]["game_profile"] = "no-such-profile"
    with pytest.raises(ValidationError, match="no-such-profile"):
        OrgConfig.model_validate(config)
