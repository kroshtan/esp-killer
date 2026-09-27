"""docs/configuration.md lists every setting that exists in the code, under its real environment variable."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from agent.config import AgentSettings
from server.alerts.settings import SmtpSettings
from server.scoring.config import ScoringConfig
from server.settings import ServerSettings

DOC = (Path(__file__).resolve().parents[2] / "docs" / "configuration.md").read_text(encoding="utf-8")


def env_names(model: type[BaseModel], prefix: str, delimiter: str) -> list[str]:
    names = []
    for name, info in model.model_fields.items():
        nested = info.annotation
        if isinstance(nested, type) and issubclass(nested, BaseModel):
            names += env_names(nested, f"{prefix}{name.upper()}{delimiter}", delimiter)
        else:
            names.append(f"{prefix}{name.upper()}")
    return names


@pytest.mark.parametrize("model", [AgentSettings, ServerSettings, SmtpSettings])
def test_every_environment_variable_is_documented(model: type[BaseModel]) -> None:
    prefix = str(model.model_config.get("env_prefix") or "")
    delimiter = str(model.model_config.get("env_nested_delimiter") or "")
    missing = [name for name in env_names(model, prefix, delimiter) if f"`{name}`" not in DOC]
    assert missing == []


def test_every_scoring_threshold_is_documented() -> None:
    missing = [name for name in ScoringConfig.model_fields if f"`{name}`" not in DOC]
    assert missing == []
