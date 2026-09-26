"""
SMTP settings for email alerts, from ``ESPK_SMTP_*`` environment variables.

These are process-level, like :mod:`server.settings`: one outgoing mail server for every org. Which address each
org's alerts go to is in config.yaml. Without a host and a from address, email alerts are off.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SmtpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESPK_SMTP_", extra="ignore")

    host: str | None = None
    # 587 is submission with STARTTLS. For implicit TLS set use_ssl (usually with port 465); starttls is then ignored.
    port: int = Field(default=587, ge=1, le=65535)
    username: str | None = None
    password: SecretStr | None = None
    from_address: str | None = None
    starttls: bool = True
    use_ssl: bool = False
    timeout_s: float = Field(default=30.0, gt=0)

    @property
    def is_configured(self) -> bool:
        """
        Whether email alerts can be sent at all.

        :return: True if a host and a from address are set
        """
        return bool(self.host) and bool(self.from_address)
