"""
The one exception alert delivery raises, whatever the channel.

The alert outbox only needs to decide between "try again later" and "give up and tell the operator", so Discord
and email failures are both mapped onto :class:`DeliveryError` and its ``permanent`` flag.
"""


class DeliveryError(Exception):
    """
    An alert could not be delivered.

    ``permanent`` is True when retrying cannot help until someone changes the configuration: a deleted webhook,
    a rejected recipient, wrong SMTP credentials. Otherwise the failure is transient (network, rate limit, server
    error) and the alert should be retried later, no sooner than ``retry_after_s`` if that is set.
    """

    def __init__(self, message: str, *, permanent: bool, retry_after_s: float | None = None) -> None:
        """
        Create the error.

        :param message: what went wrong. Must not contain secrets (webhook URLs, passwords).
        :param permanent: True if retrying without a configuration change is pointless
        :param retry_after_s: for transient failures, the earliest sensible retry, if the other side said so
        """
        super().__init__(message)
        self.permanent = permanent
        self.retry_after_s = retry_after_s
