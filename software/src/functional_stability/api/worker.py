"""RQ worker configuration entry point."""

from redis import Redis
from rq import Worker

from .config import get_settings
from .logging import configure_logging


def main() -> None:
    settings = get_settings()
    if not settings.redis_url:
        raise RuntimeError("NR_REDIS_URL must be set for the RQ worker")
    configure_logging(settings.log_level)
    Worker(["default"], connection=Redis.from_url(settings.redis_url)).work()


if __name__ == "__main__":
    main()
