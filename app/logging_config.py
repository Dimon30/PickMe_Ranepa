import logging
import logging.config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_VERBOSE_LOGGERS = ("app", "bot", "uvicorn", "uvicorn.access", "uvicorn.error")

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": _LOG_FORMAT},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
    "loggers": {
        name: {"level": "INFO", "handlers": ["console"], "propagate": False}
        for name in _VERBOSE_LOGGERS
    },
}


def setup_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
