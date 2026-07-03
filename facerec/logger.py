import logging
import logging.handlers
import os

from facerec import config


def setup_logger(name: str = "face_recog") -> logging.Logger:
    """
    Configure the ROOT logger with a rotating file + console handler.

    Call ONCE at program entry (main.py / register_faces.py / ...). Every module
    uses logging.getLogger(__name__); those loggers propagate to the root, so
    attaching the handlers here is what actually lets their records reach the file.
    (Attaching to a "face_recog" logger instead — as before — silently dropped
    every module's INFO/DEBUG because "camera", "database", ... are not its children.)

    Idempotent: safe to call more than once; handlers are added only once.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, "face_recog.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    already_configured = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
    )
    if already_configured:
        return logging.getLogger(name)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)

    # Quiet chatty third-party libraries now that we own the root handlers
    # (they propagate here too). TF C++ logs are already gated by TF_CPP_MIN_LOG_LEVEL.
    for noisy in ("tensorflow", "absl", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    return logging.getLogger(name)
