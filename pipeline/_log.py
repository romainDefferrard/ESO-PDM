"""
_log.py — shared logger for the pipeline.
Matches the style of logger.py used in LiMatch.
"""
import logging

log = logging.getLogger("pipeline")


def setup(level: int = logging.INFO) -> None:
    if log.handlers:
        return
    log.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(ch)


# Indentation levels — mirrors log_stage / log_sub / log_sub_sub from logger.py
def info(msg: str)    -> None: log.info(msg)
def sub(msg: str)     -> None: log.info("  └─ %s", msg)
def subsub(msg: str)  -> None: log.info("      └─ %s", msg)
def warn(msg: str)    -> None: log.warning(msg)
def error(msg: str)   -> None: log.error(msg)
