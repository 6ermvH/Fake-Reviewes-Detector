import sys
from pathlib import Path


def init_logger(cfg) -> None:
    log_path = Path(cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f
