import sys

def init_logger(cfg) -> None:
    f = open(cfg["log_file"], 'w', encoding='utf-8')
    sys.stdout = f
    sys.stderr = f
