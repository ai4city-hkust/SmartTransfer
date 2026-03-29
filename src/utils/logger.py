import os, sys, time


class TeeLogger:
    def __init__(self, log_path: str, mode: str = "a"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.f = open(log_path, mode, buffering=1, encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, s):
        try:
            self._stdout.write(s)
        except Exception:
            pass
        try:
            self.f.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

class TqdmToLogger:

    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        if not s:
            return
        s2 = s.replace("\r", "\n")
        self.stream.write(s2)

    def flush(self):
        self.stream.flush()

def setup_logging(save_dir: str, fold: int = None, mode: str = None) -> TeeLogger:
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = []
    if mode is not None:
        tag.append(str(mode))
    if fold is not None:
        tag.append(f"fold{fold}")
    tag = "_".join(tag) if tag else "run"
    log_path = os.path.join(save_dir, f"{tag}_{ts}.log")

    logger = TeeLogger(log_path, mode="a")
    sys.stdout = logger
    sys.stderr = logger

    print(f"[LOG] Writing stdout/stderr to: {log_path}")
    
    return logger

