"""A lock file so two loops never run against one store."""

import fcntl
import os
from pathlib import Path
from types import TracebackType


class LoopAlreadyRunningError(RuntimeError):
    pass


class LoopLock:
    def __init__(self, store_path: Path) -> None:
        self.path = store_path.with_suffix(store_path.suffix + ".lock")
        self._fd: int | None = None

    def __enter__(self) -> "LoopLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(self._fd)
            self._fd = None
            raise LoopAlreadyRunningError(f"another loop holds {self.path}") from e
        os.ftruncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
