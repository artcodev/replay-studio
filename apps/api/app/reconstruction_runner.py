from __future__ import annotations

"""Dedicated durable reconstruction worker process."""

import signal
from threading import Event

from .database import init_database
from .reconstruction_recovery import start_dedicated_reconstruction_recovery


def main() -> None:
    init_database()
    stopped = Event()

    def request_stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # The dedicated runner owns killable child processes. Cancellation can
    # therefore release physical compute even while native inference is blocked.
    monitor = start_dedicated_reconstruction_recovery()
    try:
        stopped.wait()
    finally:
        monitor.stop(timeout=5.0)


if __name__ == "__main__":
    main()
