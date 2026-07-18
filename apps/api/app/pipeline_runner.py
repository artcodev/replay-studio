from __future__ import annotations

"""Entrypoint for the durable video/multi-pass runner service."""

import signal
from threading import Event

from .database import init_database
from .pipeline_recovery import start_dedicated_pipeline_monitor


def main() -> None:
    init_database()
    stopped = Event()

    def request_stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    monitor = start_dedicated_pipeline_monitor()
    try:
        stopped.wait()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
