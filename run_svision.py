from __future__ import annotations

import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()
    from app.main import main as serve

    serve()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        if getattr(sys, "frozen", False):
            input("SVision failed to start. Press Enter to close.")
        raise
