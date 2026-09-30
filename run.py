
import sys

from cleavage_rebuild.cli import main


if __name__ == "__main__":


    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:

        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
