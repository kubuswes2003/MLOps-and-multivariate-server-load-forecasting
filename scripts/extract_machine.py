#!/usr/bin/env python3
"""Stream machine_usage.tar.gz and keep only the rows of one machine.

Works on macOS, Linux and Windows without unpacking the full archive to disk.

Example:
    python scripts/extract_machine.py --machine-id m_1932
"""

import argparse
import logging
import sys
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "machine_usage.tar.gz"
PROGRESS_EVERY_LINES = 20_000_000


def extract_machine(archive: Path, machine_id: str, output: Path) -> int:
    """Copy every line starting with ``machine_id,`` from the archive to ``output``.

    Args:
        archive: Path to machine_usage.tar.gz.
        machine_id: Machine to keep, e.g. "m_1932".
        output: Destination CSV.

    Returns:
        Number of rows written.
    """
    prefix = f"{machine_id},".encode()
    kept = 0
    with tarfile.open(archive, mode="r|gz") as tar, output.open("wb") as out:
        for member in tar:
            stream = tar.extractfile(member)
            if stream is None:
                continue
            logger.info("Scanning %s ...", member.name)
            for line_no, line in enumerate(stream, start=1):
                if line.startswith(prefix):
                    out.write(line)
                    kept += 1
                if line_no % PROGRESS_EVERY_LINES == 0:
                    logger.info("  %d M lines scanned, %d kept", line_no // 1_000_000, kept)
    return kept


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine-id", default="m_1932")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output", type=Path, default=None, help="default: data/<machine-id>.csv"
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.archive.exists():
        logger.error("Archive not found: %s (download it first, see README)", args.archive)
        return 2

    output = args.output or args.archive.parent / f"{args.machine_id}.csv"
    kept = extract_machine(args.archive, args.machine_id, output)
    if kept == 0:
        output.unlink(missing_ok=True)
        logger.error("No rows for %s — check the machine id.", args.machine_id)
        return 1
    logger.info("Wrote %d rows to %s", kept, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
