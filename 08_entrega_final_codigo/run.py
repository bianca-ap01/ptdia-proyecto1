"""Run one stage locally when IEEE-CIS competition CSVs are available."""

import argparse
from pathlib import Path

from stages import RUNNERS, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=RUNNERS)
    args = parser.parse_args()
    print(run(args.stage, Path(__file__).resolve().parent))
