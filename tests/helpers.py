"""Small helpers shared by the tests: paths and file loaders for the support data."""

import csv
import json
from pathlib import Path

# The support-data folder sits at the repository root, one level above tests/.
DATA_DIR = Path(__file__).resolve().parent.parent / "support-data"


def load_csv(relative_path: str) -> list[dict[str, str]]:
    """Read a CSV file from support-data and return its rows as dictionaries."""
    with open(DATA_DIR / relative_path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_json(relative_path: str) -> dict:
    """Read a JSON file from support-data."""
    with open(DATA_DIR / relative_path, encoding="utf-8") as file:
        return json.load(file)
