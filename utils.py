"""Shared utility functions for the waste segregation system."""

import yaml
from pathlib import Path

CLASS_MAP = {
    "paper": "paper_cardboard",
    "cardboard": "paper_cardboard",
    "plastic": "plastic",
    "glass": "miscellaneous",
    "metal": "miscellaneous",
    "trash": "miscellaneous",
}

TARGET_CLASSES = ["paper_cardboard", "plastic", "miscellaneous"]


def load_config(config_path="config.yaml"):
    """Load YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_project_root():
    """Return project root directory."""
    return Path(__file__).parent
