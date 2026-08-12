"""Pytest root marker: puts the project root on sys.path for the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
