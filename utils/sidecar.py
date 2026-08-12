"""
JSON sidecar written next to every denoised output.

Machine-readable run summary so that callers (in particular the planned
PixInsight script) can read the result without parsing stdout. The sidecar
lives at "<output>.json", e.g. "image_denoised_v3.fit.json".

Author: Ben & Claude
Date: 2026-07-31
"""

import json
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np

SIDECAR_SCHEMA_VERSION = 1


def _to_jsonable(obj):
    """Convert numpy scalars and other non-JSON types to plain Python."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_run_sidecar(output_path: Union[str, Path], payload: Dict) -> Path:
    """Write the run summary next to the output image.

    Args:
        output_path: Path of the saved image; the sidecar is written at
            the same path with ".json" appended to the full file name.
        payload: Run summary. Numpy scalars are converted automatically.

    Returns:
        Path of the written sidecar.
    """
    output_path = Path(output_path)
    sidecar_path = output_path.parent / (output_path.name + '.json')

    payload = dict(payload)
    payload['schema_version'] = SIDECAR_SCHEMA_VERSION
    payload['written_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')

    with open(sidecar_path, 'w') as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)

    print(f"📄 Run sidecar: {sidecar_path}")
    return sidecar_path
