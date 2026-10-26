from __future__ import annotations
import json
from pathlib import Path
statuses = {}
image = Path("tangaroa_streamribbon.png")
state = Path("tangaroa_streamribbon.pvsm")
statuses["rendered_image"] = "PASSED" if image.is_file() and image.stat().st_size > 10000 else "FAILED"
statuses["reloadable_state"] = "PASSED" if state.is_file() and state.stat().st_size > 1000 else "FAILED"
if state.is_file():
    text = state.read_text(encoding="utf-8", errors="ignore").lower()
    statuses["streamline_to_ribbon_pipeline"] = "PASSED" if "streamtracer" in text and "ribbon" in text else "FAILED"
    statuses["seed_and_width_parameters"] = "PASSED" if all(x in text for x in ("29.9", "0.3")) else "FAILED"
else:
    statuses["streamline_to_ribbon_pipeline"] = statuses["seed_and_width_parameters"] = "FAILED"
print(json.dumps(statuses, sort_keys=True))
