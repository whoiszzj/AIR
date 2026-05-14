import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# Convert scalar tensors and arrays into Python floats for JSON serialization.
def _to_float(value: Any) -> Optional[float]:
    """Convert a scalar-like object into float when possible."""
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.reshape(-1)[0])
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)
    except Exception:
        return None


# Round float-like values to a fixed number of decimals for JSON output.
def _round_float(value: Any, digits: int = 3) -> Optional[float]:
    """Round a scalar-like value to a fixed decimal precision."""
    scalar = _to_float(value)
    if scalar is None:
        return None
    return round(float(scalar), digits)


# Round float-like values to a fixed number of significant digits.
def _round_significant(value: Any, digits: int = 8) -> Optional[float]:
    """Round a scalar-like value to a fixed number of significant digits."""
    scalar = _to_float(value)
    if scalar is None:
        return None
    return float(f"{float(scalar):.{digits}g}")


# Round compact stage summaries with significant-digit formatting.
def _round_stage_info(stage_info: Dict[str, Any], digits: int = 5) -> Dict[str, Any]:
    """Round per-stage metric arrays in a stage summary to significant digits."""
    rounded_stage_info = dict(stage_info)
    if "num" in rounded_stage_info:
        rounded_stage_info["num"] = int(rounded_stage_info["num"])
    if isinstance(rounded_stage_info.get("psnr"), list):
        rounded_stage_info["psnr"] = [
            _round_significant(value, digits) for value in rounded_stage_info["psnr"]
        ]
    if isinstance(rounded_stage_info.get("ms-ssim"), list):
        rounded_stage_info["ms-ssim"] = [
            _round_significant(value, digits) for value in rounded_stage_info["ms-ssim"]
        ]
    if isinstance(rounded_stage_info.get("lpips"), list):
        rounded_stage_info["lpips"] = [
            _round_significant(value, digits) for value in rounded_stage_info["lpips"]
        ]
    if isinstance(rounded_stage_info.get("gaussian_num"), list):
        rounded_stage_info["gaussian_num"] = [
            _round_significant(value, digits) for value in rounded_stage_info["gaussian_num"]
        ]
    if isinstance(rounded_stage_info.get("time_cost_ms"), list):
        rounded_stage_info["time_cost_ms"] = [
            _round_significant(value, digits) for value in rounded_stage_info["time_cost_ms"]
        ]
    if isinstance(rounded_stage_info.get("mean_mse_error"), list):
        rounded_stage_info["mean_mse_error"] = [
            _round_significant(value, digits) for value in rounded_stage_info["mean_mse_error"]
        ]
    return rounded_stage_info


# Format JSON with inline scalar arrays for more compact output.
def _format_json_compact(value: Any, indent: int = 4, level: int = 0) -> str:
    """Format JSON while keeping scalar arrays on a single line."""
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"

        current_indent = " " * (indent * level)
        child_indent = " " * (indent * (level + 1))
        items = [
            f"{child_indent}{json.dumps(key, ensure_ascii=False)}: {_format_json_compact(item, indent, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{current_indent}" + "}"

    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return json.dumps(value, ensure_ascii=False)

        current_indent = " " * (indent * level)
        child_indent = " " * (indent * (level + 1))
        items = [f"{child_indent}{_format_json_compact(item, indent, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{current_indent}" + "]"

    return json.dumps(value, ensure_ascii=False)


# Write JSON files using compact scalar-array formatting.
def _write_json(path: Path, value: Any) -> None:
    """Write JSON content with stable compact formatting."""
    path.write_text(_format_json_compact(value) + "\n", encoding="utf-8")
