#!/usr/bin/env python3
"""Helpers for publishing benchmark output into device-specific reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


CPU_REPORT_PATH = Path(__file__).with_name("README_CPU.md")
GPU_REPORT_PATH = Path(__file__).with_name("README_GPU.md")


def report_path_for_device(device: object) -> Path:
    normalized = str(device).lower()
    if normalized.startswith("cpu"):
        return CPU_REPORT_PATH
    if normalized.startswith("cuda"):
        return GPU_REPORT_PATH
    raise ValueError(f"Unsupported benchmark report device: {device!r}")


def report_title_for_path(report_path: Path) -> str:
    if report_path == CPU_REPORT_PATH:
        return "# CPU Benchmark Reports"
    if report_path == GPU_REPORT_PATH:
        return "# GPU Benchmark Reports"
    return "# Benchmark Reports"


def enrich_device_settings(settings: dict[str, object]) -> dict[str, object]:
    device = str(settings.get("device", "")).lower()
    if not device.startswith("cuda"):
        return settings

    import torch

    enriched: dict[str, object] = {}
    for key, value in settings.items():
        enriched[key] = value
        if key == "device":
            enriched["gpu"] = torch.cuda.get_device_name(torch.device(device))
            enriched["torch"] = str(torch.__version__)
    return enriched


def format_setting_value(value: object) -> str:
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value!r}`"


def render_settings(settings: dict[str, object]) -> list[str]:
    lines = ["Settings:"]
    for key, value in settings.items():
        lines.append(f"- `{key}`: {format_setting_value(value)}")
    return lines


def render_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return [header, divider, *body]


def update_report_section(
    *,
    section_id: str,
    title: str,
    script_name: str,
    settings: dict[str, object],
    columns: list[str],
    rows: list[list[str]],
    report_path: Optional[Path] = None,
) -> None:
    if report_path is None:
        report_path = report_path_for_device(settings.get("device"))

    start_marker = f"<!-- benchmark-report:{section_id} start -->"
    end_marker = f"<!-- benchmark-report:{section_id} end -->"

    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
    else:
        content = report_title_for_path(report_path) + "\n\nLatest benchmark snapshots written directly by the scripts in this folder.\n"
    replacement_lines = [
        start_marker,
        f"## {title}",
        "",
        f"Source: `{script_name}`",
        "",
        *render_settings(enrich_device_settings(settings)),
        "",
        "Results:",
        "",
        *render_table(columns, rows),
        end_marker,
    ]
    replacement = "\n".join(replacement_lines)

    start = content.find(start_marker)
    end = content.find(end_marker)
    if start < 0 and end < 0:
        updated = content.rstrip() + "\n\n" + replacement
    elif start < 0 or end < 0 or end < start:
        raise ValueError(f"Incomplete report markers for section {section_id!r}.")
    else:
        end += len(end_marker)
        updated = content[:start] + replacement + content[end:]
    report_path.write_text(updated + ("\n" if not updated.endswith("\n") else ""), encoding="utf-8")
