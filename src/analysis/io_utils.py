"""Common I/O and archiving utilities for scientific analyses.

Provides standardized functions to create dated archives with metadata,
and save figures in both PNG and PLT (PltEdit) formats.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pltedit
from vfscitools import archive
from vfscitools.archive import Archive


def _clean_for_yaml(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and astropy quantities into YAML-safe types."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_clean_for_yaml(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _clean_for_yaml(v) for k, v in obj.items()}
    if hasattr(obj, "unit") and hasattr(obj, "value"):
        return f"{obj.value} {obj.unit}"
    return obj


def get_archive(
    analysis_dir: str | os.PathLike[str],
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    verbose: bool = False,
    **parameters: Any,
) -> Archive:
    """Create a new dated archive folder inside ``<analysis_dir>/archives/``."""
    root = Path(analysis_dir).resolve() / "archives"
    clean_params = {k: _clean_for_yaml(v) for k, v in parameters.items()}
    clean_meta = {k: _clean_for_yaml(v) for k, v in metadata.items()} if metadata else None
    return archive.new(name=name, root=root, verbose=verbose, metadata=clean_meta, **clean_params)


def save_figure(
    fig: plt.Figure | None = None,
    target: str | os.PathLike[str] | Archive | None = None,
    default_name: str = "figure",
    dpi: int = 300,
    analysis_name: str | None = None,
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Save a matplotlib figure in both PNG and PLT (PltEdit) formats.

    Supports multiple calling conventions:
    - save_figure(fig, target="path/to/dir", default_name="my_plot")
    - save_figure(fig, "my_plot", save_as="path/to/dir", analysis_name="analysis")
    - save_figure(fig, "my_plot", analysis_name="analysis")
    """
    if fig is None:
        fig = plt.gcf()

    # Determine if target and default_name were inverted
    # e.g., save_figure(fig, "plot_name", save_as_path, analysis_name=...)
    if isinstance(target, str) and not isinstance(default_name, str):
        target, default_name = default_name, str(target)
    elif isinstance(target, str) and isinstance(default_name, str):
        # If target has no path separators and default_name has path separators or is 'archives'
        if ("/" not in target and "\\" not in target and not target.endswith(".png") and not target.endswith(".plt")) and (
            "/" in default_name or "\\" in default_name or default_name in ("archives", "generated", "scratch")
        ):
            target, default_name = default_name, target

    # Handle analysis_name fallback if target is 'archives' or relative
    if (target is None or str(target) == "archives") and analysis_name:
        analysis_root = Path("e:/PhD-Theory/src/analysis") / analysis_name
        arc = get_archive(analysis_root, name=default_name)
        target = arc.path

    if target is None:
        target_path = Path.cwd() / default_name
    elif isinstance(target, Archive):
        target_path = target.path / default_name
    else:
        target_path = Path(target)
        if target_path.is_dir() or (not target_path.suffix and not target_path.exists()):
            if target_path.is_dir():
                target_path = target_path / default_name
        else:
            target_path = target_path.with_suffix("")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = target_path.with_suffix(".png")
    plt_path = target_path.with_suffix(".plt")

    # Save PNG
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    # Save PLT via PltEdit
    pltedit.save(fig, plt_path)

    return png_path, plt_path


def save_dataset(
    target: Any,
    filename: str | None = None,
    save_as: str | os.PathLike[str] | Archive | None = None,
    analysis_name: str | None = None,
    **kwargs: Any,
) -> Path:
    """Save numerical data to an .npz file in the archive or target directory.

    Supports both:
    - save_dataset(dict_data, "filename", save_as=path, analysis_name="name")
    - save_dataset(target_path, "filename", **data)
    """
    if isinstance(target, dict):
        data = target
        file_base = filename or "dataset"
        dest = save_as or kwargs.pop("save_as", None) or kwargs.pop("target", None)
    else:
        dest = target
        file_base = filename or "dataset"
        data = kwargs

    if (dest is None or str(dest) == "archives") and analysis_name:
        analysis_root = Path("e:/PhD-Theory/src/analysis") / analysis_name
        arc = get_archive(analysis_root, name=file_base)
        dest = arc.path

    if dest is None:
        dest_dir = Path.cwd()
    elif isinstance(dest, Archive):
        dest_dir = dest.path
    else:
        dest_dir = Path(dest)
        if dest_dir.is_file() or dest_dir.suffix:
            dest_dir = dest_dir.parent

    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file_base
    if file_path.suffix != ".npz":
        file_path = file_path.with_suffix(".npz")

    # Ensure arrays are numpy-compatible
    clean_data = {}
    for k, v in data.items():
        if isinstance(v, (np.ndarray, list, tuple, float, int, bool)):
            clean_data[k] = np.asarray(v) if not isinstance(v, (float, int, bool)) else v
        else:
            clean_data[k] = np.asarray([_clean_for_yaml(v)])

    np.savez_compressed(file_path, **clean_data)
    return file_path
