"""Convert Cell Tracking Challenge datasets to OME-Zarr + GEFF.

Usage: `uv run ctc_convert.py train|test <dataset_dir> <out_dir> [--overwrite]`

Train layout written for each dataset::

    <dataset>-train.zarr/
        sample_<NN>/
            zarr.json, 0/, 1/, ...      # raw image
            labels/{GT-TRA,ST,GT-SEG,ERR-SEG}/
            tracks.geff                 # from GT-TRA

Test layout (raw images only for now)::

    <dataset>-test.zarr/
        sample_<NN>/
            zarr.json, 0/, 1/, ...
"""

import contextlib
import functools
import logging
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tifffile
import typer
import yaml
import zarr
from geff.convert import convert_ctc, ctc_tiffs_to_zarr
from geff.convert._ctc import _chunks_and_shards, _ome_dims, _register_label
from geff.convert._scale_pyramid import generate_scale_pyramid
from geff.core_io import read_to_memory
from geff.core_io._base_write import write_props_arrays
from geff_spec import GeffMetadata, RelatedObject
from geff_spec.utils import add_or_update_props_metadata
from tqdm import tqdm
from yaozarrs.v06 import ImageLabel, LabelImage, LabelSource, Multiscale
from yaozarrs.write.v06 import prepare_image

# Per-sequence physical metadata, keyed by "<dataset>_<seq>" (e.g. "BF-C2DL-HSC_01")
DEFAULT_SCALES_PATH = Path(__file__).parent / "scales.yaml"
# scales.yaml unit names -> OME-Zarr unit names
SPACE_UNITS = {"microns": "micrometer"}
TIME_UNITS = {"min": "minute"}

# label name -> (directory relative to <dataset>/, filename glob), formatted with the sequence
TRAIN_LABELS: dict[str, tuple[str, str]] = {
    "ST": ("{seq}_ST/SEG", "man_seg*.tif"),
    "GT-SEG": ("{seq}_GT/SEG", "man_seg*.tif"),
    "ERR-SEG": ("{seq}_ERR_SEG", "mask*.tif"),
}
GT_TRA_LABEL = "GT-TRA"
GT_TRA_DIR = "{seq}_GT/TRA"

_FRAME_INDEX = re.compile(r"(\d+)\.tiff?$")
_SLICE_INDEX = re.compile(r"_(\d+)_(\d+)\.tiff?$")

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def log_step(description: str) -> Iterator[None]:
    """Log the start and (timed) completion of a conversion step."""
    logger.info("%s ...", description)
    start = time.perf_counter()
    yield
    logger.info("%s done (%.1fs)", description, time.perf_counter() - start)


@functools.cache
def _load_scales(scales_path: Path) -> dict[str, dict[str, Any]]:
    with open(scales_path) as f:
        return yaml.safe_load(f)


def get_metadata(dataset_name: str, seq: str, scales_path: Path = DEFAULT_SCALES_PATH) -> dict:
    """geff converter kwargs (pixel_size, pixel_unit, time_scale, time_unit) for a sequence."""
    key = f"{dataset_name}_{seq}"
    scales = _load_scales(Path(scales_path))
    if key not in scales:
        raise KeyError(f"No entry {key!r} in {scales_path}")
    entry = scales[key]
    try:
        pixel_unit = SPACE_UNITS[entry["pixel_unit"]]
        time_unit = TIME_UNITS[entry["time_unit"]]
    except KeyError as e:
        raise ValueError(f"Unknown unit {e} for {key!r} in {scales_path}") from e
    return {
        "pixel_size": [float(v) for v in entry["pixel_scale"]],
        "pixel_unit": pixel_unit,
        "time_scale": float(entry["time_scale"]),
        "time_unit": time_unit,
    }


def find_sequences(dataset_dir: Path) -> list[str]:
    """Sequence names (digit-only subdirectories, e.g. "01", "02"), sorted."""
    sequences = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not sequences:
        raise FileNotFoundError(f"No sequence directories (e.g. '01') found in {dataset_dir}")
    return sequences


def frame_index(path: Path) -> int:
    """Frame index encoded in a CTC tiff filename (e.g. `man_seg0058.tif` -> 58)."""
    match = _FRAME_INDEX.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse a frame index from filename {path.name!r}")
    return int(match.group(1))


def slice_index(path: Path) -> tuple[int, int]:
    """(frame, z) encoded in a CTC slice tiff filename (e.g. `man_seg_000_013.tif` -> (0, 13))."""
    match = _SLICE_INDEX.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse a slice index from filename {path.name!r}")
    return int(match.group(1)), int(match.group(2))


def write_label_sequence(
    frame_dir: Path,
    pattern: str,
    sample_path: Path,
    label_name: str,
    overwrite: bool = False,
    pixel_size: list[float] | None = None,
    pixel_unit: Any = None,
    time_scale: float | None = None,
    time_unit: Any = None,
) -> None:
    """Write tiffs as an OME-Zarr label image in `<sample_path>/labels/<label_name>`.

    The array always spans every frame of the sample's raw image; frames without a tiff (e.g.
    the temporally sparse GT-SEG) are left as zeros. The sample's raw image must already exist.
    Labels are written at a single scale only, so only the original labels are distributed.

    The tiffs are either full frames (`<name><t>.tif`) or, for the sparse 3D GT-SEG, single
    (y, x) slices (`<name>_<t>_<z>.tif`) that are written into their (t, z) position, leaving
    the rest of the volume as zeros.
    """
    files = sorted(frame_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {frame_dir}")

    raw_shape = zarr.open_group(sample_path, mode="r")["0"].shape
    n_frames, frame_shape = raw_shape[0], raw_shape[1:]

    is_slice = [_SLICE_INDEX.search(f.name) is not None for f in files]
    if any(is_slice) and not all(is_slice):
        raise ValueError(f"Cannot mix slice (<t>_<z>) and full frame tiffs in {frame_dir}")
    slices = all(is_slice)
    indices = [slice_index(f) if slices else (frame_index(f),) for f in files]
    item_shape = frame_shape[1:] if slices else frame_shape

    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate frame indices in {frame_dir}")
    max_frame = max(index[0] for index in indices)
    if max_frame >= n_frames:
        raise ValueError(
            f"Frame index {max_frame} in {frame_dir} is out of range for {n_frames} raw frames"
        )
    if slices:
        max_z = max(index[1] for index in indices)
        if max_z >= raw_shape[1]:
            raise ValueError(
                f"Z index {max_z} in {frame_dir} is out of range for {raw_shape[1]} z slices"
            )

    first = tifffile.imread(files[0])
    if first.shape != item_shape:
        raise ValueError(
            f"Label {'slices' if slices else 'frames'} in {frame_dir} have shape {first.shape}, "
            f"expected {item_shape}"
        )
    if first.dtype.kind not in "iu":
        raise ValueError(f"Label images must have an integer data type, got {first.dtype}.")

    dims = _ome_dims(len(frame_shape), False, pixel_size, pixel_unit, time_scale, time_unit)
    label_image = LabelImage.model_validate(
        {
            "multiscales": [Multiscale.from_dims(dims)],
            # `labels/<label_name>` -> `../../` is the image, per the OME-Zarr convention
            "image-label": ImageLabel(source=LabelSource(image="../../")),
        }
    )
    shape = (n_frames, *frame_shape)
    chunks, shards = _chunks_and_shards(shape, False)
    label_path = sample_path / "labels" / label_name
    _, arrays = prepare_image(
        label_path,
        label_image,
        (shape, first.dtype),
        chunks=chunks,
        shards=shards,
        writer="zarr",
        overwrite=overwrite,
    )
    array = arrays["0"]
    logger.info(
        "Writing label %s: %d tiffs into %d frames", label_name, len(files), n_frames
    )
    for index, file in tqdm(
        zip(indices, files, strict=True), total=len(files), desc=label_name, unit="file"
    ):
        array[index] = np.asarray(tifffile.imread(file), dtype=first.dtype)

    _register_label(sample_path, label_name)


def add_label_ids_to_geff(sample_path: Path, label_name: str) -> None:
    """Add a node property with each node's id in the label image `label_name`.

    Node positions are pixel coordinates, so the label under a node's centroid is used
    directly. Nodes whose centroid is on background (including frames the label image has no
    tiff for) are marked missing. The label image is also added to the geff's related objects.
    """
    geff_path = sample_path / "tracks.geff"
    label_path = sample_path / "labels" / label_name
    labels = zarr.open_group(label_path, mode="r")["0"]

    axis_names = ["t", "y", "x"] if labels.ndim == 3 else ["t", "z", "y", "x"]
    node_props = read_to_memory(
        geff_path, structure_validation=False, node_props=axis_names, edge_props=[]
    )["node_props"]
    frames, *spatial = (np.rint(node_props[name]["values"]).astype(int) for name in axis_names)
    ids = np.zeros(len(frames), dtype=labels.dtype)
    for t in np.unique(frames):
        in_frame = frames == t
        frame = labels[int(t)]
        ids[in_frame] = frame[tuple(coords[in_frame] for coords in spatial)]
    missing = ids == 0
    if missing.any():
        logger.warning(
            "%d of %d nodes are on background in label %s", missing.sum(), len(ids), label_name
        )

    prop_name = f"{label_name.lower().replace('-', '_')}_id"
    props_metadata = write_props_arrays(
        geff_path,
        "nodes",
        {prop_name: {"values": ids, "missing": missing if missing.any() else None}},
    )
    metadata = add_or_update_props_metadata(GeffMetadata.read(geff_path), props_metadata, "node")
    metadata.related_objects = [
        *(metadata.related_objects or []),
        RelatedObject(
            type="labels", path=os.path.relpath(label_path, geff_path), node_prop=prop_name
        ),
    ]
    metadata.write(geff_path)


def convert_train_sample(
    dataset_dir: Path,
    seq: str,
    sample_path: Path,
    overwrite: bool = False,
    num_scale_levels: int | None = None,
    **metadata: Any,
) -> None:
    """Convert sequence `seq` of a CTC training dataset to `sample_path`."""
    # raw image, GT-TRA labels and tracks.geff; goes first as overwriting removes `labels/`
    with log_step("Converting raw images, GT-TRA labels, tracks.geff and pyramids"):
        convert_ctc(
            dataset_dir / GT_TRA_DIR.format(seq=seq),
            dataset_dir / seq,
            sample_path,
            label_name=GT_TRA_LABEL,
            overwrite=overwrite,
            num_scale_levels=num_scale_levels,
            **metadata,
        )
    for label_name, (subdir, pattern) in TRAIN_LABELS.items():
        write_label_sequence(
            dataset_dir / subdir.format(seq=seq),
            pattern,
            sample_path,
            label_name,
            overwrite=overwrite,
            **metadata,
        )
        add_label_ids_to_geff(sample_path, label_name)


def convert_train_dataset(
    dataset_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    num_scale_levels: int | None = None,
    scales_path: Path = DEFAULT_SCALES_PATH,
) -> Path:
    """Convert every sequence of a CTC training dataset to `<out_dir>/<name>-train.zarr`."""
    dataset_dir = Path(dataset_dir)
    name = dataset_dir.name
    root = Path(out_dir) / f"{name}-train.zarr"

    sequences = find_sequences(dataset_dir)
    zarr.open_group(root, mode="a", zarr_format=3)
    logger.info("Converting %s (%d sequences) -> %s", name, len(sequences), root)
    for i, seq in enumerate(sequences, start=1):
        logger.info("Sequence %s (%d/%d)", seq, i, len(sequences))
        convert_train_sample(
            dataset_dir,
            seq,
            root / f"sample_{seq}",
            overwrite=overwrite,
            num_scale_levels=num_scale_levels,
            **get_metadata(name, seq, scales_path),
        )
    return root


def convert_test_dataset(
    dataset_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    num_scale_levels: int | None = None,
    scales_path: Path = DEFAULT_SCALES_PATH,
) -> Path:
    """Convert the raw images of a CTC test dataset to `<out_dir>/<name>-test.zarr`.

    TODO: tracks.geff (sparse tracks, no edges) and the drosophila test labels.
    """
    dataset_dir = Path(dataset_dir)
    name = dataset_dir.name
    root = Path(out_dir) / f"{name}-test.zarr"

    sequences = find_sequences(dataset_dir)
    zarr.open_group(root, mode="a", zarr_format=3)
    logger.info("Converting %s (%d sequences) -> %s", name, len(sequences), root)
    for i, seq in enumerate(sequences, start=1):
        logger.info("Sequence %s (%d/%d)", seq, i, len(sequences))
        sample_path = root / f"sample_{seq}"
        metadata = get_metadata(name, seq, scales_path)
        with log_step("Converting raw images"):
            ctc_tiffs_to_zarr(dataset_dir / seq, sample_path, overwrite=overwrite, **metadata)
        with log_step("Building scale pyramid for raw images"):
            generate_scale_pyramid(
                zarr.open_group(sample_path, mode="r+"), num_levels=num_scale_levels
            )
    return root


app = typer.Typer(add_completion=False)


@app.callback()
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

DatasetDir = Annotated[Path, typer.Argument(exists=True, file_okay=False)]
OutDir = Annotated[Path, typer.Argument(file_okay=False)]
NumScaleLevels = Annotated[int | None, typer.Option(min=1)]
ScalesPath = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command("train")
def train_command(
    dataset_dir: DatasetDir,
    out_dir: OutDir,
    overwrite: bool = False,
    num_scale_levels: NumScaleLevels = None,
    scales_path: ScalesPath = DEFAULT_SCALES_PATH,
) -> None:
    """Convert a CTC training dataset to <out_dir>/<dataset>-train.zarr (OME-Zarr + GEFF)."""
    root = convert_train_dataset(dataset_dir, out_dir, overwrite, num_scale_levels, scales_path)
    logger.info("Wrote %s", root)


@app.command("test")
def test_command(
    dataset_dir: DatasetDir,
    out_dir: OutDir,
    overwrite: bool = False,
    num_scale_levels: NumScaleLevels = None,
    scales_path: ScalesPath = DEFAULT_SCALES_PATH,
) -> None:
    """Convert a CTC test dataset to <out_dir>/<dataset>-test.zarr (raw images only)."""
    root = convert_test_dataset(dataset_dir, out_dir, overwrite, num_scale_levels, scales_path)
    logger.info("Wrote %s", root)


if __name__ == "__main__":
    app()
