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
from geff.convert._scale_pyramid import DEFAULT_SPATIAL_CHUNK_SIZE, generate_scale_pyramid
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


def write_label_sequence(
    frame_dir: Path,
    pattern: str,
    sample_path: Path,
    label_name: str,
    num_scale_levels: int | None = None,
    min_chunk_size: int = DEFAULT_SPATIAL_CHUNK_SIZE,
    overwrite: bool = False,
    pixel_size: list[float] | None = None,
    pixel_unit: Any = None,
    time_scale: float | None = None,
    time_unit: Any = None,
) -> None:
    """Write tiffs as an OME-Zarr label image in `<sample_path>/labels/<label_name>`.

    The array always spans every frame of the sample's raw image; frames without a tiff (e.g.
    the temporally sparse GT-SEG) are left as zeros. The sample's raw image must already exist.
    """
    files = sorted(frame_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {frame_dir}")

    raw_shape = zarr.open_group(sample_path, mode="r")["0"].shape
    n_frames, frame_shape = raw_shape[0], raw_shape[1:]

    indices = [frame_index(f) for f in files]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate frame indices in {frame_dir}")
    if max(indices) >= n_frames:
        raise ValueError(
            f"Frame index {max(indices)} in {frame_dir} is out of range for "
            f"{n_frames} raw frames"
        )

    first = tifffile.imread(files[0])
    if first.shape != frame_shape:
        raise ValueError(
            f"Label frames in {frame_dir} have shape {first.shape}, raw frames {frame_shape}"
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
    for t, file in tqdm(
        zip(indices, files, strict=True), total=len(files), desc=label_name, unit="frame"
    ):
        array[t] = np.asarray(tifffile.imread(file), dtype=first.dtype)

    _register_label(sample_path, label_name)
    with log_step(f"Building scale pyramid for label {label_name}"):
        generate_scale_pyramid(
            zarr.open_group(label_path, mode="r+"),
            num_levels=num_scale_levels,
            chunk_size=min_chunk_size,
            is_label=True,
        )


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
            num_scale_levels=num_scale_levels,
            overwrite=overwrite,
            **metadata,
        )


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
