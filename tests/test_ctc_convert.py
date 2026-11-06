import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from geff.convert import ctc_tiffs_to_zarr
from geff.core_io import read_to_memory
from geff_spec import GeffMetadata

from ctc_convert import (
    add_label_ids_to_geff,
    get_metadata,
    convert_test_dataset,
    convert_train_dataset,
    find_sequences,
    frame_index,
    slice_index,
    write_label_sequence,
)

N_FRAMES = 4
SHAPE = (64, 64)
SPARSE_FRAMES = (1, 3)
N_Z = 5
SLICE_SHAPE = (16, 16)


def _write_3d_raw_sample(tmp_path: Path) -> Path:
    """An OME-Zarr sample with a raw (T, Z, Y, X) image."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for t in range(N_FRAMES):
        tifffile.imwrite(raw / f"t{t:03d}.tif", np.full((N_Z, *SLICE_SHAPE), t, dtype=np.uint8))
    sample = tmp_path / "sample"
    ctc_tiffs_to_zarr(raw, sample)
    return sample


def _write_slices(label_dir: Path, slices: dict[tuple[int, int], int]) -> None:
    """Write a 2D tiff `man_seg_<t>_<z>.tif` filled with the given label for each (t, z)."""
    label_dir.mkdir()
    for (t, z), label in slices.items():
        tifffile.imwrite(
            label_dir / f"man_seg_{t:03d}_{z:03d}.tif", np.full(SLICE_SHAPE, label, dtype=np.uint16)
        )


def _labels_frame(t: int) -> np.ndarray:
    frame = np.zeros(SHAPE, dtype=np.uint16)
    frame[5:15, 5 + t : 15 + t] = 1
    frame[40:50, 40:50] = 2
    return frame


def _write_seq(root: Path, seq: str) -> None:
    (root / seq).mkdir(parents=True)
    tra = root / f"{seq}_GT" / "TRA"
    seg = root / f"{seq}_GT" / "SEG"
    st = root / f"{seq}_ST" / "SEG"
    err = root / f"{seq}_ERR_SEG"
    for d in (tra, seg, st, err):
        d.mkdir(parents=True)
    for t in range(N_FRAMES):
        tifffile.imwrite(root / seq / f"t{t:04d}.tif", np.full(SHAPE, t, dtype=np.uint8))
        frame = _labels_frame(t)
        tifffile.imwrite(tra / f"man_track{t:04d}.tif", frame)
        tifffile.imwrite(st / f"man_seg{t:04d}.tif", frame)
        tifffile.imwrite(err / f"mask{t:04d}.tif", frame)
        if t in SPARSE_FRAMES:
            tifffile.imwrite(seg / f"man_seg{t:04d}.tif", frame)
    (tra / "man_track.txt").write_text(f"1 0 {N_FRAMES - 1} 0\n2 0 {N_FRAMES - 1} 0\n")


SCALES = """
Fake-Dataset_01:
  pixel_unit: microns
  pixel_scale: [0.5, 0.25]
  time_unit: min
  time_scale: 5
Fake-Dataset_02:
  pixel_unit: microns
  pixel_scale: [0.5, 0.25]
  time_unit: min
  time_scale: 5
"""


@pytest.fixture
def scales(tmp_path: Path) -> Path:
    path = tmp_path / "scales.yaml"
    path.write_text(SCALES)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "Fake-Dataset"
    for seq in ("01", "02"):
        _write_seq(root, seq)
    return root


class Test_frame_index:
    def test_parses_trailing_digits(self):
        assert frame_index(Path("man_seg0058.tif")) == 58

    def test_raises_without_digits(self):
        with pytest.raises(ValueError, match="Cannot parse a frame index"):
            frame_index(Path("man_seg.tif"))


class Test_slice_index:
    def test_parses_frame_and_z(self):
        assert slice_index(Path("man_seg_000_013.tif")) == (0, 13)

    def test_raises_for_full_frame_filename(self):
        with pytest.raises(ValueError, match="Cannot parse a slice index"):
            slice_index(Path("man_seg0058.tif"))


class Test_find_sequences:
    def test_finds_digit_directories_only(self, dataset: Path):
        assert find_sequences(dataset) == ["01", "02"]

    def test_raises_when_empty(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No sequence directories"):
            find_sequences(tmp_path)


class Test_convert_train_dataset:
    def test_layout(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        assert root.name == "Fake-Dataset-train.zarr"
        for seq in ("01", "02"):
            sample = root / f"sample_{seq}"
            assert (sample / "zarr.json").exists()
            assert (sample / "tracks.geff").exists()
            assert zarr.open_group(sample)["0"].shape == (N_FRAMES, *SHAPE)
            labels_meta = json.loads((sample / "labels" / "zarr.json").read_text())
            assert sorted(labels_meta["attributes"]["ome"]["labels"]) == [
                "ERR-SEG",
                "GT-SEG",
                "GT-TRA",
                "ST",
            ]
            for label in ("GT-TRA", "ST", "GT-SEG", "ERR-SEG"):
                assert (sample / "labels" / label / "zarr.json").exists()

    def test_gt_seg_is_zero_on_missing_frames(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        gt_seg = zarr.open_group(root / "sample_01" / "labels" / "GT-SEG")["0"][:]
        for t in range(N_FRAMES):
            expected = _labels_frame(t) if t in SPARSE_FRAMES else np.zeros(SHAPE)
            np.testing.assert_array_equal(gt_seg[t], expected)

    def test_dense_labels_match_input(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        st = zarr.open_group(root / "sample_02" / "labels" / "ST")["0"][:]
        for t in range(N_FRAMES):
            np.testing.assert_array_equal(st[t], _labels_frame(t))

    def test_existing_output_requires_overwrite(self, dataset: Path, tmp_path: Path, scales: Path):
        convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        with pytest.raises(FileExistsError, match="Use overwrite=True"):
            convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        convert_train_dataset(dataset, tmp_path / "out", overwrite=True, scales_path=scales)


class Test_label_ids_on_nodes:
    @pytest.fixture
    def geff_path(self, dataset: Path, tmp_path: Path, scales: Path) -> Path:
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        return root / "sample_01" / "tracks.geff"

    def test_each_label_is_a_related_object_with_its_own_node_prop(self, geff_path: Path):
        metadata = GeffMetadata.read(geff_path)
        assert {Path(r.path).name: r.node_prop for r in metadata.related_objects} == {
            "GT-TRA": "tracklet_id",
            "ST": "st_id",
            "GT-SEG": "gt_seg_id",
            "ERR-SEG": "err_seg_id",
        }
        assert {"st_id", "gt_seg_id", "err_seg_id"} <= set(metadata.node_props_metadata)

    def test_dense_labels_match_tracklet_ids(self, geff_path: Path):
        props = read_to_memory(geff_path)["node_props"]
        for name in ("st_id", "err_seg_id"):
            np.testing.assert_array_equal(props[name]["values"], props["tracklet_id"]["values"])
            assert props[name]["missing"] is None

    def test_sparse_gt_seg_is_missing_on_frames_without_labels(self, geff_path: Path):
        props = read_to_memory(geff_path)["node_props"]
        has_label = np.isin(props["t"]["values"], SPARSE_FRAMES)
        gt_seg = props["gt_seg_id"]
        np.testing.assert_array_equal(gt_seg["missing"], ~has_label)
        np.testing.assert_array_equal(
            gt_seg["values"][has_label], props["tracklet_id"]["values"][has_label]
        )


class Test_add_label_ids_to_geff:
    def test_uses_label_under_node_and_marks_background_missing(
        self, dataset: Path, tmp_path: Path, scales: Path
    ):
        sample = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales) / "sample_01"

        # A label image with different ids and larger objects than GT-TRA's, in which the
        # second object is absent in the last frame.
        new_labels = tmp_path / "new_labels"
        new_labels.mkdir()
        for t in range(N_FRAMES):
            frame = np.zeros(SHAPE, dtype=np.uint16)
            frame[3:17, 3 + t : 17 + t] = 7
            if t < N_FRAMES - 1:
                frame[38:52, 38:52] = 9
            tifffile.imwrite(new_labels / f"man_seg{t:04d}.tif", frame)
        write_label_sequence(new_labels, "man_seg*.tif", sample, "NEW")

        add_label_ids_to_geff(sample, "NEW")

        geff_path = sample / "tracks.geff"
        props = read_to_memory(geff_path)["node_props"]
        tracklet = props["tracklet_id"]["values"]
        t = props["t"]["values"]
        new_id = props["new_id"]
        missing = (tracklet == 2) & (t == N_FRAMES - 1)
        np.testing.assert_array_equal(new_id["missing"], missing)
        np.testing.assert_array_equal(new_id["values"][tracklet == 1], 7)
        np.testing.assert_array_equal(new_id["values"][(tracklet == 2) & ~missing], 9)

        related = {Path(r.path).name: r.node_prop for r in GeffMetadata.read(geff_path).related_objects}
        assert related["NEW"] == "new_id"


class Test_write_label_sequence:
    def test_raises_on_out_of_range_frame(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        extra = tmp_path / "extra"
        extra.mkdir()
        tifffile.imwrite(extra / "man_seg0099.tif", _labels_frame(0))
        with pytest.raises(ValueError, match="out of range for 4 raw frames"):
            write_label_sequence(extra, "man_seg*.tif", root / "sample_01", "BAD")

    def test_raises_on_shape_mismatch(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        extra = tmp_path / "extra"
        extra.mkdir()
        tifffile.imwrite(extra / "man_seg0000.tif", np.zeros((8, 8), dtype=np.uint16))
        with pytest.raises(ValueError, match="have shape"):
            write_label_sequence(extra, "man_seg*.tif", root / "sample_01", "BAD")

    def test_raises_without_matching_files(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_train_dataset(dataset, tmp_path / "out", scales_path=scales)
        with pytest.raises(FileNotFoundError, match="No files matching"):
            write_label_sequence(tmp_path, "nothing*.tif", root / "sample_01", "BAD")


class Test_write_label_sequence_slices:
    def test_slots_each_slice_into_its_frame_and_z(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        # two annotated slices in frame 0, one in frame 2, none in frames 1 and 3
        slices = {(0, 1): 1, (0, 3): 2, (2, 4): 3}
        _write_slices(tmp_path / "seg", slices)

        write_label_sequence(tmp_path / "seg", "man_seg_*.tif", sample, "GT-SEG")

        labels = zarr.open_group(sample / "labels" / "GT-SEG")["0"][:]
        expected = np.zeros((N_FRAMES, N_Z, *SLICE_SHAPE), dtype=np.uint16)
        for (t, z), label in slices.items():
            expected[t, z] = label
        np.testing.assert_array_equal(labels, expected)

    def test_raises_on_out_of_range_z(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        _write_slices(tmp_path / "seg", {(0, N_Z): 1})
        with pytest.raises(ValueError, match=f"out of range for {N_Z} z slices"):
            write_label_sequence(tmp_path / "seg", "man_seg_*.tif", sample, "GT-SEG")

    def test_raises_on_out_of_range_frame(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        _write_slices(tmp_path / "seg", {(N_FRAMES, 0): 1})
        with pytest.raises(ValueError, match=f"out of range for {N_FRAMES} raw frames"):
            write_label_sequence(tmp_path / "seg", "man_seg_*.tif", sample, "GT-SEG")

    def test_raises_on_wrong_slice_shape(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        (tmp_path / "seg").mkdir()
        tifffile.imwrite(tmp_path / "seg" / "man_seg_000_001.tif", np.zeros((8, 8), np.uint16))
        with pytest.raises(ValueError, match="have shape"):
            write_label_sequence(tmp_path / "seg", "man_seg_*.tif", sample, "GT-SEG")

    def test_raises_on_duplicate_slice(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        _write_slices(tmp_path / "seg", {(0, 1): 1})
        tifffile.imwrite(tmp_path / "seg" / "other_000_001.tif", np.ones(SLICE_SHAPE, np.uint16))
        with pytest.raises(ValueError, match="Duplicate"):
            write_label_sequence(tmp_path / "seg", "*.tif", sample, "GT-SEG")

    def test_raises_on_mixed_slices_and_full_frames(self, tmp_path: Path):
        sample = _write_3d_raw_sample(tmp_path)
        _write_slices(tmp_path / "seg", {(0, 1): 1})
        tifffile.imwrite(
            tmp_path / "seg" / "man_seg0002.tif", np.ones((N_Z, *SLICE_SHAPE), np.uint16)
        )
        with pytest.raises(ValueError, match="Cannot mix"):
            write_label_sequence(tmp_path / "seg", "*.tif", sample, "GT-SEG")


class Test_convert_test_dataset:
    def test_writes_raw_only(self, dataset: Path, tmp_path: Path, scales: Path):
        root = convert_test_dataset(dataset, tmp_path / "out", scales_path=scales)
        assert root.name == "Fake-Dataset-test.zarr"
        sample = root / "sample_01"
        raw = zarr.open_group(sample)["0"][:]
        assert raw.shape == (N_FRAMES, *SHAPE)
        assert raw[2, 0, 0] == 2
        assert not (sample / "labels").exists()
        assert not (sample / "tracks.geff").exists()


class Test_get_metadata:
    def test_maps_units_and_scales(self, scales: Path):
        assert get_metadata("Fake-Dataset", "01", scales) == {
            "pixel_size": [0.5, 0.25],
            "pixel_unit": "micrometer",
            "time_scale": 5.0,
            "time_unit": "minute",
        }

    def test_raises_for_missing_sequence(self, scales: Path):
        with pytest.raises(KeyError, match="No entry 'Fake-Dataset_03'"):
            get_metadata("Fake-Dataset", "03", scales)

    def test_raises_for_unknown_unit(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "X_01:\n  pixel_unit: parsecs\n  pixel_scale: [1, 1]\n  time_unit: min\n  time_scale: 1\n"
        )
        with pytest.raises(ValueError, match="Unknown unit 'parsecs'"):
            get_metadata("X", "01", path)

    def test_real_scales_file_covers_bf_c2dl_hsc(self):
        meta = get_metadata("BF-C2DL-HSC", "02")
        assert meta["pixel_size"] == [0.645, 0.645]
        assert meta["time_scale"] == 5.0
