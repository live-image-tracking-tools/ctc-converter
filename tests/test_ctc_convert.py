import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from ctc_convert import (
    get_metadata,
    convert_test_dataset,
    convert_train_dataset,
    find_sequences,
    frame_index,
    write_label_sequence,
)

N_FRAMES = 4
SHAPE = (64, 64)
SPARSE_FRAMES = (1, 3)


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
