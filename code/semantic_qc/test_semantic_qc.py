"""Synthetic checks only: no real images, reviews, or experiment outputs are opened."""
import ast
import contextlib
import io
import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter, laplace

import build_semantic_roi as builder
import semantic_common as common
import verify_semantic_roi as validator

PROJECT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def six_image_schema():
    with patch.object(common, "EXPECTED_IMAGES", 6), patch.object(builder, "EXPECTED_IMAGES", 6), patch.object(validator, "EXPECTED_IMAGES", 6):
        yield


def fixture(root):
    source = root / "code/image_processing_cpu"
    foreground = root / "code/image_processing"
    data = root / "data"
    for path in [data, root / "code/data_audit", root / "code/experiments", source / "feature_records", source / "detections", foreground / "foreground_masks"]:
        path.mkdir(parents=True, exist_ok=True)
    for directory in common.AUTO_IMAGE_DIRS:
        (source / directory).mkdir()
    shutil.copyfile(PROJECT / "code/experiments/run_experiments.py", root / "code/experiments/run_experiments.py")
    for name in ["roi_pipeline.py", "roi_job_control.py", "verify_roi_outputs.py"]:
        shutil.copyfile(PROJECT / "code/image_processing" / name, source / name)
    converter, _, _ = common.color_converter(root)
    frozen_tree = ast.parse((PROJECT / "code/image_processing/roi_pipeline.py").read_text())
    stats_node = next(node for node in frozen_tree.body if isinstance(node, ast.FunctionDef) and node.name == "stats")
    namespace = {"np": np, "CHANNELS": common.CHANNELS, "rgb_to_color9": converter,
                 "binary_erosion": binary_erosion, "gaussian_filter": gaussian_filter, "laplace": laplace}
    exec(compile(ast.Module(body=[stats_node], type_ignores=[]), "frozen_source_stats", "exec"), namespace)
    stats = namespace["stats"]
    metadata, inventory, originals, reviews = [], [], [], []
    for index in range(6):
        image = ((np.arange(16 * 12 * 3).reshape(12, 16, 3) * 19 + index * 31) % 256).astype(np.uint8)
        image_path = f"synthetic_{index}.png"
        Image.fromarray(image).save(data / image_path)
        fg = np.zeros((12, 16), bool)
        fg[2:11, 2:15] = True
        beak = np.zeros_like(fg)
        beak[3:5, 0:3] = True
        feather = fg.copy()
        feather[2:6, 2:7] = False
        feather &= ~beak
        flags = {"foreground_ok": int(index != 5), "head_ok": int(index != 5), "beak_ok": int(index != 5), "feather_ok": int(index != 5)}
        primary = fg | (beak if flags["beak_ok"] else False)
        flip = index != 5
        Image.fromarray(fg.astype(np.uint8) * 255).save(foreground / "foreground_masks" / f"{index:04d}.png")
        for region, mask in [("beak", beak), ("feather", feather)]:
            Image.fromarray(mask.astype(np.uint8) * 255).save(source / (region + "_masks") / f"{index:04d}.png")
        for region, mask in [("foreground", primary), ("beak", beak), ("feather", feather)]:
            Image.fromarray((np.fliplr(mask) if flip else mask).astype(np.uint8) * 255).save(source / ("aligned_" + region + "_masks") / f"{index:04d}.png")
        rgba = np.dstack([np.fliplr(image) if flip else image, (np.fliplr(primary) if flip else primary).astype(np.uint8) * 255])
        Image.fromarray(rgba).save(source / "aligned_rgba" / f"{index:04d}.png")
        (source / "detections" / f"{index:04d}.json").write_text("{}")
        meta = {"image_index": index, "image_path": image_path, "age_label": index % 5 + 1,
                "day_group": f"synthetic_day_{index}", "width": 16, "height": 12}
        metadata.append(meta)
        inventory.append({"image_path": image_path, "sha256": common.sha256_file(data / image_path)})
        rgb = image.astype(np.float32) / 255
        gray = (rgb @ np.array([.299, .587, .114], np.float32)) * 255
        record = {**meta, **flags, "foreground_confidence": .8 if flags["foreground_ok"] else .1,
                  "head_confidence": .7 if flags["head_ok"] else 0., "beak_confidence": .6 if flags["beak_ok"] else 0.,
                  "head_box_json": "[2,2,7,6]", "beak_box_json": "[0,3,3,5]", "orientation_flip": int(flip),
                  "orientation_confidence": .6 if flip else 0., "head_to_beak_dx": -3. if flip else float("nan"),
                  "head_to_beak_dy": 0. if flip else float("nan"), "roi_qc": "pass" if flip else "beak_uncertain"}
        for region, mask in [("foreground", fg), ("beak", beak), ("feather", feather)]:
            record.update(stats(rgb, gray, mask, region, flags[region + "_ok"]))
        record["background_gray_mean"] = float(gray[~primary].mean())
        record["background_gray_std"] = float(gray[~primary].std())
        originals.append(record)
        common.atomic_json(source / "feature_records" / f"{index:04d}.json", record, allow_nan=True)
        review = {"qc_id": f"QC{6-index:04d}", "image_index": index, **{field: "accept" for field in common.REVIEW_FIELDS}, "reason": "synthetic"}
        if index == 1:
            review["head_correct"] = "uncertain"
        if index == 2:
            review["beak_correct"] = "reject"
        if index == 3:
            review["foreground_usable"] = "reject"
        if index == 4:
            review["feather_correct"] = "uncertain"
            review["orientation_usable"] = "uncertain"
        reviews.append(review)
    common.write_csv(root / "code/data_audit/image_metadata.csv", metadata)
    common.write_csv(root / "code/data_audit/inventory.csv", inventory)
    common.write_csv(source / "features.csv", originals)
    np.savez(source / "raw_roi_features.npz", features=np.zeros((6, 18), np.float32))
    source_config = {"model_device": "cpu"}
    common.atomic_json(source / "processing_config.json", source_config)
    common.atomic_json(source / "processing_summary.json", {"n_processed": 6, "device": "cpu",
                       "records_sha256": common.original_record_digest(source, 6),
                       "source_script_sha256": common.sha256_file(source / "roi_pipeline.py"),
                       "config_sha256": hashlib.sha256(json.dumps(source_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    common.write_csv(foreground / "foreground_manifest.csv", [{"image_index": index} for index in range(6)])
    manifest, mapping = root / "reviews.csv", root / "mapping.csv"
    common.write_csv(manifest, reviews, common.MANIFEST_FIELDS)
    common.write_csv(mapping, [{"anonymous_id": row["qc_id"], "image_index": row["image_index"]} for row in reviews])
    return source, foreground, data, manifest, mapping


class SemanticChecks(unittest.TestCase):
    def test_real_1143_gate_rejects_partial_reviews_before_output(self):
        with tempfile.TemporaryDirectory(prefix="semantic_schema_") as temporary:
            root = Path(temporary)
            mapping = root / "mapping.csv"
            manifest = root / "reviews.csv"
            common.write_csv(mapping, [{"anonymous_id": f"QC{index+1:04d}", "image_index": index} for index in range(1143)])
            common.write_csv(manifest, [{"qc_id": "QC0001", "image_index": 0, **{field: "accept" for field in common.REVIEW_FIELDS}, "reason": "synthetic"}], common.MANIFEST_FIELDS)
            with self.assertRaisesRegex(ValueError, "All 1143 reviews"):
                builder.build(root, root / "code/image_processing_cpu", root / "code/image_processing", root / "data", manifest, mapping)
            self.assertFalse((root / "code" / common.VERSION).exists())

    def test_synthetic_build_validation_and_no_healing(self):
        with tempfile.TemporaryDirectory(prefix="semantic_integration_") as temporary, six_image_schema():
            root = Path(temporary)
            source, foreground, data, manifest, mapping = fixture(root)
            before = {path: common.sha256_file(path) for path in source.rglob("*") if path.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                output = builder.build(root, source, foreground, data, manifest, mapping)
                self.assertFalse((output / "ROI_VALIDATED.flag").exists())
                self.assertTrue(validator.verify(root))
            self.assertTrue((output / "ROI_VALIDATED.flag").is_file())
            self.assertEqual(before, {path: common.sha256_file(path) for path in source.rglob("*") if path.is_file()})
            records = [json.loads((output / "feature_records" / f"{index:04d}.json").read_text()) for index in range(6)]
            self.assertEqual([r["head_ok"] for r in records], [1, 0, 1, 1, 1, 0])
            self.assertEqual([r["beak_ok"] for r in records], [1, 1, 0, 1, 1, 0])
            self.assertEqual([r["foreground_ok"] for r in records], [1, 1, 1, 0, 1, 0])
            self.assertEqual([r["feather_ok"] for r in records], [1, 0, 1, 0, 0, 0])
            self.assertEqual([r["orientation_flip"] for r in records], [1, 0, 0, 1, 0, 0])
            archive = np.load(output / "raw_roi_features.npz")
            self.assertTrue(np.isnan(archive["features"][[2, 5], :9]).all())
            self.assertTrue(np.isnan(archive["features"][[1, 3, 4, 5], 9:]).all())
            for region in ["foreground", "beak", "feather"]:
                for record in records:
                    if not record[region + "_ok"]:
                        self.assertTrue(all(np.isnan(record[region + "_" + suffix]) for suffix in common.MEASUREMENT_SUFFIXES))
            for index in range(6):
                with Image.open(output / "beak_masks" / f"{index:04d}.png") as opened:
                    self.assertGreater(np.asarray(opened).sum(), 0)
            report = json.loads((output / "semantic_validation.json").read_text())
            self.assertEqual(report["column_float32_max_abs"], 0.)
            self.assertEqual(report["color_max_abs_error_threshold"], 1e-6)
            self.assertEqual(report["issues"], [])
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                builder.build(root, source, foreground, data, manifest, mapping)
            # Altering a bound NPZ invalidates the gate before any downstream use.
            (output / "raw_roi_features.npz").write_bytes(b"synthetic tamper")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Bound source artifact changed"):
                validator.verify(root)
            self.assertFalse((output / "ROI_VALIDATED.flag").exists())

    def test_mapping_and_review_values_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="semantic_reviews_") as temporary, six_image_schema():
            root = Path(temporary)
            _, _, _, manifest, mapping = fixture(root)
            reviews = common.read_csv(manifest)
            reviews[0]["head_correct"] = ""
            common.write_csv(manifest, reviews, common.MANIFEST_FIELDS)
            with self.assertRaisesRegex(ValueError, "Unreviewed or invalid"):
                common.read_reviews(manifest, mapping)
            reviews[0]["head_correct"] = "accept"
            reviews[0]["qc_id"], reviews[1]["qc_id"] = reviews[1]["qc_id"], reviews[0]["qc_id"]
            common.write_csv(manifest, reviews, common.MANIFEST_FIELDS)
            with self.assertRaisesRegex(ValueError, "independent anonymous key"):
                common.read_reviews(manifest, mapping)


if __name__ == "__main__":
    unittest.main(verbosity=2)
