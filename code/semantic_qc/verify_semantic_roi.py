#!/usr/bin/env python3
"""Independently validate every image and measurement in semantic ROI v1."""
from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter, laplace

from semantic_common import (CHANNELS, EXPECTED_IMAGES, MEASUREMENT_SUFFIXES, NATIVE_MASK_DIRS,
                             POLICY, REVIEW_FIELDS, VERSION, atomic_json, canonical_hash,
                             check_server_paths, color_converter, load_source_records, read_csv,
                             read_metadata, read_reviews, same_number, sha256_file,
                             verify_assets, verify_original_summary, write_csv)

COLOR_ERROR_THRESHOLD = 1e-6  # Unchanged from the original raw-feature verifier.


def verify(project_root):
    root = Path(project_root).resolve()
    base = root / "code" / VERSION
    check_server_paths([root, base])
    flag = base / "ROI_VALIDATED.flag"
    flag.unlink(missing_ok=True)
    provenance = json.loads((base / "semantic_provenance.json").read_text())
    payload, signature = provenance["payload"], provenance["signature_sha256"]
    if canonical_hash(payload) != signature or payload["version"] != VERSION or payload["n_images"] != EXPECTED_IMAGES:
        raise ValueError("Semantic signature or version mismatch")
    if Path(payload["processing_directory"]).resolve() != base:
        raise ValueError("Semantic provenance belongs to a different output directory")
    roots = payload["source_roots"]
    source = Path(roots["auto"])
    foreground = Path(roots["foreground"])
    data = Path(roots["data"])
    if source.resolve() != root / "code/image_processing_cpu" or Path(roots["project"]).resolve() != root:
        raise ValueError("Semantic provenance does not bind this project's original CPU version")
    check_server_paths([source, foreground, data])
    if canonical_hash(POLICY) != payload["policy_sha256"]:
        raise ValueError("Semantic policy changed after export")
    source_rows = read_csv(base / "source_assets.csv")
    output_rows = read_csv(base / "output_assets.csv")
    for filename, key, rows in [("source_assets.csv", "source_assets", source_rows), ("output_assets.csv", "output_assets", output_rows)]:
        if sha256_file(base / filename) != payload[key + "_file_sha256"]:
            raise ValueError("Bound artifact manifest changed: " + filename)
        # CSV stores byte counts as strings; restore the original canonical representation.
        typed = [{**row, "bytes": int(row["bytes"])} for row in rows]
        if canonical_hash(typed) != payload[key + "_canonical_sha256"]:
            raise ValueError("Artifact manifest canonical digest mismatch: " + filename)
    print("Verifying original source and exported artifact hashes", flush=True)
    verify_assets(source_rows, roots)
    verify_assets(output_rows, {"semantic": base})
    original_provenance = verify_original_summary(source, EXPECTED_IMAGES)
    if original_provenance["records_sha256"] != payload["source_records_sha256"]:
        raise ValueError("Original record digest differs from the semantic signature")
    native_rows = [{**row, "bytes": int(row["bytes"])} for row in source_rows if
                  (row["namespace"] == "foreground" and row["relative_path"].startswith("foreground_masks/"))
                  or (row["namespace"] == "auto" and row["relative_path"].startswith(("beak_masks/", "feather_masks/")))]
    if canonical_hash(native_rows) != payload["source_native_masks_sha256"]:
        raise ValueError("Native candidate mask digest differs from the semantic signature")
    if sha256_file(base / "semantic_qc_manifest.csv") != payload["semantic_qc_manifest_sha256"] or sha256_file(base / "anonymous_mapping.csv") != payload["anonymous_mapping_sha256"]:
        raise ValueError("Semantic review or anonymous mapping digest mismatch")
    if sha256_file(base / "processing_config.json") != payload["processing_config_sha256"]:
        raise ValueError("Semantic processing configuration changed")
    for name, expected_sha in payload["script_sha256"].items():
        if sha256_file(Path(__file__).resolve().parent / name) != expected_sha or sha256_file(base / "pipeline_sources" / name) != expected_sha:
            raise ValueError("Semantic implementation differs from the signed export: " + name)
    converter, color_file_sha, color_function_sha = color_converter(root)
    if color_file_sha != payload["frozen_color_file_sha256"] or color_function_sha != payload["frozen_color_function_ast_sha256"]:
        raise ValueError("Frozen color conversion changed")
    reviews = read_reviews(base / "semantic_qc_manifest.csv", base / "anonymous_mapping.csv")
    metadata = read_metadata(root / "code/data_audit/image_metadata.csv")
    originals = load_source_records(source, metadata)
    csv_rows = read_csv(base / "features.csv")
    if len(csv_rows) != EXPECTED_IMAGES or [int(row["image_index"]) for row in csv_rows] != list(range(EXPECTED_IMAGES)):
        raise ValueError("Semantic feature CSV must contain the complete ordered image set")
    archive = np.load(base / "raw_roi_features.npz", allow_pickle=False)
    if set(archive.files) != {"features", "channels", "image_index", "beak_ok", "feather_ok"}:
        raise ValueError("Unexpected semantic NPZ fields")
    features = archive["features"]
    if features.shape != (EXPECTED_IMAGES, 18) or features.dtype != np.float32 or list(archive["channels"]) != CHANNELS or not np.array_equal(archive["image_index"], np.arange(EXPECTED_IMAGES)):
        raise ValueError("Semantic NPZ shape/dtype/order mismatch")
    for region in ["beak", "feather"]:
        flags = archive[region + "_ok"]
        if flags.shape != (EXPECTED_IMAGES,) or not np.issubdtype(flags.dtype, np.integer) or not np.isin(flags, [0, 1]).all():
            raise ValueError("Semantic NPZ contains invalid QC flags: " + region)
    expected_pngs = {f"{index:04d}.png" for index in range(EXPECTED_IMAGES)}
    for directory in [*NATIVE_MASK_DIRS.values(), "aligned_foreground_masks", "aligned_beak_masks", "aligned_feather_masks", "aligned_rgba"]:
        if {path.name for path in (base / directory).iterdir()} != expected_pngs:
            raise ValueError("Incomplete or extraneous semantic image artifacts: " + directory)
    issues, numerical_rows = [], []
    records = []
    exact_checks = 0
    total_color_checks = 0

    def issue(index, name, **detail):
        issues.append({"image_index": index, "issue": name, **detail})

    def compare(index, name, actual, expected):
        nonlocal exact_checks
        exact_checks += 1
        if not same_number(actual, expected):
            issue(index, name, actual=str(actual), expected=str(expected))

    for index, (original, meta, feature_row) in enumerate(zip(originals, metadata, csv_rows)):
        review = reviews[index]
        record = json.loads((base / "feature_records" / f"{index:04d}.json").read_text())
        records.append(record)
        # Independent rule calculation; this validator does not call the builder's transform.
        expected_flags = {
            "head_ok": int(original["head_ok"] == 1 and review["head_correct"] == "accept"),
            "beak_ok": int(original["beak_ok"] == 1 and review["beak_correct"] == "accept"),
            "foreground_ok": int(original["foreground_ok"] == 1 and review["foreground_usable"] == "accept"),
        }
        expected_flags["feather_ok"] = int(original["feather_ok"] == 1 and review["feather_correct"] == "accept"
                                            and expected_flags["head_ok"] == 1 and expected_flags["foreground_ok"] == 1)
        direction_valid = (review["orientation_usable"] == "accept" and expected_flags["head_ok"] == 1
                           and expected_flags["beak_ok"] == 1 and original["orientation_confidence"] > 0)
        for field, expected in expected_flags.items():
            if type(record.get(field)) is not int or record[field] not in (0, 1):
                issue(index, "invalid_qc_flag", field=field)
            compare(index, "semantic_rule_" + field, record.get(field), expected)
            if int(record.get(field, 0)) > original[field]:
                issue(index, "automatic_rejection_restored", field=field)
        for field, value in original.items():
            compare(index, "auto_preservation_" + field, record.get("auto_" + field), value)
        mutable = {"head_ok", "beak_ok", "foreground_ok", "feather_ok", "orientation_flip", "orientation_confidence",
                   "head_confidence", "beak_confidence", "foreground_confidence", "head_to_beak_dx", "head_to_beak_dy",
                   "roi_qc", "background_gray_mean", "background_gray_std", "original_head_direction",
                   "visible_anatomical_side", "foreground_method", "BCI_rgb_mean", "OCD_foreground_gray_std",
                   "PFD_feather_gray_std", "OCD_feather_gray_std", "OB_foreground_gray_mean"}
        mutable.update(region + "_" + suffix for region in NATIVE_MASK_DIRS for suffix in MEASUREMENT_SUFFIXES)
        for field in set(original) - mutable:
            compare(index, "unmodified_original_" + field, record.get(field), original[field])
        for field in ["image_index", "image_path", "day_group", "age_label", "width", "height"]:
            compare(index, "metadata_" + field, record.get(field), original[field])
        for field in REVIEW_FIELDS:
            compare(index, "review_" + field, record.get("semantic_" + field), review[field])
        compare(index, "review_id", record.get("semantic_qc_id"), review["qc_id"])
        compare(index, "review_reason", record.get("semantic_reason"), review["reason"])
        compare(index, "review_version", record.get("semantic_review_version"), VERSION)
        if type(record.get("orientation_flip")) is not int or record["orientation_flip"] not in (0, 1):
            issue(index, "invalid_orientation_flip_flag")
        compare(index, "orientation_flip", record.get("orientation_flip"), original["orientation_flip"] if direction_valid else 0)
        compare(index, "orientation_confidence", record.get("orientation_confidence"), original["orientation_confidence"] if direction_valid else 0.)
        for field in ["head_to_beak_dx", "head_to_beak_dy"]:
            compare(index, field, record.get(field), original[field] if direction_valid else float("nan"))
        expected_direction = ("left" if original["head_to_beak_dx"] < 0 else "right") if direction_valid else "uncertain"
        compare(index, "original_head_direction", record.get("original_head_direction"), expected_direction)
        compare(index, "visible_anatomical_side", record.get("visible_anatomical_side"), "unknown")
        compare(index, "foreground_method", record.get("foreground_method"), "MobileSAM_manual_point_QC" if index == 229 else "YOLO11n_seg_bird_class14_imgsz640")
        expected_tags = ["semantic_" + field + "_" + review[field] for field in REVIEW_FIELDS if review[field] != "accept"]
        expected_tags.extend(region + "_uncertain" for region in ["foreground", "beak", "feather"] if not expected_flags[region + "_ok"])
        if not expected_flags["head_ok"]:
            expected_tags.append("head_uncertain")
        if not direction_valid:
            expected_tags.append("orientation_preserved_uncertain")
        compare(index, "allowed_qc_tags", record.get("roi_qc"), "|".join(expected_tags) if expected_tags else "pass")
        for region in ["foreground", "head", "beak"]:
            compare(index, region + "_confidence", record.get(region + "_confidence"), original[region + "_confidence"] if expected_flags[region + "_ok"] else 0.)
        if set(feature_row) != set(record):
            issue(index, "CSV_record_columns_mismatch")
        for field, expected in record.items():
            actual = feature_row.get(field)
            if isinstance(expected, (float, int)):
                try:
                    actual = float(actual)
                except (TypeError, ValueError):
                    issue(index, "CSV_numeric_parse_error", field=field)
                    continue
            compare(index, "CSV_record_" + field, actual, expected)
        with Image.open(data / meta["image_path"], formats=["JPEG", "PNG"]) as opened:
            image = np.asarray(opened.convert("RGB"))
        if image.shape != (int(meta["height"]), int(meta["width"]), 3):
            issue(index, "native_RGB_geometry")
        masks = {}
        for region, directory in NATIVE_MASK_DIRS.items():
            native_path = base / directory / f"{index:04d}.png"
            source_path = (foreground if region == "foreground" else source) / directory / f"{index:04d}.png"
            if sha256_file(native_path) != sha256_file(source_path):
                issue(index, "native_candidate_bytes_changed", region=region)
            with Image.open(native_path, formats=["PNG"]) as opened:
                masks[region] = np.asarray(opened) > 0
            if masks[region].shape != image.shape[:2]:
                raise ValueError(f"Native candidate shape mismatch at {index}: {region}")
            compare(index, region + "_pixel_count", record[region + "_pixels"], int(masks[region].sum()))
            for suffix in MEASUREMENT_SUFFIXES:
                field = region + "_" + suffix
                expected = original[field] if expected_flags[region + "_ok"] else float("nan")
                compare(index, "native_measurement_preserved_or_NaN_" + field, record[field], expected)
                if expected_flags[region + "_ok"] and not np.isfinite(record[field]):
                    issue(index, "accepted_measurement_nonfinite", field=field)
                if not expected_flags[region + "_ok"] and not np.isnan(record[field]):
                    issue(index, "disabled_measurement_not_NaN", field=field)
        if (masks["beak"] & masks["feather"]).any():
            issue(index, "native_beak_feather_overlap")
        if (masks["feather"] & ~masks["foreground"]).any():
            issue(index, "native_feather_not_subset_foreground")
        primary = masks["foreground"] | (masks["beak"] if expected_flags["beak_ok"] else False)
        flip = bool(original["orientation_flip"]) if direction_valid else False
        for region, native in [("foreground", primary), ("beak", masks["beak"]), ("feather", masks["feather"])]:
            with Image.open(base / ("aligned_" + region + "_masks") / f"{index:04d}.png", formats=["PNG"]) as opened:
                aligned = np.asarray(opened) > 0
            if not np.array_equal(aligned, np.fliplr(native) if flip else native):
                issue(index, "aligned_mask_mismatch", region=region)
        with Image.open(base / "aligned_rgba" / f"{index:04d}.png", formats=["PNG"]) as opened:
            rgba = np.asarray(opened)
        expected_image = np.fliplr(image) if flip else image
        expected_alpha = np.fliplr(primary) if flip else primary
        if rgba.shape != (*image.shape[:2], 4) or not np.array_equal(rgba[:, :, :3], expected_image):
            issue(index, "native_RGB_values_changed")
        if rgba.shape != (*image.shape[:2], 4) or not np.array_equal(rgba[:, :, 3], expected_alpha.astype(np.uint8) * 255):
            issue(index, "RGBA_alpha_mismatch")
        rgb = image.astype(np.float32) / 255
        gray = (rgb @ np.array([.299, .587, .114], np.float32)) * 255
        highpass = gray - gaussian_filter(gray, sigma=3)
        laplacian = laplace(gray)
        for region, mask in masks.items():
            if not expected_flags[region + "_ok"]:
                continue
            values = converter(rgb[mask])
            if not len(values):
                issue(index, "accepted_empty_native_mask", region=region)
                continue
            maxima32 = {"mean": 0., "std": 0.}
            maxima64 = {"mean": 0., "std": 0.}
            for column, channel in enumerate(CHANNELS):
                column_values = values[:, column]
                for operation in ["mean", "std"]:
                    saved = float(record[region + "_" + channel + "_" + operation])
                    raw32 = float(getattr(column_values, operation)())
                    reference64 = float(getattr(column_values.astype(np.float64), operation)())
                    error32, error64 = abs(saved - raw32), abs(saved - reference64)
                    maxima32[operation] = max(maxima32[operation], error32)
                    maxima64[operation] = max(maxima64[operation], error64)
                    total_color_checks += 1
                    compare(index, "column_float32_exact_" + region + "_" + channel + "_" + operation, saved, raw32)
                    if not np.isfinite(error32) or error32 > COLOR_ERROR_THRESHOLD:
                        issue(index, "native_color_error_above_unchanged_threshold", region=region, channel=channel, operation=operation, max_abs=str(error32))
                    if not np.isfinite(error64) or error64 > COLOR_ERROR_THRESHOLD:
                        issue(index, "native_color_float64_reference_error", region=region, channel=channel, operation=operation, max_abs=str(error64))
            scalar_values = gray[mask]
            scalar_recalculation = {"gray_mean": float(scalar_values.mean()), "gray_std": float(scalar_values.std())}
            core = binary_erosion(mask, iterations=3)
            if core.sum() < 100:
                core = mask
            residuals = highpass[core]
            scalar_recalculation["highpass_mad"] = float(np.median(np.abs(residuals - np.median(residuals))))
            scalar_recalculation["laplacian_variance"] = float(laplacian[core].var())
            hist = np.bincount(np.clip(scalar_values.astype(int), 0, 255), minlength=256)
            probabilities = hist[hist > 0] / hist.sum()
            scalar_recalculation["gray_entropy"] = float(-(probabilities * np.log2(probabilities)).sum())
            for suffix, actual in scalar_recalculation.items():
                compare(index, "native_scalar_float32_exact_" + region + "_" + suffix, record[region + "_" + suffix], actual)
            numerical_rows.append({"image_index": index, "qc_id": review["qc_id"], "region": region,
                                   "pixels": len(values), "mean_column_float32_max_abs": maxima32["mean"],
                                   "std_column_float32_max_abs": maxima32["std"],
                                   "mean_float64_reference_max_abs": maxima64["mean"],
                                   "std_float64_reference_max_abs": maxima64["std"],
                                   "gray_mean_float64_reference_abs": abs(record[region + "_gray_mean"] - float(scalar_values.astype(np.float64).mean())),
                                   "gray_std_float64_reference_abs": abs(record[region + "_gray_std"] - float(scalar_values.astype(np.float64).std()))})
        for region, feature_slice, operation in [("beak", slice(0, 9), "mean"), ("feather", slice(9, 18), "std")]:
            valid = expected_flags[region + "_ok"] == 1
            compare(index, "NPZ_" + region + "_flag", int(archive[region + "_ok"][index]), int(valid))
            expected_vector = np.array([record[region + "_" + channel + "_" + operation] for channel in CHANNELS], np.float32) if valid else np.full(9, np.nan, np.float32)
            if not np.array_equal(features[index, feature_slice], expected_vector, equal_nan=True):
                issue(index, "NPZ_record_vector_mismatch", region=region)
            if valid != bool(np.isfinite(features[index, feature_slice]).all()) or (not valid and not np.isnan(features[index, feature_slice]).all()):
                issue(index, "NPZ_validity_NaN_mismatch", region=region)
        aliases = {"BCI_rgb_mean": float(np.mean([record["beak_" + channel + "_mean"] for channel in CHANNELS[:3]]) * 255) if expected_flags["beak_ok"] else float("nan"),
                   "OCD_foreground_gray_std": record["foreground_gray_std"], "PFD_feather_gray_std": record["feather_gray_std"],
                   "OCD_feather_gray_std": record["feather_gray_std"], "OB_foreground_gray_mean": record["foreground_gray_mean"]}
        for field, expected in aliases.items():
            compare(index, "alias_" + field, record[field], expected)
        background = gray[~primary]
        for operation in ["mean", "std"]:
            compare(index, "background_" + operation, record["background_gray_" + operation], float(getattr(background, operation)()) if background.size else float("nan"))
        if index % 50 == 0 or index + 1 == EXPECTED_IMAGES:
            print(f"Independent semantic native/numerical verification {index + 1}/{EXPECTED_IMAGES}", flush=True)
    summary = json.loads((base / "processing_summary.json").read_text())
    if summary.get("mode") != "independent_semantic_postprocessing" or summary.get("semantic_input_signature_sha256") != signature or not summary.get("source_immutable_recheck_passed"):
        issue(-1, "independent_summary_provenance_mismatch")
    for field in ["foreground_ok", "head_ok", "beak_ok", "feather_ok"]:
        compare(-1, "summary_" + field, summary.get(field), sum(record[field] for record in records))
    compare(-1, "summary_n_processed", summary.get("n_processed"), EXPECTED_IMAGES)
    compare(-1, "summary_orientation_flipped", summary.get("orientation_flipped"), sum(record["orientation_flip"] for record in records))
    compare(-1, "summary_orientation_reliable", summary.get("orientation_reliable"), sum(record["orientation_confidence"] > 0 for record in records))
    print("Rechecking source and exported artifact hashes after full numerical verification", flush=True)
    verify_assets(source_rows, roots)
    verify_assets(output_rows, {"semantic": base})
    if numerical_rows:
        write_csv(base / "numerical_recalculation.csv", numerical_rows)
    report = {"version": VERSION, "status": "passed" if not issues else "failed", "n_images": EXPECTED_IMAGES,
              "validation_host": platform.node(), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "semantic_input_signature_sha256": signature, "semantic_manifest_sha256": payload["semantic_qc_manifest_sha256"],
              "source_immutable_hashes_verified": True, "exported_artifact_hashes_verified": True,
              "full1143_numerical_geometry_pass": not issues, "rgb_preserved_and_coordinate_alignment_verified": not issues,
              "n_exact_field_checks": exact_checks, "n_native_color_statistic_checks": total_color_checks,
              "n_raw_feature_recalculation_checks": len(numerical_rows),
              "color_max_abs_error_threshold": COLOR_ERROR_THRESHOLD, "column_float32_exact_equality_required": True,
              "float64_channel_reference_checked_at_same_threshold": True,
              "gray_float64_reference_role": "Diagnostic only; every saved gray/texture measurement must independently equal the original float32 calculation exactly.",
              "column_float32_max_abs": max((max(row["mean_column_float32_max_abs"], row["std_column_float32_max_abs"]) for row in numerical_rows), default=0.),
              "float64_channel_reference_max_abs": max((max(row["mean_float64_reference_max_abs"], row["std_float64_reference_max_abs"]) for row in numerical_rows), default=0.),
              "n_foreground_ok": sum(record["foreground_ok"] for record in records),
              "n_beak_ok": sum(record["beak_ok"] for record in records), "n_feather_ok": sum(record["feather_ok"] for record in records),
              "n_flipped": sum(record["orientation_flip"] for record in records), "issues": issues}
    atomic_json(base / "semantic_validation.json", report)
    atomic_json(base / "roi_validation.json", report)
    if not issues:
        atomic_json(flag, {"version": VERSION, "semantic_input_signature_sha256": signature,
                           "semantic_validation_sha256": sha256_file(base / "semantic_validation.json"),
                           "features_sha256": sha256_file(base / "features.csv"), "validated_images": EXPECTED_IMAGES,
                           "status": "passed"})
    print(json.dumps(report, indent=2), flush=True)
    return not issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    base = args.project_root.resolve() / "code" / VERSION
    try:
        passed = verify(args.project_root)
    except Exception as error:
        check_server_paths([base])
        if base.is_dir():
            (base / "ROI_VALIDATED.flag").unlink(missing_ok=True)
            failure = {"version": VERSION, "status": "failed", "full1143_numerical_geometry_pass": False,
                       "error_type": type(error).__name__, "error": str(error)}
            atomic_json(base / "semantic_validation.json", failure)
            atomic_json(base / "roi_validation.json", failure)
        raise
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
