#!/usr/bin/env python3
"""Build a new semantic ROI version only after all 1143 anonymous reviews exist."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_common import (CHANNELS, EXPECTED_IMAGES, MEASUREMENT_SUFFIXES, NATIVE_MASK_DIRS,
                             POLICY, REVIEW_FIELDS, VERSION, asset_row, atomic_json, canonical_hash,
                             check_server_paths, color_converter, load_source_records, read_metadata,
                             read_reviews, sha256_file, source_assets, verify_assets, verify_original_summary, write_csv)


def semantic_record(original, review):
    """Apply review-only gating; measurements of accepted candidates are never adjusted."""
    record = dict(original)
    record.update({"auto_" + field: value for field, value in original.items()})
    accepted = {field: review[field] == "accept" for field in REVIEW_FIELDS}
    record["head_ok"] = int(bool(original["head_ok"]) and accepted["head_correct"])
    record["beak_ok"] = int(bool(original["beak_ok"]) and accepted["beak_correct"])
    record["foreground_ok"] = int(bool(original["foreground_ok"]) and accepted["foreground_usable"])
    record["feather_ok"] = int(bool(original["feather_ok"]) and accepted["feather_correct"]
                               and bool(record["head_ok"]) and bool(record["foreground_ok"]))
    orientation = (accepted["orientation_usable"] and bool(record["head_ok"])
                   and bool(record["beak_ok"]) and original["orientation_confidence"] > 0)
    record["orientation_flip"] = int(original["orientation_flip"]) if orientation else 0
    record["orientation_confidence"] = original["orientation_confidence"] if orientation else 0.0
    record["head_to_beak_dx"] = original["head_to_beak_dx"] if orientation else float("nan")
    record["head_to_beak_dy"] = original["head_to_beak_dy"] if orientation else float("nan")
    record["original_head_direction"] = ("left" if original["head_to_beak_dx"] < 0 else "right") if orientation else "uncertain"
    record["visible_anatomical_side"] = "unknown"
    for region in ["foreground", "head", "beak"]:
        if not record[region + "_ok"]:
            record[region + "_confidence"] = 0.0
    for region in NATIVE_MASK_DIRS:
        if not record[region + "_ok"]:
            for suffix in MEASUREMENT_SUFFIXES:
                record[region + "_" + suffix] = float("nan")
    record["BCI_rgb_mean"] = float(np.mean([record["beak_" + c + "_mean"] for c in CHANNELS[:3]]) * 255) if record["beak_ok"] else float("nan")
    record["OCD_foreground_gray_std"] = record["foreground_gray_std"]
    record["PFD_feather_gray_std"] = record["feather_gray_std"]
    record["OCD_feather_gray_std"] = record["feather_gray_std"]
    record["OB_foreground_gray_mean"] = record["foreground_gray_mean"]
    record["foreground_method"] = "MobileSAM_manual_point_QC" if original["image_index"] == 229 else "YOLO11n_seg_bird_class14_imgsz640"
    record["semantic_qc_id"] = review["qc_id"]
    record["semantic_review_version"] = VERSION
    for field in REVIEW_FIELDS:
        record["semantic_" + field] = review[field]
    record["semantic_reason"] = review["reason"]
    tags = ["semantic_" + field + "_" + review[field] for field in REVIEW_FIELDS if review[field] != "accept"]
    tags.extend(region + "_uncertain" for region in ["foreground", "beak", "feather"] if not record[region + "_ok"])
    if not record["head_ok"]:
        tags.append("head_uncertain")
    if not orientation:
        tags.append("orientation_preserved_uncertain")
    record["roi_qc"] = "|".join(tags) if tags else "pass"
    return record


def build(project_root, source, foreground, data, manifest, mapping):
    project_root, source, foreground, data, manifest, mapping = [Path(p).resolve() for p in [project_root, source, foreground, data, manifest, mapping]]
    output = project_root / "code" / VERSION
    check_server_paths([project_root, source, foreground, data, manifest, mapping, output])
    if source != project_root / "code/image_processing_cpu":
        raise ValueError("The semantic source must be the original work/image_processing_cpu directory")
    if source == foreground or output in [source, foreground] or output.exists():
        raise ValueError("Semantic output must be new and separate from all source directories; existing output is never overwritten")
    reviews = read_reviews(manifest, mapping)  # No output is created before complete review acceptance.
    metadata_path = project_root / "code/data_audit/image_metadata.csv"
    inventory_path = project_root / "code/data_audit/inventory.csv"
    metadata = read_metadata(metadata_path)
    originals = load_source_records(source, metadata)
    original_provenance = verify_original_summary(source, EXPECTED_IMAGES)
    source_config = json.loads((source / "processing_config.json").read_text())
    if source_config.get("model_device") != "cpu":
        raise ValueError("Semantic postprocessing requires the uniform CPU source version")
    started = time.time()
    print("Hashing the immutable original CPU inputs", flush=True)
    bound_assets = source_assets(source, foreground, data, project_root, metadata, inventory_path)
    _, color_file_sha, color_function_sha = color_converter(project_root)
    code_directory = Path(__file__).resolve().parent
    script_shas = {name: sha256_file(code_directory / name) for name in
                   ["build_semantic_roi.py", "semantic_common.py", "verify_semantic_roi.py"]}
    manifest_sha, mapping_sha = sha256_file(manifest), sha256_file(mapping)
    staging = output.with_name("." + VERSION + ".staging_" + str(os.getpid()))
    staging.mkdir(exist_ok=False)
    for directory in [*NATIVE_MASK_DIRS.values(), "aligned_foreground_masks", "aligned_beak_masks",
                      "aligned_feather_masks", "aligned_rgba", "feature_records", "detections", "pipeline_sources"]:
        (staging / directory).mkdir()
    shutil.copyfile(manifest, staging / "semantic_qc_manifest.csv")
    shutil.copyfile(mapping, staging / "anonymous_mapping.csv")
    shutil.copyfile(source / "processing_config.json", staging / "source_auto_processing_config.json")
    shutil.copyfile(source / "processing_summary.json", staging / "source_auto_processing_summary.json")
    for name in script_shas:
        shutil.copyfile(code_directory / name, staging / "pipeline_sources" / name)
    write_csv(staging / "source_assets.csv", bound_assets)
    records = []
    for index, (original, meta) in enumerate(zip(originals, metadata)):
        record = semantic_record(original, reviews[index])
        with Image.open(data / meta["image_path"], formats=["JPEG", "PNG"]) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        if rgb.shape != (int(meta["height"]), int(meta["width"]), 3):
            raise ValueError("Original native geometry mismatch at " + str(index))
        masks = {}
        for region, directory in NATIVE_MASK_DIRS.items():
            source_path = (foreground if region == "foreground" else source) / directory / f"{index:04d}.png"
            target_path = staging / directory / f"{index:04d}.png"
            shutil.copyfile(source_path, target_path)
            with Image.open(source_path, formats=["PNG"]) as opened:
                masks[region] = np.asarray(opened) > 0
            if masks[region].shape != rgb.shape[:2] or int(masks[region].sum()) != int(record[region + "_pixels"]):
                raise ValueError(f"Original candidate geometry/pixel count mismatch: {index} {region}")
        primary = masks["foreground"] | (masks["beak"] if record["beak_ok"] else False)
        flip = bool(record["orientation_flip"])
        for region, mask in [("foreground", primary), ("beak", masks["beak"]), ("feather", masks["feather"])]:
            aligned = np.fliplr(mask) if flip else mask
            Image.fromarray(aligned.astype(np.uint8) * 255).save(staging / ("aligned_" + region + "_masks") / f"{index:04d}.png", compress_level=1)
        aligned_rgb = np.fliplr(rgb) if flip else rgb
        aligned_alpha = np.fliplr(primary) if flip else primary
        rgba = np.dstack([aligned_rgb, aligned_alpha.astype(np.uint8) * 255])
        Image.fromarray(rgba).save(staging / "aligned_rgba" / f"{index:04d}.png", compress_level=1)
        gray = ((rgb.astype(np.float32) / 255) @ np.array([.299, .587, .114], np.float32)) * 255
        background = gray[~primary]
        record["background_gray_mean"] = float(background.mean()) if background.size else float("nan")
        record["background_gray_std"] = float(background.std()) if background.size else float("nan")
        shutil.copyfile(source / "detections" / f"{index:04d}.json", staging / "detections" / f"{index:04d}.json")
        atomic_json(staging / "feature_records" / f"{index:04d}.json", record, allow_nan=True)
        records.append(record)
        if index % 50 == 0 or index + 1 == EXPECTED_IMAGES:
            print(f"Semantic native/aligned exports {index + 1}/{EXPECTED_IMAGES}", flush=True)
    write_csv(staging / "features.csv", records)
    features = np.full((EXPECTED_IMAGES, 18), np.nan, np.float32)
    for index, record in enumerate(records):
        if record["beak_ok"]:
            features[index, :9] = [record["beak_" + channel + "_mean"] for channel in CHANNELS]
        if record["feather_ok"]:
            features[index, 9:] = [record["feather_" + channel + "_std"] for channel in CHANNELS]
    np.savez(staging / "raw_roi_features.npz", features=features, channels=CHANNELS,
             image_index=np.arange(EXPECTED_IMAGES), beak_ok=np.array([r["beak_ok"] for r in records]),
             feather_ok=np.array([r["feather_ok"] for r in records]))
    config = dict(source_config)
    config.update({"processing_directory": str(output), "source_processing_directory": str(foreground),
                   "semantic_version": VERSION, "semantic_source_processing_directory": str(source),
                   "semantic_qc_manifest_sha256": manifest_sha, "anonymous_mapping_sha256": mapping_sha,
                   "semantic_policy": POLICY, "semantic_script_sha256": script_shas,
                   "semantic_source_assets_sha256": sha256_file(staging / "source_assets.csv"),
                   "semantic_validation_required_before_training": True})
    atomic_json(staging / "processing_config.json", config)
    print("Rechecking immutable source hashes before committing the new version", flush=True)
    roots = {"auto": str(source), "foreground": str(foreground), "data": str(data), "project": str(project_root)}
    verify_assets(bound_assets, roots)
    if sha256_file(manifest) != manifest_sha or sha256_file(mapping) != mapping_sha:
        raise ValueError("Review manifest or anonymous mapping changed during semantic export")
    output_assets = [asset_row("semantic", path.relative_to(staging), path)
                     for path in sorted(staging.rglob("*")) if path.is_file()]
    write_csv(staging / "output_assets.csv", output_assets)
    payload = {"version": VERSION, "n_images": EXPECTED_IMAGES, "source_roots": roots,
               "processing_directory": str(output), "semantic_qc_manifest_sha256": manifest_sha,
               "anonymous_mapping_sha256": mapping_sha, "policy_sha256": canonical_hash(POLICY),
               "source_assets_file_sha256": sha256_file(staging / "source_assets.csv"),
               "source_assets_canonical_sha256": canonical_hash(bound_assets),
               "source_records_sha256": original_provenance["records_sha256"],
               "source_native_masks_sha256": canonical_hash([row for row in bound_assets if
                   (row["namespace"] == "foreground" and row["relative_path"].startswith("foreground_masks/"))
                   or (row["namespace"] == "auto" and row["relative_path"].startswith(("beak_masks/", "feather_masks/")))]),
               "output_assets_file_sha256": sha256_file(staging / "output_assets.csv"),
               "output_assets_canonical_sha256": canonical_hash(output_assets),
               "processing_config_sha256": sha256_file(staging / "processing_config.json"),
               "script_sha256": script_shas, "frozen_color_file_sha256": color_file_sha,
               "frozen_color_function_ast_sha256": color_function_sha}
    signature = canonical_hash(payload)
    atomic_json(staging / "semantic_provenance.json", {"signature_sha256": signature, "payload": payload})
    summary = {"version": VERSION, "status": "generated_pending_independent_validation", "n_processed": len(records),
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": platform.node(),
               "elapsed_seconds": time.time() - started, "mode": "independent_semantic_postprocessing", "device": "cpu",
               "source_processing_directory": str(source), "source_immutable_recheck_passed": True,
               "semantic_input_signature_sha256": signature, "qc_manifest_sha256": manifest_sha,
               "anonymous_mapping_sha256": mapping_sha, "source_script_sha256": script_shas["build_semantic_roi.py"],
               "semantic_script_sha256": script_shas,
               "source_records_sha256": payload["source_records_sha256"],
               "source_native_masks_sha256": payload["source_native_masks_sha256"],
               "records_sha256": canonical_hash([r for r in output_assets if r["relative_path"].startswith("feature_records/")]),
               "foreground_ok": sum(r["foreground_ok"] for r in records), "head_ok": sum(r["head_ok"] for r in records),
               "beak_ok": sum(r["beak_ok"] for r in records), "feather_ok": sum(r["feather_ok"] for r in records),
               "orientation_flipped": sum(r["orientation_flip"] for r in records),
               "orientation_reliable": sum(r["orientation_confidence"] > 0 for r in records),
               "semantic_review_counts": {field: {value: sum(row[field] == value for row in reviews.values())
                                                    for value in ["accept", "uncertain", "reject"]} for field in REVIEW_FIELDS}}
    atomic_json(staging / "processing_summary.json", summary)
    if output.exists():
        raise ValueError("Semantic output appeared during processing; refusing to replace it")
    staging.rename(output)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--auto-source", type=Path)
    parser.add_argument("--foreground-source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-map", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = args.data or os.environ.get("EAGLE_DATA_DIR")
    if not data:
        parser.error("--data or EAGLE_DATA_DIR is required")
    build(root, args.auto_source or root / "code/image_processing_cpu",
          args.foreground_source or root / "code/image_processing", data, args.manifest, args.qc_map)


if __name__ == "__main__":
    main()
