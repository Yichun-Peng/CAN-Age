"""Shared file/schema helpers for the separately versioned semantic ROI pass."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

EXPECTED_IMAGES = 1143
VERSION = "image_processing_cpu_semantic_v1"
CHANNELS = ["red", "green", "blue", "hue", "saturation", "value", "lab_l", "lab_a", "lab_b"]
REVIEW_FIELDS = ["head_correct", "beak_correct", "feather_correct", "foreground_usable", "orientation_usable"]
MANIFEST_FIELDS = ["qc_id", "image_index", *REVIEW_FIELDS, "reason"]
REVIEW_VALUES = {"accept", "uncertain", "reject"}
MEASUREMENT_SUFFIXES = [*(channel + "_" + operation for channel in CHANNELS for operation in ["mean", "std"]),
                        "gray_mean", "gray_std", "highpass_mad", "laplacian_variance", "gray_entropy"]
NATIVE_MASK_DIRS = {"foreground": "foreground_masks", "beak": "beak_masks", "feather": "feather_masks"}
AUTO_IMAGE_DIRS = ["beak_masks", "feather_masks", "aligned_beak_masks", "aligned_feather_masks",
                   "aligned_foreground_masks", "aligned_rgba"]
POLICY = {
    "review_values": sorted(REVIEW_VALUES),
    "acceptance": "Each semantic flag can only retain an already valid automatic flag; uncertain and reject disable it.",
    "head_dependency": "Nonaccepted or automatically invalid head disables head, feather, and orientation; it does not restore or remove a separately reviewed beak.",
    "beak_dependency": "Nonaccepted or automatically invalid beak disables beak and orientation.",
    "foreground_dependency": "Nonaccepted or automatically invalid foreground disables foreground and feather.",
    "feather_dependency": "Feather requires automatic feather validity and accepted valid head, foreground, and feather.",
    "orientation_dependency": "Retain the automatic flip and confidence only for accepted orientation with valid accepted head and beak and positive automatic orientation confidence; otherwise flip=0, confidence=0, direction=uncertain.",
    "measurements": "Retain accepted original native measurements exactly; disabled region color, gray, and texture measurements become NaN. Pixel counts and candidate geometry remain unchanged.",
    "candidates": "Copy all original native candidate masks byte for byte, including disabled candidates.",
    "alignment": "Native foreground union final valid beak, reflected only by the final orientation flip. Original decoded RGB values are preserved exactly.",
    "background": "Recompute background gray mean/std from native RGB outside the final foreground union accepted beak.",
    "labels": "Age and day metadata are copied after review linkage; they never enter semantic decisions.",
    "aggregation": "Independent postprocessing provenance; no original inference-worker summary is reused as a new aggregation summary.",
    "numerical_verification": "All valid native ROI channels use the original column-wise float32 mean/std; maximum error threshold stays 1e-6 and exact equality is also required. Float64 reductions provide a separate reference.",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def original_record_digest(source, n_images=EXPECTED_IMAGES):
    """Use the original worker's filename-plus-bytes digest without importing it."""
    digest = hashlib.sha256()
    for index in range(n_images):
        path = Path(source) / "feature_records" / f"{index:04d}.json"
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_original_summary(source, n_images):
    """Authenticate the source aggregation; do not use it as the semantic summary."""
    source = Path(source)
    summary = json.loads((source / "processing_summary.json").read_text())
    config = json.loads((source / "processing_config.json").read_text())
    legacy_config_digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected = {"n_processed": n_images, "device": "cpu", "records_sha256": original_record_digest(source, n_images),
                "source_script_sha256": sha256_file(source / "roi_pipeline.py"), "config_sha256": legacy_config_digest}
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError("Original CPU aggregation provenance mismatch: " + field)
    return expected


def atomic_json(path, value, *, allow_nan=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=allow_nan)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_metadata(path):
    rows = read_csv(path)
    if len(rows) != EXPECTED_IMAGES or [int(row["image_index"]) for row in rows] != list(range(EXPECTED_IMAGES)):
        raise ValueError("Metadata must contain the ordered complete 1143-image set")
    return rows


def read_reviews(manifest, mapping):
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError("Semantic manifest columns/order must be exactly " + ",".join(MANIFEST_FIELDS))
        reviews = list(reader)
    map_rows = read_csv(mapping)
    links = {}
    for row in map_rows:
        qc_id = row.get("qc_id", row.get("anonymous_id", ""))
        index = int(row["image_index"])
        if qc_id in links:
            raise ValueError("Duplicate anonymous mapping ID: " + qc_id)
        links[qc_id] = index
    expected_ids = {f"QC{i:04d}" for i in range(1, EXPECTED_IMAGES + 1)}
    if set(links) != expected_ids or set(links.values()) != set(range(EXPECTED_IMAGES)):
        raise ValueError("Anonymous mapping must be a bijection over QC0001..QC1143 and indices 0..1142")
    if len(reviews) != EXPECTED_IMAGES:
        raise ValueError(f"All 1143 reviews are required before processing; found {len(reviews)}")
    indexed = {}
    seen_ids = set()
    for row in reviews:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed semantic manifest row")
        qc_id = row["qc_id"]
        if not re.fullmatch(r"QC\d{4}", qc_id) or qc_id in seen_ids:
            raise ValueError("Invalid or duplicate semantic qc_id: " + qc_id)
        index = int(row["image_index"])
        if index in indexed or links.get(qc_id) != index:
            raise ValueError("Semantic image_index does not match the independent anonymous key: " + qc_id)
        for field in REVIEW_FIELDS:
            if row[field] not in REVIEW_VALUES:
                raise ValueError(f"Unreviewed or invalid {field} for {qc_id}: {row[field]!r}")
        indexed[index] = row
        seen_ids.add(qc_id)
    if set(indexed) != set(range(EXPECTED_IMAGES)) or seen_ids != expected_ids:
        raise ValueError("Semantic manifest does not cover the complete anonymous image set")
    return indexed


def load_source_records(source, metadata):
    paths = sorted((Path(source) / "feature_records").glob("*.json"))
    if len(paths) != EXPECTED_IMAGES:
        raise ValueError("Original CPU record set must contain exactly 1143 records")
    records = []
    for index, (path, meta) in enumerate(zip(paths, metadata)):
        record = json.loads(path.read_text())
        if path.name != f"{index:04d}.json" or record.get("image_index") != index:
            raise ValueError("Source record filename/index mismatch: " + str(path))
        for field in ["image_path", "day_group"]:
            if record.get(field) != meta[field]:
                raise ValueError(f"Source metadata mismatch at {index}: {field}")
        for field in ["age_label", "width", "height"]:
            if record.get(field) != int(meta[field]):
                raise ValueError(f"Source metadata mismatch at {index}: {field}")
        for field in ["foreground_ok", "head_ok", "beak_ok", "feather_ok", "orientation_flip"]:
            if type(record.get(field)) is not int or record[field] not in (0, 1):
                raise ValueError(f"Invalid original automatic flag at {index}: {field}")
        if not np.isfinite(record["orientation_confidence"]) or record["orientation_confidence"] < 0:
            raise ValueError("Invalid automatic orientation confidence")
        for region in NATIVE_MASK_DIRS:
            required = [region + "_pixels", *(region + "_" + suffix for suffix in MEASUREMENT_SUFFIXES)]
            if any(field not in record for field in required):
                raise ValueError(f"Incomplete original {region} measurements at {index}")
        if any(field.startswith("auto_") or field.startswith("semantic_") for field in record):
            raise ValueError("Source must be the original CPU version, not another semantic derivative")
        records.append(record)
    return records


def color_converter(project_root):
    """Execute only the unchanged numerical function; importing models is unnecessary."""
    path = Path(project_root) / "code/experiments/run_experiments.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rgb_to_color9"]
    if len(functions) != 1:
        raise ValueError("Expected one frozen rgb_to_color9 function")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["rgb_to_color9"], sha256_file(path), hashlib.sha256(ast.dump(functions[0]).encode()).hexdigest()


def asset_row(namespace, relative, path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing or empty source artifact: " + str(path))
    return {"namespace": namespace, "relative_path": str(relative), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def source_assets(source, foreground, data, project_root, metadata, inventory_path):
    source, foreground, data, project_root = map(Path, [source, foreground, data, project_root])
    rows = []
    for index, meta in enumerate(metadata):
        for directory in ["feature_records", "detections", *AUTO_IMAGE_DIRS]:
            extension = ".json" if directory in {"feature_records", "detections"} else ".png"
            relative = Path(directory) / f"{index:04d}{extension}"
            rows.append(asset_row("auto", relative, source / relative))
        relative = Path("foreground_masks") / f"{index:04d}.png"
        rows.append(asset_row("foreground", relative, foreground / relative))
        relative = Path(meta["image_path"])
        if relative.is_absolute() or not (data / relative).resolve().is_relative_to(data.resolve()):
            raise ValueError("Source photo path escapes data directory")
        rows.append(asset_row("data", relative, data / relative))
    for name in ["processing_config.json", "processing_summary.json", "features.csv", "raw_roi_features.npz",
                 "roi_pipeline.py", "roi_job_control.py", "verify_roi_outputs.py"]:
        rows.append(asset_row("auto", name, source / name))
    for path in sorted((source / "worker_jobs").rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            rows.append(asset_row("auto", path.relative_to(source), path))
    rows.append(asset_row("foreground", "foreground_manifest.csv", foreground / "foreground_manifest.csv"))
    for relative in ["code/data_audit/image_metadata.csv", "code/data_audit/inventory.csv", "code/experiments/run_experiments.py"]:
        rows.append(asset_row("project", relative, project_root / relative))
    inventory = {row["image_path"]: row["sha256"] for row in read_csv(inventory_path)}
    for row in rows:
        if row["namespace"] == "data" and inventory.get(row["relative_path"]) != row["sha256"]:
            raise ValueError("Original photo SHA differs from frozen inventory: " + row["relative_path"])
    return rows


def verify_assets(rows, roots):
    for row in rows:
        path = Path(roots[row["namespace"]]) / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError("Bound source artifact changed: " + str(path))


def check_server_paths(paths):
    if os.environ.get("BWHD_ROOT"):
        allowed = Path(os.environ["BWHD_ROOT"]).resolve()
        for path in paths:
            if not Path(path).resolve().is_relative_to(allowed):
                raise ValueError("Server path outside BWHD_ROOT: " + str(path))


def same_number(left, right):
    try:
        return bool(left == right or (np.isnan(float(left)) and np.isnan(float(right))))
    except (TypeError, ValueError):
        return left == right
