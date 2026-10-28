"""Reproducible, annotated-image accuracy and CPU latency comparisons."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from detectors import Detection, add_detector_arguments, create_detector, experiment_metadata


def intersection_over_union(a, b):
    overlap_width = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    overlap_height = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    intersection = overlap_width * overlap_height
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(predictions, ground_truth, threshold=0.5):
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    for prediction in sorted(predictions, key=lambda box: box.confidence, reverse=True):
        candidate = max(unmatched, key=lambda i: intersection_over_union(prediction, ground_truth[i]),
                        default=None)
        if candidate is not None and intersection_over_union(prediction, ground_truth[candidate]) >= threshold:
            true_positives += 1
            unmatched.remove(candidate)
    return true_positives, len(predictions) - true_positives, len(ground_truth) - true_positives


def summarize(samples, counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mean_ms = float(np.mean(samples))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_ms": mean_ms, "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "samples": len(samples),
            "inference_fps": 1000 / mean_ms if mean_ms > 0 else 0.0}


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            not isinstance(manifest.get("images"), list) or not manifest["images"]):
        raise ValueError("标注清单需要 schema_version=1 和非空 images 列表")
    records = []
    for record in manifest["images"]:
        if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) and record[key].strip()
                for key in ("file", "game", "style")):
            raise ValueError("每张图必须包含非空 file、game、style 字符串")
        if not isinstance(record.get("boxes"), list):
            raise ValueError("每张图必须提供 boxes 列表，负样本使用空列表")
        boxes = []
        for box in record["boxes"]:
            if (not isinstance(box, list) or len(box) != 4 or
                    not all(isinstance(value, (int, float)) for value in box) or
                    not np.isfinite(box).all() or min(box[:2]) < 0 or min(box[2:]) <= 0):
                raise ValueError("标注框必须是有效像素坐标 [x, y, width, height]")
            boxes.append(Detection(*box))
        filename = path.parent / record["file"]
        if not filename.is_file():
            raise ValueError(f"找不到数据集图像：{filename}")
        records.append({**record, "path": filename, "truth": boxes})
    return records


def evaluate(detector, records, repeat=5, warmup=3, iou=0.5):
    groups, rows, image_hashes = {}, [], {}
    all_times, all_counts = [], np.zeros(3, dtype=int)
    for index, record in enumerate(records):
        data = record["path"].read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像：{record['path']}")
        if any(box.x + box.width > image.shape[1] or box.y + box.height > image.shape[0]
               for box in record["truth"]):
            raise ValueError(f"标注框超出图像边界：{record['path']}")
        image_hashes[record["file"]] = hashlib.sha256(data).hexdigest()
        if index == 0:
            for _ in range(warmup):
                detector.detect(image)
        times = []
        for sample in range(repeat):
            started = time.perf_counter()
            predictions = detector.detect(image)
            times.append((time.perf_counter() - started) * 1000)
            if sample == 0:
                counts = match_predictions(predictions, record["truth"], iou)
        key = (record["game"], record["style"])
        group = groups.setdefault(key, {"times": [], "counts": np.zeros(3, dtype=int), "images": 0})
        group["times"].extend(times)
        group["counts"] += counts
        group["images"] += 1
        all_times.extend(times)
        all_counts += counts
        rows.append({"file": record["file"], "game": record["game"], "style": record["style"],
                     **summarize(times, counts)})
    summary = {"overall": {"images": len(records), **summarize(all_times, all_counts.tolist())},
               "groups": [{"game": game, "style": style, "images": group["images"],
                           **summarize(group["times"], group["counts"].tolist())}
                          for (game, style), group in sorted(groups.items())],
               "image_sha256": image_hashes}
    return rows, summary


def build_parser():
    parser = argparse.ArgumentParser(description="按游戏和画风评测人头检测准确率及推理延迟")
    add_detector_arguments(parser)
    parser.add_argument("--manifest", required=True, help="JSON 图像标注清单")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", help="报告路径前缀，生成 .csv 和 .json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.repeat <= 0 or args.warmup < 0 or not 0 < args.iou <= 1:
            raise ValueError("repeat 必须为正数，warmup 不能为负数，IoU 必须在 (0, 1] 内")
        records = load_manifest(args.manifest)
        detector = create_detector(args)
        rows, summary = evaluate(detector, records, args.repeat, args.warmup, args.iou)
        summary.update(experiment_metadata(args))
        summary["created_utc"] = datetime.now(timezone.utc).isoformat()
        summary["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
        prefix = Path(args.output or datetime.now().strftime("results/benchmark-%Y%m%d-%H%M%S-%f"))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json")
        if csv_path.exists() or json_path.exists():
            raise ValueError("报告已存在，请换一个 --output，避免覆盖实验记录")
        with csv_path.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("x", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
        print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
        print(f"报告：{csv_path} 和 {json_path}")
    except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
