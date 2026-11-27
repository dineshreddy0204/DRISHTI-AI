"""Prepare India Driving Dataset (IDD) Detection annotations in YOLO format.

Maps IDD categories cleanly to the DRISHTI AI 17-class canonical taxonomy
without fabricating unsupported classes.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from frigate.drishti.indian_traffic import CANONICAL_INDIAN_CLASSES

CLASS_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_INDIAN_CLASSES)}

IDD_TO_CANONICAL: dict[str, str] = {
    "person": "person",
    "rider": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "scooter": "scooter",
    "autorickshaw": "auto_rickshaw",
    "auto_rickshaw": "auto_rickshaw",
    "e_rickshaw": "e_rickshaw",
    "car": "car",
    "suv": "suv",
    "van": "van",
    "bus": "bus",
    "mini_bus": "mini_bus",
    "truck": "truck",
    "tractor": "tractor",
    "ambulance": "ambulance",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "vehicle fallback": "unknown_vehicle",
    "caravan": "van",
    "trailer": "truck",
}


def convert_voc_xml(xml_path: Path, out_txt: Path) -> tuple[int, int]:
    """Convert one Pascal-VOC style IDD XML annotation to YOLO normalized txt."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return 0, 0
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return 0, 0

    lines: list[str] = []
    skipped = 0
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        canonical = IDD_TO_CANONICAL.get(raw_name)
        if canonical is None:
            skipped += 1
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            skipped += 1
            continue
        xmin = max(0.0, float(bndbox.findtext("xmin", "0")))
        ymin = max(0.0, float(bndbox.findtext("ymin", "0")))
        xmax = min(width, float(bndbox.findtext("xmax", "0")))
        ymax = min(height, float(bndbox.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{CLASS_TO_ID[canonical]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), skipped


def write_data_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "idd_drishti.yaml"
    names_block = "\n".join(
        f"  {idx}: {name}" for idx, name in enumerate(CANONICAL_INDIAN_CLASSES)
    )
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    converted = 0
    total_boxes = 0
    total_skipped = 0
    for xml_file in sorted(args.annotations_dir.rglob("*.xml")):
        rel = xml_file.relative_to(args.annotations_dir).with_suffix(".txt")
        boxes, skipped = convert_voc_xml(xml_file, args.output_dir / "labels" / rel)
        converted += 1
        total_boxes += boxes
        total_skipped += skipped

    yaml_path = write_data_yaml(args.output_dir)
    print(
        json.dumps(
            {
                "converted_files": converted,
                "total_boxes": total_boxes,
                "skipped_annotations": total_skipped,
                "data_yaml": str(yaml_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

