"""Indian traffic taxonomy, honest class availability, and uncertain-state mapping."""

from dataclasses import dataclass

CANONICAL_INDIAN_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "motorcycle",
    "scooter",
    "auto_rickshaw",
    "e_rickshaw",
    "car",
    "suv",
    "van",
    "bus",
    "mini_bus",
    "truck",
    "tractor",
    "ambulance",
    "traffic_light",
    "road_obstacle",
    "unknown_vehicle",
)

VEHICLE_CLASSES: frozenset[str] = frozenset(
    {
        "bicycle",
        "motorcycle",
        "scooter",
        "auto_rickshaw",
        "e_rickshaw",
        "car",
        "suv",
        "van",
        "bus",
        "mini_bus",
        "truck",
        "tractor",
        "ambulance",
        "unknown_vehicle",
    }
)

COCO80_TO_INDIAN: dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
}

INDIAN_ONLY_CLASSES: tuple[str, ...] = (
    "scooter",
    "auto_rickshaw",
    "e_rickshaw",
    "suv",
    "van",
    "mini_bus",
    "tractor",
    "ambulance",
    "road_obstacle",
)


@dataclass(frozen=True)
class MappedDetection:
    """Result of mapping a raw detector class to the canonical Indian taxonomy."""

    label: str
    raw_label: str
    is_uncertain: bool
    uncertainty_reason: str | None = None


def map_detection_label(
    raw_label: str,
    score: float,
    supported_labels: set[str] | frozenset[str] | None = None,
    score_threshold: float = 0.35,
    uncertain_floor: float = 0.25,
    class_margin: float | None = None,
    min_class_margin: float = 0.08,
) -> MappedDetection:
    """Map a raw model label to a canonical Indian traffic class.

    Never fabricates unsupported Indian-specific classes (e.g. auto_rickshaw)
    from COCO 'car' predictions. Emits 'unknown_vehicle' when a vehicle
    prediction is below the confident threshold or has ambiguous class margin.
    """
    normalized = raw_label.strip().lower().replace(" ", "_")
    if raw_label in COCO80_TO_INDIAN and (
        supported_labels is None or raw_label not in supported_labels
    ):
        canonical = COCO80_TO_INDIAN[raw_label]
    elif normalized in CANONICAL_INDIAN_CLASSES:
        canonical = normalized
    else:
        canonical = COCO80_TO_INDIAN.get(raw_label, normalized)

    is_vehicle = canonical in VEHICLE_CLASSES
    if score < score_threshold:
        if is_vehicle and score >= uncertain_floor:
            return MappedDetection(
                label="unknown_vehicle",
                raw_label=raw_label,
                is_uncertain=True,
                uncertainty_reason=f"low_confidence({score:.2f}<{score_threshold:.2f})",
            )
        return MappedDetection(
            label=canonical,
            raw_label=raw_label,
            is_uncertain=True,
            uncertainty_reason=f"below_threshold({score:.2f})",
        )

    if (
        is_vehicle
        and class_margin is not None
        and class_margin < min_class_margin
    ):
        return MappedDetection(
            label="unknown_vehicle",
            raw_label=raw_label,
            is_uncertain=True,
            uncertainty_reason=f"ambiguous_margin({class_margin:.2f})",
        )

    return MappedDetection(
        label=canonical,
        raw_label=raw_label,
        is_uncertain=False,
        uncertainty_reason=None,
    )

