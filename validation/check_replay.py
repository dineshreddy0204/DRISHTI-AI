"""Measure the Hyderabad replay's motion calibration and regional Frigate detections."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np

from frigate.camera import PTZMetrics
from frigate.config import FrigateConfig
from frigate.detectors.device import build_detector_config
from frigate.motion.improved_motion import ImprovedMotionDetector
from frigate.object_detection.base import LocalObjectDetector
from frigate.track.norfair_tracker import NorfairTracker
from frigate.util.object import reduce_detections
from frigate.video.detect import detect


def evaluate(config_path: Path) -> dict:
    config = FrigateConfig.parse_yaml(config_path.read_text())
    camera = config.cameras["hyderabad_replay"]
    model = config.model_for_camera(camera.name)
    device = config.devices_for_model(model)[0]
    detector = LocalObjectDetector(
        build_detector_config(device, model), labels=model.labelmap_path
    )
    width, height, fps = camera.detect.width, camera.detect.height, camera.detect.fps
    motion = ImprovedMotionDetector((height, width), camera.motion, fps)
    left = ((width - height) // 2) // 4 * 4
    region = (left, 0, left + height, height)
    tracker = NorfairTracker(camera, PTZMetrics())
    tracks = {}
    counts = Counter()
    frame_name = "drishti-check-" + uuid4().hex
    sampled = eligible = 0
    command = [
        config.ffmpeg.ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        camera.ffmpeg.inputs[0].path,
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
        frame_buffer = tracker.frame_manager.create(frame_name, width * height * 3 // 2)
        try:
            while raw := decoder.stdout.read(width * height * 3 // 2):
                yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
                frame_buffer[:] = raw
                sampled += 1
                boxes = motion.detect(yuv)
                if motion.is_calibrating() or not boxes:
                    tracker.update_frame_times(frame_name, sampled / fps)
                    continue
                eligible += 1
                detections = detect(
                    camera.detect,
                    detector,
                    yuv,
                    model,
                    region,
                    camera.objects.track,
                    camera.objects.filters,
                )
                counts.update(item[0] for item in detections)
                tracker.match_and_update(
                    frame_name,
                    sampled / fps,
                    reduce_detections((height, width), detections),
                )
                tracks.update(
                    {key: obj["label"] for key, obj in tracker.tracked_objects.items()}
                )
        finally:
            del frame_buffer
            tracker.frame_manager.delete(frame_name)
            decoder.stdout.close()
        if decoder.wait():
            raise RuntimeError("FFmpeg could not finish decoding the replay")
    return {
        "camera": camera.name,
        "motion_threshold": camera.motion.threshold,
        "sampled_frames": sampled,
        "calibrated_frames_with_motion": eligible,
        "test_region_xyxy": region,
        "filtered_detector_boxes": dict(counts),
        "tracker_ids_by_label": dict(Counter(tracks.values())),
        "counting_unit": "box per sampled frame in the fixed test region",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/opt/drishti/config.yml"))
    print(json.dumps(evaluate(parser.parse_args().config), indent=2))
