"""Regression checks for rejected media and YOLOX's empty NMS result."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frigate.drishti_video import InvalidVideo, probe
from frigate.util.model import post_process_yolox


class DrishtiVideoTests(unittest.TestCase):
    def test_below_threshold_predictions_produce_no_boxes(self):
        head = np.full((1, 1, 85), 0.1, dtype=np.float32)
        rows = post_process_yolox(
            head, 416, 416, np.zeros((1, 1, 2)), np.ones((1, 1, 1))
        )
        self.assertEqual(float(rows.sum()), 0.0)

    def test_rejects_renamed_non_video_before_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "bad.mp4"
            fake.write_bytes(b"this is not video")
            with self.assertRaises(InvalidVideo):
                probe(fake, "ffprobe")
