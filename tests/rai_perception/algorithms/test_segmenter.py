# Copyright (C) 2025 Julia Jia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for GDSegmenter algorithm."""

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import rclpy
from rai_perception.algorithms.segmenter import GDSegmenter
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D


class TestGDSegmenter:
    """Test cases for GDSegmenter class."""

    def setup_method(self):
        """Initialize ROS2 before tests that use ROS2 messages."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                time.sleep(0.1)
                rclpy.shutdown()
        except Exception:
            pass

    @contextmanager
    def _patch_segmenter_dependencies(self):
        """Context manager to patch all GDSegmenter dependencies."""
        with (
            patch("rai_perception.algorithms.segmenter.build_sam2") as mock_build,
            patch(
                "rai_perception.algorithms.segmenter.SAM2ImagePredictor"
            ) as mock_predictor_class,
        ):
            mock_model = MagicMock()
            mock_build.return_value = mock_model
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor
            yield mock_build, mock_predictor

    def _create_test_weights_file(self, tmp_path):
        """Helper to create a test weights file."""
        weights_path = tmp_path / "weights.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"test")
        return weights_path

    def _create_test_bbox(self, center_x, center_y, size_x, size_y):
        """Helper to create a test bounding box."""
        bbox = BoundingBox2D()
        bbox.center.position.x = center_x
        bbox.center.position.y = center_y
        bbox.size_x = size_x
        bbox.size_y = size_y
        return bbox

    def test_gdsegmenter_initialization(self, tmp_path):
        """Test GDSegmenter initialization with default config."""
        weights_path = self._create_test_weights_file(tmp_path)

        with self._patch_segmenter_dependencies() as (mock_build, _):
            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            assert segmenter.weight_path == str(weights_path)
            assert segmenter.device == "cpu"
            assert hasattr(segmenter, "sam2_model")
            assert hasattr(segmenter, "sam2_predictor")
            mock_build.assert_called_once()

    def test_gdsegmenter_initialization_with_config_path(self, tmp_path):
        """Test GDSegmenter initialization with config_path (ignored for SAM2)."""
        weights_path = self._create_test_weights_file(tmp_path)
        config_path = tmp_path / "custom_config.yml"
        config_path.write_text("test: config")

        with self._patch_segmenter_dependencies() as (mock_build, _):
            GDSegmenter(str(weights_path), config_path=str(config_path), use_cuda=False)

            # SAM2 uses Hydra config module, so config_path is ignored
            # build_sam2 is called with the default config name
            mock_build.assert_called_once_with(
                "seg_config.yml", str(weights_path), device="cpu"
            )

    def test_gdsegmenter_get_segmentation(self, tmp_path):
        """Test GDSegmenter get_segmentation method."""
        weights_path = self._create_test_weights_file(tmp_path)

        with (
            self._patch_segmenter_dependencies() as (_, mock_predictor),
            patch(
                "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray"
            ) as mock_convert,
        ):
            # Mock image conversion
            mock_img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_convert.return_value = mock_img_array

            # Mock segmentation predictions
            mask1 = np.zeros((100, 100), dtype=np.float32)
            mask1[10:50, 10:50] = 1.0
            mask2 = np.zeros((100, 100), dtype=np.float32)
            mask2[60:90, 60:90] = 1.0
            mock_predictor.predict.side_effect = [
                (mask1, None, None),
                (mask2, None, None),
            ]

            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            # Create test inputs
            image_msg = Image()
            bbox1 = self._create_test_bbox(30.0, 30.0, 40.0, 40.0)
            bbox2 = self._create_test_bbox(75.0, 75.0, 30.0, 30.0)

            masks = segmenter.get_segmentation(image_msg, [bbox1, bbox2])

            assert len(masks) == 2
            assert isinstance(masks[0], np.ndarray)
            assert isinstance(masks[1], np.ndarray)
            mock_predictor.set_image.assert_called_once_with(mock_img_array)
            assert mock_predictor.predict.call_count == 2

    def test_gdsegmenter_get_segmentation_empty_bboxes(self, tmp_path):
        """Test GDSegmenter get_segmentation with empty bbox list."""
        weights_path = self._create_test_weights_file(tmp_path)

        with (
            self._patch_segmenter_dependencies() as (_, mock_predictor),
            patch(
                "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray"
            ) as mock_convert,
        ):
            mock_img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_convert.return_value = mock_img_array

            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            image_msg = Image()
            masks = segmenter.get_segmentation(image_msg, [])

            assert len(masks) == 0
            mock_predictor.set_image.assert_called_once()
            mock_predictor.predict.assert_not_called()

    def test_gdsegmenter_get_boxes_xyxy(self, tmp_path):
        """Test internal _get_boxes_xyxy conversion method."""
        weights_path = self._create_test_weights_file(tmp_path)

        with self._patch_segmenter_dependencies():
            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            bbox = self._create_test_bbox(50.0, 50.0, 40.0, 30.0)
            xyxy_boxes = segmenter._get_boxes_xyxy([bbox])

            assert len(xyxy_boxes) == 1
            expected = np.array([30.0, 35.0, 70.0, 65.0])
            np.testing.assert_array_almost_equal(xyxy_boxes[0], expected)
