# Copyright (C) 2025 Robotec.AI
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

"""Tests for SegmentationService.

Tests the model-agnostic segmentation service that uses the segmentation model registry.
"""

from unittest.mock import patch

import numpy as np
import pytest
import rclpy
from rai_perception.services.segmentation_service import SegmentationService
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D

from rai_interfaces.msg import RAIDetectionArray
from rai_interfaces.srv import RAIGroundedSam
from tests.rai_perception.conftest import (
    create_valid_weights_file,
    patch_ros2_for_service_tests,
)

# Service name default changed from "grounded_sam_segment" to "/segmentation"
SEGMENTATION_SERVICE_NAME = "/segmentation"


class MockGDSegmenter:
    """Mock GDSegmenter for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_segmentation(self, image, boxes):
        """Mock segmentation that returns simple masks."""
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[10:50, 10:50] = 1.0
        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[60:90, 60:90] = 1.0
        return [mask1, mask2]


class TestSegmentationService:
    """Tests for SegmentationService."""

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test initialization."""
        # Create weights file with the expected filename
        weights_path = tmp_path / "vision" / "weights" / "sam2_hiera_large.pt"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.segmenter.GDSegmenter", MockGDSegmenter),
            patch("rai_perception.models.segmentation.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                SegmentationService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.segmenter import GDSegmenter

            mock_get_model.return_value = (GDSegmenter, "config_path")
            mock_load_model.return_value = MockGDSegmenter(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounded_sam",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        SEGMENTATION_SERVICE_NAME,
                    ),
                ]
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            assert instance._segmenter is not None
            instance.stop()

    @pytest.mark.timeout(10)
    def test_segment_callback_empty_detections(self, tmp_path, mock_connector):
        """Test segment callback with empty detections."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "sam2_hiera_large.pt"
        create_valid_weights_file(weights_path)

        class EmptySegmenter:
            def __init__(self, weights_path):
                self.weights_path = weights_path

            def get_segmentation(self, image, boxes):
                return []

        with (
            patch("rai_perception.algorithms.segmenter.GDSegmenter", EmptySegmenter),
            patch("rai_perception.models.segmentation.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                SegmentationService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.segmenter import GDSegmenter

            mock_get_model.return_value = (GDSegmenter, "config_path")
            mock_load_model.return_value = EmptySegmenter(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounded_sam",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        SEGMENTATION_SERVICE_NAME,
                    ),
                ]
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            request = RAIGroundedSam.Request()
            request.source_img = Image()
            request.detections = RAIDetectionArray()
            request.detections.detections = []

            response = RAIGroundedSam.Response()
            result = instance._segment_callback(request, response)

            assert len(result.masks) == 0
            instance.stop()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "sam2_hiera_large.pt"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.segmenter.GDSegmenter", MockGDSegmenter),
            patch("rai_perception.models.segmentation.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                SegmentationService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.segmenter import GDSegmenter

            mock_get_model.return_value = (GDSegmenter, "config_path")
            mock_load_model.return_value = MockGDSegmenter(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounded_sam",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        SEGMENTATION_SERVICE_NAME,
                    ),
                ]
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            with patch.object(
                instance.ros2_connector, "create_service"
            ) as mock_create_service:
                instance.run()

                mock_create_service.assert_called_once()
                call_args = mock_create_service.call_args
                assert (
                    call_args[1].get("service_type")
                    == "rai_interfaces/srv/RAIGroundedSam"
                    or call_args[0][2] == "rai_interfaces/srv/RAIGroundedSam"
                )

            instance.stop()

    @pytest.mark.timeout(10)
    def test_segment_callback(self, tmp_path, mock_connector):
        """Test segment callback processes request correctly."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "sam2_hiera_large.pt"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.segmenter.GDSegmenter", MockGDSegmenter),
            patch("rai_perception.models.segmentation.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                SegmentationService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.segmenter import GDSegmenter

            mock_get_model.return_value = (GDSegmenter, "config_path")
            mock_load_model.return_value = MockGDSegmenter(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounded_sam",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        SEGMENTATION_SERVICE_NAME,
                    ),
                ]
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            request = RAIGroundedSam.Request()
            request.source_img = Image()

            detection1 = Detection2D()
            detection1.bbox = BoundingBox2D()
            detection1.bbox.center.position.x = 30.0
            detection1.bbox.center.position.y = 30.0
            detection1.bbox.size_x = 40.0
            detection1.bbox.size_y = 40.0

            detection2 = Detection2D()
            detection2.bbox = BoundingBox2D()
            detection2.bbox.center.position.x = 75.0
            detection2.bbox.center.position.y = 75.0
            detection2.bbox.size_x = 30.0
            detection2.bbox.size_y = 30.0

            request.detections = RAIDetectionArray()
            request.detections.detections = [detection1, detection2]

            response = RAIGroundedSam.Response()
            result = instance._segment_callback(request, response)

            assert len(result.masks) == 2
            assert result is response

            instance.stop()
