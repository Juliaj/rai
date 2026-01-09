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

"""Tests for DetectionService.

Tests the model-agnostic detection service that uses the detection model registry.
"""

from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rai_perception.agents.grounding_dino import GDINO_SERVICE_NAME
from rai_perception.services.detection_service import DetectionService
from rai_perception.vision_markup.boxer import Box
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from tests.rai_perception.conftest import (
    create_valid_weights_file,
    patch_ros2_for_service_tests,
)


class MockGDBoxer:
    """Mock GDBoxer for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
        """Mock box detection."""
        box1 = Box((50.0, 50.0), 40.0, 40.0, classes[0], 0.9)
        box2 = Box((100.0, 100.0), 30.0, 30.0, classes[1], 0.8)
        return [box1, box2]


def setup_mock_clock(instance):
    """Setup mock clock for tests."""
    from builtin_interfaces.msg import Time as BuiltinTime

    class TimeWithToMsg(BuiltinTime):
        """BuiltinTime wrapper that adds to_msg() method for compatibility."""

        def to_msg(self):
            return self

    mock_clock = MagicMock()
    mock_time = MagicMock()
    mock_ts = TimeWithToMsg()
    mock_time.to_msg.return_value = mock_ts
    mock_clock.now.return_value = mock_time

    instance.ros2_connector.node.get_clock = MagicMock(return_value=mock_clock)


class TestDetectionService:
    """Tests for DetectionService."""

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test initialization."""
        # Create weights file with the expected filename
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.vision_markup.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                DetectionService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.vision_markup.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)

            # Set ROS2 parameters for service
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounding_dino",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        GDINO_SERVICE_NAME,
                    ),
                ]
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            assert instance._boxer is not None
            instance.stop()

    @pytest.mark.timeout(10)
    def test_classify_callback_empty_boxes(self, tmp_path, mock_connector):
        """Test classify callback with no detections."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        class EmptyBoxer:
            def __init__(self, weights_path):
                self.weights_path = weights_path

            def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
                return []

        with (
            patch("rai_perception.vision_markup.boxer.GDBoxer", EmptyBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                DetectionService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.vision_markup.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = EmptyBoxer(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounding_dino",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        GDINO_SERVICE_NAME,
                    ),
                ]
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(instance)
            result = instance._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            instance.stop()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.vision_markup.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                DetectionService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.vision_markup.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounding_dino",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        GDINO_SERVICE_NAME,
                    ),
                ]
            )

            instance = DetectionService(
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
                    call_args[0][0] == GDINO_SERVICE_NAME
                    or call_args[0][0] == "/detection"
                )
                assert (
                    call_args[1].get("service_type")
                    == "rai_interfaces/srv/RAIGroundingDino"
                    or call_args[0][2] == "rai_interfaces/srv/RAIGroundingDino"
                )

            instance.stop()

    @pytest.mark.timeout(10)
    def test_classify_callback(self, tmp_path, mock_connector):
        """Test classify callback processes request correctly."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.vision_markup.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_service_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch.object(
                DetectionService, "_load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.vision_markup.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)
            mock_connector.node.set_parameters(
                [
                    Parameter(
                        "model_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        "grounding_dino",
                    ),
                    Parameter(
                        "service_name",
                        rclpy.parameter.Parameter.Type.STRING,
                        GDINO_SERVICE_NAME,
                    ),
                ]
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur, dragon"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(instance)
            result = instance._classify_callback(request, response)

            # Verify response
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]
            assert result is response

            instance.stop()
