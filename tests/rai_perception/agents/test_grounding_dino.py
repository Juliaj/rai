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
# See the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rai_perception.agents.grounding_dino import (
    GroundingDinoAgent,
)
from rai_perception.algorithms.boxer import Box
from sensor_msgs.msg import Image

from tests.rai_perception.agents.test_base_vision_agent import (
    cleanup_agent,
    create_valid_weights_file,
)
from tests.rai_perception.conftest import patch_ros2_for_agent_tests


def setup_mock_clock(agent):
    """Setup mock clock for agent tests.

    The code calls clock().now().to_msg() to get ts, then passes ts to
    to_detection_msg which expects rclpy.time.Time and calls ts.to_msg() again.
    However, ts is also assigned to response.detections.header.stamp which expects
    builtin_interfaces.msg.Time.

    ROS2 Humble vs Jazzy difference:
    - Humble: Strict type checking in __debug__ mode requires actual BuiltinTime
      instances, not MagicMock objects. Using MagicMock causes AssertionError.
    - Jazzy: More lenient with MagicMock, but BuiltinTime instances don't allow
      dynamically adding methods (AttributeError when accessing to_msg).

    Solution: Create a wrapper class that inherits from BuiltinTime and adds to_msg().
    """
    from builtin_interfaces.msg import Time as BuiltinTime

    class TimeWithToMsg(BuiltinTime):
        """BuiltinTime wrapper that adds to_msg() method for compatibility."""

        def to_msg(self):
            return self

    mock_clock = MagicMock()
    mock_time = MagicMock()
    # Create a TimeWithToMsg instance (passes isinstance checks and has to_msg())
    mock_ts = TimeWithToMsg()
    mock_time.to_msg.return_value = mock_ts
    mock_clock.now.return_value = mock_time
    agent.ros2_connector._node.get_clock = MagicMock(return_value=mock_clock)


class MockGDBoxer:
    """Mock GDBoxer for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
        """Mock box detection."""
        box1 = Box((50.0, 50.0), 40.0, 40.0, classes[0], 0.9)
        box2 = Box((100.0, 100.0), 30.0, 30.0, classes[1], 0.8)
        return [box1, box2]


class TestGroundingDinoAgent:
    """Test cases for GroundingDinoAgent.

    Note: All tests patch ROS2Connector to prevent hanging. BaseVisionAgent.__init__
    creates a real ROS2Connector which requires ROS2 to be initialized, so we patch
    it to use a mock instead for unit testing.
    """

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test GroundingDinoAgent initialization."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_agent_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch(
                "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)

            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            assert agent._service._boxer is not None

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_init_default_path(self, mock_connector):
        """Test GroundingDinoAgent initialization with default path."""
        weights_path = (
            Path.home() / ".cache/rai/vision/weights/groundingdino_swint_ogc.pth"
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_agent_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch(
                "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)

            agent = GroundingDinoAgent(ros2_name="test")

            assert agent._service._boxer is not None

            cleanup_agent(agent)
            weights_path.unlink()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_agent_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch(
                "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)

            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            with patch.object(
                agent.ros2_connector, "create_service"
            ) as mock_create_service:
                agent.run()

                mock_create_service.assert_called_once()

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_classify_callback(self, tmp_path, mock_connector):
        """Test classify callback processes request correctly."""
        # Create fake weights file with the expected filename (service checks if file exists)
        weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.algorithms.boxer.GDBoxer", MockGDBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_agent_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch(
                "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = MockGDBoxer(weights_path)

            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            # Create mock request
            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur, dragon"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            # Call callback via service
            result = agent._service._classify_callback(request, response)

            # Verify response
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]
            assert result is response

            cleanup_agent(agent)

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
            patch("rai_perception.algorithms.boxer.GDBoxer", EmptyBoxer),
            patch("rai_perception.models.detection.get_model") as mock_get_model,
            patch_ros2_for_agent_tests(mock_connector),
            patch("rai_perception.services.base_vision_service.download_weights"),
            patch(
                "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
            ) as mock_load_model,
        ):
            from rai_perception.algorithms.boxer import GDBoxer

            mock_get_model.return_value = (GDBoxer, "config_path")
            mock_load_model.return_value = EmptyBoxer(weights_path)

            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            result = agent._service._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            cleanup_agent(agent)
