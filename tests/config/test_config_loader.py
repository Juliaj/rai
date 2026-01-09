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

from unittest.mock import MagicMock

import pytest
import yaml
from rai.config.loader import get_config_path, load_python_config, load_yaml_config


class TestLoadYAMLConfig:
    """Test cases for load_yaml_config function."""

    def test_load_yaml_config_basic(self, tmp_path):
        """Test basic YAML config loading."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {"key1": "value1", "key2": 42, "nested": {"key": "value"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_yaml_config(config_file)

        assert result == config_data

    def test_load_yaml_config_with_node_name(self, tmp_path):
        """Test YAML config loading with ROS2 node name pattern."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "test_node": {
                "ros__parameters": {
                    "param1": "value1",
                    "param2": 42,
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_yaml_config(config_file, node_name="test_node")

        assert result == {"param1": "value1", "param2": 42}
        assert "ros__parameters" not in result

    def test_load_yaml_config_missing_file(self, tmp_path):
        """Test YAML config loading with missing file."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_yaml_config(config_file)

        assert "Config file not found" in str(exc_info.value)

    def test_load_yaml_config_invalid_yaml(self, tmp_path):
        """Test YAML config loading with invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            load_yaml_config(config_file)


class TestLoadPythonConfig:
    """Test cases for load_python_config function."""

    def test_load_python_config_basic(self, tmp_path):
        """Test basic Python config loading."""
        config_file = tmp_path / "test_config.py"
        config_content = """
CONFIG_VALUE = "test_value"
CONFIG_NUMBER = 42
"""
        with open(config_file, "w") as f:
            f.write(config_content)

        module = load_python_config(config_file)

        assert module.CONFIG_VALUE == "test_value"
        assert module.CONFIG_NUMBER == 42

    def test_load_python_config_missing_file(self, tmp_path):
        """Test Python config loading with missing file."""
        config_file = tmp_path / "nonexistent.py"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_python_config(config_file)

        assert "Config file not found" in str(exc_info.value)


class TestGetConfigPath:
    """Test cases for get_config_path function."""

    def test_get_config_path_from_parameter(self, tmp_path):
        """Test getting config path from ROS2 parameter."""
        mock_node = MagicMock()
        mock_param = MagicMock()
        mock_param.value = str(tmp_path / "custom_config.yaml")
        mock_node.has_parameter.return_value = True
        mock_node.get_parameter.return_value = mock_param

        result = get_config_path(
            "config_path_param", mock_node, tmp_path, "default.yaml"
        )

        assert result == tmp_path / "custom_config.yaml"

    def test_get_config_path_default_when_missing(self, tmp_path):
        """Test getting default config path when parameter is missing."""
        mock_node = MagicMock()
        mock_node.has_parameter.return_value = False

        result = get_config_path(
            "config_path_param", mock_node, tmp_path, "default.yaml"
        )

        assert result == tmp_path / "default.yaml"
