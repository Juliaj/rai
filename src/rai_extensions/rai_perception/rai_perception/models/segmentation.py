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

"""Segmentation model registry.

Maps segmentation model names to their algorithm classes and configuration paths.
Facilitates switching between models (e.g., developer wants to use a new segmentation model)
without hardcoding model-specific logic or modifying service code.

Example:
    AlgorithmClass, config_path = get_model("grounded_sam")
    segmenter = AlgorithmClass(weights_path, use_cuda=True)
"""

from pathlib import Path
from typing import Tuple, Type

from rai_perception.algorithms.segmenter import GDSegmenter

# Registry: model_name -> (AlgorithmClass, config_path)
# To add a new segmentation model, add an entry here with the model name, algorithm class, and config path.
#
# IMPORTANT: Config loading is model-specific. Different model libraries handle configs differently:
# - Some accept file paths directly (e.g., GroundingDINO's Model class)
# - Some use Hydra internally (e.g., SAM2's build_sam2 function)
# - Some may use other config systems
#
# When adding a new model:
# 1. Check how the model library loads configs (file path, Hydra, etc.)
# 2. If the library initializes its own config system (like Hydra), don't interfere - let it handle initialization
# 3. Return the appropriate config identifier (full path, config name, etc.) based on what the library expects
#
# For SAM2 (grounded_sam): build_sam2 internally uses Hydra and handles its own initialization.
# We pass the full path and let build_sam2 handle Hydra setup. Do NOT initialize Hydra in the algorithm
# class as it conflicts with build_sam2's internal Hydra initialization.
#
# Note: Decorator-based registration (e.g., @register_segmentation_model("name")) is an alternative
# that allows classes to register themselves. Consider switching to decorators if you have many models
# (10+) or want registration at the class definition site rather than a central registry.
_SEGMENTATION_REGISTRY: dict[str, Tuple[Type, str]] = {
    "grounded_sam": (
        GDSegmenter,
        str((Path(__file__).parent.parent / "configs" / "seg_config.yml").resolve()),
    ),
}


def get_model(name: str) -> Tuple[Type, str]:
    """Get segmentation model class and config path by name.

    Args:
        name: Model name (e.g., "grounded_sam")

    Returns:
        Tuple of (AlgorithmClass, config_path)
        - config_path is typically a full file path, but format depends on the model library
        - See registry comments for model-specific config loading requirements

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in _SEGMENTATION_REGISTRY:
        available = ", ".join(_SEGMENTATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown segmentation model '{name}'. Available models: {available}"
        )
    return _SEGMENTATION_REGISTRY[name]


def list_available_models() -> list[str]:
    """List all available segmentation model names.

    Returns:
        List of model names
    """
    return list(_SEGMENTATION_REGISTRY.keys())
