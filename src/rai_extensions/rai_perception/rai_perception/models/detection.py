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

"""Detection model registry.

Maps detection model names to their algorithm classes and configuration paths.
Facilitates switching between models (e.g., developer wants to use a new detection model)
without hardcoding model-specific logic or modifying service code.

Example:
    AlgorithmClass, config_path = get_model("grounding_dino")
    boxer = AlgorithmClass(weights_path, config_path=config_path)
"""

from pathlib import Path
from typing import Tuple, Type

from rai_perception.vision_markup.boxer import GDBoxer

# Registry: model_name -> (AlgorithmClass, config_path)
# To add a new detection model, add an entry here with the model name, algorithm class, and config path.
#
# Note: Decorator-based registration (e.g., @register_detection_model("name")) is an alternative
# that allows classes to register themselves. Consider switching to decorators if you have many models
# (10+) or want registration at the class definition site rather than a central registry.
_DETECTION_REGISTRY: dict[str, Tuple[Type, str]] = {
    "grounding_dino": (
        GDBoxer,
        str(Path(__file__).parent.parent / "configs" / "gdino_config.py"),
    ),
}


def get_model(name: str) -> Tuple[Type, str]:
    """Get detection model class and config path by name.

    Args:
        name: Model name (e.g., "grounding_dino")

    Returns:
        Tuple of (AlgorithmClass, config_path)

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in _DETECTION_REGISTRY:
        available = ", ".join(_DETECTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown detection model '{name}'. Available models: {available}"
        )
    return _DETECTION_REGISTRY[name]


def list_available_models() -> list[str]:
    """List all available detection model names.

    Returns:
        List of model names
    """
    return list(_DETECTION_REGISTRY.keys())
