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

import copy
from typing import Any, Dict, Literal, Optional

from rai_perception.tools.pcl_detection import (
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
)

PresetName = Literal["high", "medium", "low", "top_down", "centroid"]


_PRESETS: Dict[PresetName, Dict[str, Any]] = {
    "high": {
        "filter_config": {
            "strategy": "isolation_forest",
            "if_contamination": 0.01,
            "min_points": 30,
        },
        "estimator_config": {
            "strategy": "top_plane",
            "ransac_iterations": 500,
            "distance_threshold_m": 0.005,
        },
    },
    "medium": {
        "filter_config": {
            "strategy": "isolation_forest",
            "if_contamination": 0.05,
            "min_points": 20,
        },
        "estimator_config": {
            "strategy": "top_plane",
            "ransac_iterations": 200,
            "distance_threshold_m": 0.01,
        },
    },
    "low": {
        "filter_config": {
            "strategy": "dbscan",
            "dbscan_eps": 0.02,
            "dbscan_min_samples": 10,
            "min_points": 20,
        },
        "estimator_config": {
            "strategy": "centroid",
            "min_points": 10,
        },
    },
    "top_down": {
        "filter_config": {
            "strategy": "isolation_forest",
            "if_contamination": 0.05,
        },
        "estimator_config": {
            "strategy": "top_plane",
            "top_percentile": 0.05,
            "ransac_iterations": 300,
        },
    },
    "centroid": {
        "filter_config": {
            "strategy": "dbscan",
            "dbscan_eps": 0.02,
        },
        "estimator_config": {
            "strategy": "centroid",
        },
    },
}


def get_preset(preset_name: PresetName) -> Dict[str, any]:
    """Get preset configuration by name.

    Args:
        preset_name: Name of preset ("high", "medium", "low", "top_down", "centroid")

    Returns:
        Dictionary with filter_config and estimator_config (deep copy)

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in _PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(_PRESETS.keys())}"
        )
    return copy.deepcopy(_PRESETS[preset_name])


def apply_preset(
    preset_name: PresetName,
    base_filter_config: Optional[PointCloudFilterConfig] = None,
    base_estimator_config: Optional[GrippingPointEstimatorConfig] = None,
) -> tuple[PointCloudFilterConfig, GrippingPointEstimatorConfig]:
    """Apply preset to base configs, merging preset values with base configs.

    Args:
        preset_name: Name of preset to apply
        base_filter_config: Base filter config (optional)
        base_estimator_config: Base estimator config (optional)

    Returns:
        Tuple of (filter_config, estimator_config) with preset applied
    """
    preset = get_preset(preset_name)

    # Apply filter preset
    filter_dict = base_filter_config.model_dump() if base_filter_config else {}
    filter_dict.update(preset.get("filter_config", {}))
    filter_config = PointCloudFilterConfig(**filter_dict)

    # Apply estimator preset
    estimator_dict = base_estimator_config.model_dump() if base_estimator_config else {}
    estimator_dict.update(preset.get("estimator_config", {}))
    estimator_config = GrippingPointEstimatorConfig(**estimator_dict)

    return filter_config, estimator_config


def list_presets() -> list[PresetName]:
    """List all available preset names.

    Returns:
        List of preset names
    """
    return list(_PRESETS.keys())
