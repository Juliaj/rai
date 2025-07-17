# Copyright (C) 2024 Robotec.AI
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

import cv2
import numpy as np
from typing import List, Optional, Any


def visualize_detections_on_image(
    image: np.ndarray, 
    detections: List, 
    output_path: Optional[str] = None,
    save_image: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on the image and optionally save it.
    
    Args:
        image: Input image as numpy array (BGR format)
        detections: List of Detection2D objects from GDINO response
        output_path: Path to save the annotated image (optional)
        save_image: Whether to save the image to disk
        
    Returns:
        Annotated image as numpy array
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for i, detection in enumerate(detections):
        bbox = detection.bbox
        class_name = detection.results[0].hypothesis.class_id
        confidence = detection.results[0].hypothesis.score
        
        # Calculate bounding box coordinates
        x_center = int(bbox.center.position.x)
        y_center = int(bbox.center.position.y)
        width = int(bbox.size_x)
        height = int(bbox.size_y)
        
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the annotated image if requested
    if save_image and output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def filter_large_detections(
    detections: List[Any], 
    image_shape: tuple,
    max_size_ratio: float = 0.25,
    min_size: int = 20
) -> List[Any]:
    """
    Filter out detections that are too large (likely false positives).
    
    Args:
        detections: List of Detection2D objects
        image_shape: Tuple of (height, width) of the image
        max_size_ratio: Maximum allowed size as ratio of image dimensions (default: 0.25)
        min_size: Minimum allowed size in pixels (default: 20)
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    img_height, img_width = image_shape[:2]
    
    # Calculate maximum allowed detection size
    max_width = img_width * max_size_ratio
    max_height = img_height * max_size_ratio
    
    for detection in detections:
        bbox = detection.bbox
        width = bbox.size_x
        height = bbox.size_y
        
        # Check if detection is within reasonable size limits
        if (min_size <= width <= max_width and 
            min_size <= height <= max_height):
            filtered.append(detection)
    
    return filtered 