import sys
sys.path.append('src/rai_core')
import os
import h5py
import json
import numpy as np
from PIL import Image
from rai_perception import depth_to_point_cloud

# Replicate the test data loading
data_path = 'tests/resources/detection_data/cracker_box'
with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

# Load depth
with h5py.File(os.path.join(data_path, 'np1_42_depth.h5'), 'r') as f:
    depth = np.array(f['depth']).astype(np.float32)

# Load mask
mask_pil = Image.open(os.path.join(data_path, 'np1_42_mask.pbm'))
camera_info = metadata['camera_info']
depth_w, depth_h = camera_info['depth']['width'], camera_info['depth']['height']
mask_resized = mask_pil.resize((depth_w, depth_h), Image.NEAREST)
mask = np.array(mask_resized)

print('Depth shape:', depth.shape)
print('Mask shape:', mask.shape)
print('Depth min/max:', depth.min(), depth.max())
print('Mask unique values:', np.unique(mask))

# Check mask application
binary_mask = mask == 255
print('Binary mask sum:', binary_mask.sum(), 'out of', binary_mask.size)

# Apply mask and convert to meters
masked_depth = np.zeros_like(depth)
depth_scale = camera_info.get('depth_scale', 0.001)
masked_depth[binary_mask] = depth[binary_mask] * depth_scale

print('Masked depth min/max:', masked_depth.min(), masked_depth.max())
print('Masked depth nonzero count:', np.count_nonzero(masked_depth))

# Generate point cloud
points = depth_to_point_cloud(
    masked_depth,
    camera_info['fx'], camera_info['fy'],
    camera_info['cx'], camera_info['cy']
)

print('Points shape:', points.shape if points.size > 0 else 'Empty')
print('Points min/max:', points.min(), points.max() if points.size > 0 else 'Empty')