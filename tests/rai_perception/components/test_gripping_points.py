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

import numpy as np
import pytest
from rai_perception.components.gripping_points import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
)


@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@pytest.fixture
def multi_object_point_clouds():
    """Create multiple point clouds for testing."""
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]], dtype=np.float32),
    ]


class TestGrippingPointEstimator:
    """Test cases for GrippingPointEstimator."""

    def test_centroid_strategy(self, sample_point_cloud):
        """Test centroid strategy."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        result = estimator.run([sample_point_cloud])

        assert len(result) == 1
        assert result[0].shape == (3,)
        # Centroid should be mean of all points
        expected = sample_point_cloud.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_centroid_empty_points(self):
        """Test centroid strategy with empty point cloud."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        empty_cloud = np.zeros((0, 3), dtype=np.float32)
        result = estimator.run([empty_cloud])

        assert len(result) == 0

    def test_top_plane_strategy(self):
        """Test top_plane strategy."""
        # Create points with varying Z values
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 2.0],  # Top point
            ],
            dtype=np.float32,
        )
        config = GrippingPointEstimatorConfig(
            strategy="top_plane", top_percentile=0.2, min_points=3
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should be close to top point
        assert result[0][2] > 1.0

    def test_top_plane_fallback_to_centroid(self):
        """Test top_plane falls back to centroid when insufficient points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        config = GrippingPointEstimatorConfig(strategy="top_plane", min_points=5)
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should use centroid fallback
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_biggest_plane_strategy(self):
        """Test biggest_plane strategy with planar points."""
        # Create points on a plane
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.2, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        config = GrippingPointEstimatorConfig(
            strategy="biggest_plane",
            min_points=3,
            ransac_iterations=50,
            distance_threshold_m=0.01,
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Z should be close to 1.0 (plane height)
        assert abs(result[0][2] - 1.0) < 0.1

    def test_biggest_plane_fallback_to_centroid(self):
        """Test biggest_plane falls back to centroid when insufficient points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        config = GrippingPointEstimatorConfig(strategy="biggest_plane", min_points=5)
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_multiple_clouds(self, multi_object_point_clouds):
        """Test processing multiple point clouds."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        result = estimator.run(multi_object_point_clouds)

        assert len(result) == 2
        assert all(gp.shape == (3,) for gp in result)

    def test_unknown_strategy_fallback(self, sample_point_cloud):
        """Test that unknown strategy falls back to centroid."""
        config = GrippingPointEstimatorConfig()
        config.strategy = "unknown_strategy"  # type: ignore
        estimator = GrippingPointEstimator(config)
        result = estimator.run([sample_point_cloud])

        assert len(result) == 1
        # Should use centroid as fallback
        expected = sample_point_cloud.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)


class TestPointCloudFilter:
    """Test cases for PointCloudFilter."""

    def test_isolation_forest_strategy(self):
        """Test isolation_forest filtering strategy."""
        # Create points with outliers
        inliers = np.random.rand(50, 3).astype(np.float32) * 0.1
        outliers = np.array(
            [[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]], dtype=np.float32
        )
        points = np.vstack([inliers, outliers])

        config = PointCloudFilterConfig(
            strategy="isolation_forest",
            min_points=10,
            if_contamination=0.1,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        # Should have fewer points (outliers removed)
        assert result[0].shape[0] < points.shape[0]
        assert result[0].shape[0] >= inliers.shape[0] * 0.8  # Most inliers kept

    def test_dbscan_strategy(self):
        """Test DBSCAN filtering strategy."""
        # Create two clusters
        cluster1 = np.random.rand(30, 3).astype(np.float32) * 0.1
        cluster2 = (np.random.rand(20, 3).astype(np.float32) * 0.1) + np.array(
            [5.0, 5.0, 5.0]
        )
        points = np.vstack([cluster1, cluster2])

        config = PointCloudFilterConfig(
            strategy="dbscan",
            min_points=10,
            dbscan_eps=0.5,
            dbscan_min_samples=5,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        # Should keep largest cluster
        assert result[0].shape[0] > 0

    def test_kmeans_largest_cluster_strategy(self):
        """Test kmeans_largest_cluster filtering strategy."""
        # Create two clusters
        cluster1 = np.random.rand(40, 3).astype(np.float32) * 0.1
        cluster2 = (np.random.rand(20, 3).astype(np.float32) * 0.1) + np.array(
            [5.0, 5.0, 5.0]
        )
        points = np.vstack([cluster1, cluster2])

        config = PointCloudFilterConfig(
            strategy="kmeans_largest_cluster",
            min_points=10,
            kmeans_k=2,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        assert result[0].shape[0] > 0

    def test_filter_insufficient_points(self):
        """Test filtering with insufficient points returns original."""
        points = np.random.rand(5, 3).astype(np.float32)
        config = PointCloudFilterConfig(strategy="isolation_forest", min_points=10)
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        assert result[0].shape == points.shape

    def test_filter_empty_points(self):
        """Test filtering empty point cloud."""
        empty_cloud = np.zeros((0, 3), dtype=np.float32)
        config = PointCloudFilterConfig(strategy="isolation_forest")
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([empty_cloud])

        assert len(result) == 0

    def test_multiple_clouds(self, multi_object_point_clouds):
        """Test filtering multiple point clouds."""
        config = PointCloudFilterConfig(strategy="isolation_forest", min_points=2)
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run(multi_object_point_clouds)

        assert len(result) == 2
        assert all(filt.shape[1] == 3 for filt in result)

    def test_unknown_strategy_returns_original(self, sample_point_cloud):
        """Test that unknown strategy returns original points."""
        config = PointCloudFilterConfig()
        config.strategy = "unknown_strategy"  # type: ignore
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([sample_point_cloud])

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], sample_point_cloud)
