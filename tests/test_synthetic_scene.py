"""Tests for the multi-seed synthetic validation demonstration."""

import pytest

from examples.synthetic_scene import _binary_metrics, simulate_scene


def test_fusion_matches_sar_inside_cloud_and_is_best_outside_for_twenty_seeds():
    for seed in range(20):
        scene = simulate_scene(seed)
        truth = scene.truth
        cloud = scene.cloud
        classes = scene.classifications

        assert (classes["Fused"][cloud] == classes["SAR-only"][cloud]).all()

        outside_metrics = {
            name: _binary_metrics(truth[~cloud], prediction[~cloud] == 0)
            for name, prediction in classes.items()
        }
        assert outside_metrics["Fused"]["iou"] >= max(
            outside_metrics["Optical-only"]["iou"],
            outside_metrics["SAR-only"]["iou"],
        )


def test_simulation_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        simulate_scene(-1)
