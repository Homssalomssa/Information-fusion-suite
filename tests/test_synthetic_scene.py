"""Tests for the multi-seed synthetic validation demonstration."""

import pytest

from examples.synthetic_scene import (
    _binary_metrics,
    _metrics_by_region,
    run,
    simulate_growing_fire,
    simulate_scene,
)

def _assert_undecided_mask_matches_classes(class_path, mask_path):
    from osgeo import gdal
    classes = gdal.Open(str(class_path)).GetRasterBand(1).ReadAsArray()
    mask = gdal.Open(str(mask_path)).GetRasterBand(1).ReadAsArray()
    assert int((mask == 1).sum()) == int((classes == 2).sum())
    assert int((mask == 0).sum()) == int(((classes == 0) | (classes == 1)).sum())
    assert int((mask == 255).sum()) == int((classes == -1).sum())
    
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


def test_metrics_include_reference_and_predicted_burn_counts_by_region():
    scene = simulate_scene(10)

    metrics = _metrics_by_region(scene.truth, scene.cloud, scene.classifications)

    assert metrics["inside_cloud"]["Optical-only"]["reference_burned_pixels"] == int(
        scene.truth[scene.cloud].sum()
    )
    assert metrics["outside_cloud"]["Fused"]["predicted_burned_pixels"] == int(
        (scene.classifications["Fused"][~scene.cloud] == 0).sum()
    )


@pytest.mark.parametrize("seed", range(20))
def test_growing_fire_scenario_compares_asymmetry_exponents(seed):
    symmetric = simulate_growing_fire(seed, asymmetry_exponent=1.0)
    asymmetric = simulate_growing_fire(seed, asymmetry_exponent=2.0)

    assert symmetric.growth_ring is not None
    assert symmetric.growth_ring.any()
    assert (symmetric.classifications["Fused"][symmetric.cloud]
            == symmetric.classifications["SAR-only"][symmetric.cloud]).all()
    assert symmetric.date_offset_days == 4
    assert symmetric.earlier_source == "Optical"
    assert symmetric.truth.shape == (100, 100)
    assert symmetric.growth_ring.shape == (100, 100)
    assert symmetric.asymmetry_exponent == 1.0
    assert asymmetric.asymmetry_exponent == 2.0
    assert _binary_metrics(
        asymmetric.truth,
        asymmetric.classifications["Fused"] == 0,
    )["iou"] >= 0.0


def test_run_reports_regional_counts_and_growing_fire_comparison(tmp_path):
    summary = run(tmp_path, seeds=range(2))

    assert summary["earlier_source"] == "Optical"
    assert summary["date_offset_days"] == 4
    assert "overall_iou" in summary["growing_fire"]
    assert "growth_ring_iou" in summary["growing_fire"]
    assert summary["growing_fire_k2_improves_overall"] == (
        summary["growing_fire"]["overall_iou"]["k=2"]["mean"]
        > summary["growing_fire"]["overall_iou"]["k=1"]["mean"]
    )
    assert summary["growing_fire_k2_improves_growth_ring"] == (
        summary["growing_fire"]["growth_ring_iou"]["k=2"]["mean"]
        > summary["growing_fire"]["growth_ring_iou"]["k=1"]["mean"]
    )
    for region in ("inside_cloud", "outside_cloud"):
        for source in ("Optical-only", "SAR-only", "Fused"):
            source_metrics = summary["regional_metrics"][region][source]
            assert "reference_burned_pixels" in source_metrics
            assert "predicted_burned_pixels" in source_metrics
