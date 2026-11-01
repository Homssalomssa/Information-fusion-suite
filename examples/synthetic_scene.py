"""Generate and score a synthetic burned-area possibility-fusion example."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

from possibility_fusion.core.possibility import adaptive_fusion, decide
from possibility_fusion.core.temporal import discount_for_date_offset, temporal_reliability


@dataclass
class SyntheticScene:
    """Inputs and source/fused decisions for one deterministic synthetic scene."""

    truth: NDArray[np.bool_]
    cloud: NDArray[np.bool_]
    optical_score: NDArray[np.float32]
    sar_score: NDArray[np.float32]
    classifications: dict[str, NDArray[np.int8]]
    optical_date: date
    sar_date: date
    temporal_reliability: float
    date_offset_days: int
    earlier_source: str
    asymmetry_exponent: float
    growth_ring: NDArray[np.bool_] | None = None


def _possibility_from_burn_score(
    burn_score: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convert a burn score to a normalized two-class possibility distribution."""
    unburned_score = 1.0 - burn_score
    maximum = np.maximum(burn_score, unburned_score)
    return np.stack((burn_score / maximum, unburned_score / maximum)).astype(np.float32)


def _binary_metrics(
    reference: NDArray[np.bool_],
    prediction: NDArray[np.bool_],
) -> dict[str, float]:
    """Calculate burned-class IoU and F1."""
    true_positive = np.count_nonzero(reference & prediction)
    false_positive = np.count_nonzero(~reference & prediction)
    false_negative = np.count_nonzero(reference & ~prediction)
    union = true_positive + false_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "iou": float(true_positive / union) if union else 1.0,
        "f1": float(2 * true_positive / f1_denominator) if f1_denominator else 1.0,
    }


def _metrics_by_region(
    truth: NDArray[np.bool_],
    cloud: NDArray[np.bool_],
    classifications: dict[str, NDArray[np.int8]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Measure burned-class scores and undecided fraction inside/outside cloud."""
    regions = {"inside_cloud": cloud, "outside_cloud": ~cloud}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for region_name, mask in regions.items():
        region_result: dict[str, dict[str, float]] = {}
        pixel_count = int(np.count_nonzero(mask))
        for source, classes in classifications.items():
            scores = _binary_metrics(truth[mask], classes[mask] == 0)
            scores["undecided_fraction"] = float(
                np.count_nonzero(classes[mask] == 2) / pixel_count
            )
            scores["reference_burned_pixels"] = float(
                np.count_nonzero(truth & mask)
            )
            scores["predicted_burned_pixels"] = float(
                np.count_nonzero((classes == 0) & mask)
            )
            region_result[source] = scores
        result[region_name] = region_result
    return result


def simulate_scene(seed: int) -> SyntheticScene:
    """Generate one deterministic 100x100 cloud/date-offset validation scene."""
    return _simulate_scene(seed, growing_fire=False, asymmetry_exponent=2.0)


def simulate_growing_fire(
    seed: int,
    *,
    asymmetry_exponent: float = 2.0,
) -> SyntheticScene:
    """Generate fire growth between the optical and later SAR acquisitions."""
    return _simulate_scene(
        seed,
        growing_fire=True,
        asymmetry_exponent=asymmetry_exponent,
    )


def _simulate_scene(
    seed: int,
    *,
    growing_fire: bool,
    asymmetry_exponent: float,
) -> SyntheticScene:
    """Generate source evidence, apply temporal discounting, and classify."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not np.isfinite(asymmetry_exponent) or asymmetry_exponent < 1.0:
        raise ValueError("asymmetry_exponent must be finite and at least 1.")
    rng = np.random.default_rng(int(seed))
    rows, columns = np.mgrid[:100, :100]
    optical_truth = (rows - 50) ** 2 + (columns - 50) ** 2 <= 21**2
    truth = (
        (rows - 50) ** 2 + (columns - 50) ** 2 <= 26**2
        if growing_fire
        else optical_truth
    )
    growth_ring = truth & ~optical_truth if growing_fire else None

    optical_date = date(2024, 8, 10)
    sar_date = optical_date + timedelta(days=4)
    synthetic_fire_end = optical_date + timedelta(days=2)
    tau_days = 14.0
    date_offset_days = (sar_date - optical_date).days
    alpha_time = temporal_reliability(
        optical_date,
        sar_date,
        tau_days=tau_days,
    )
    earlier_source = (
        "Optical" if date_offset_days > 0 else "SAR" if date_offset_days < 0 else "Neither"
    )

    optical_signal = np.where(optical_truth, 2.4, -2.4)
    optical_score = (
        1.0 / (1.0 + np.exp(-(optical_signal + rng.normal(0, 1.3, truth.shape))))
    ).astype(np.float32)
    sar_signal = np.where(truth, 2.4, -2.4)
    sar_score = (
        1.0 / (1.0 + np.exp(-(sar_signal + rng.normal(0, 1.3, truth.shape))))
    ).astype(np.float32)

    cloud = (rows >= 34) & (rows < 67) & (columns >= 34) & (columns < 67)
    optical_score[cloud] = rng.uniform(0.0, 1.0, np.count_nonzero(cloud))
    optical_possibility = _possibility_from_burn_score(optical_score)
    sar_possibility = _possibility_from_burn_score(sar_score)

    optical_reliability = np.ones(truth.shape, dtype=np.float32)
    optical_reliability[cloud] = 0.0
    optical_discounted = discount_for_date_offset(
        optical_possibility,
        optical_reliability,
        alpha_time=alpha_time,
        source_is_earlier=date_offset_days > 0,
        is_post_fire_pair=True,
        asymmetry_exponent=asymmetry_exponent,
        source_date=optical_date,
        fire_end_date=synthetic_fire_end,
    )
    sar_discounted = discount_for_date_offset(
        sar_possibility,
        1.0,
        alpha_time=alpha_time,
        source_is_earlier=date_offset_days < 0,
        is_post_fire_pair=True,
        asymmetry_exponent=asymmetry_exponent,
        source_date=sar_date,
        fire_end_date=synthetic_fire_end,
    )
    fused, _ = adaptive_fusion(optical_discounted, sar_discounted)
    classifications = {
        "Optical-only": decide(optical_discounted)[0],
        "SAR-only": decide(sar_discounted)[0],
        "Fused": decide(fused)[0],
    }

    return SyntheticScene(
        truth=truth,
        cloud=cloud,
        optical_score=optical_score,
        sar_score=sar_score,
        classifications=classifications,
        optical_date=optical_date,
        sar_date=sar_date,
        temporal_reliability=alpha_time,
        date_offset_days=date_offset_days,
        earlier_source=earlier_source,
        asymmetry_exponent=asymmetry_exponent,
        growth_ring=growth_ring,
    )


def _plot_evidence(
    output: Path,
    truth: NDArray[np.bool_],
    optical_score: NDArray[np.float32],
    sar_score: NDArray[np.float32],
    cloud: NDArray[np.bool_],
    optical_date: date,
    sar_date: date,
    temporal_reliability: float,
) -> None:
    """Save input truth, sensor evidence, and cloud-mask overview."""
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    panels = (
        (truth, "Reference burned blob", "gray"),
        (optical_score, "Optical burn score\n(cloud overlay)", "magma"),
        (sar_score, f"SAR burn score\n(acquired {sar_date})", "magma"),
        (cloud, "Optical cloud / smoke mask", "gray"),
    )
    for axis, (data, title, color_map) in zip(axes, panels):
        axis.imshow(data, cmap=color_map, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    axes[1].contour(cloud, levels=[0.5], colors=["cyan"], linewidths=0.8)
    figure.suptitle(
        f"Synthetic inputs: optical {optical_date}, SAR {sar_date}; "
        f"date-offset reliability={temporal_reliability:.3f}"
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_classification(
    output: Path,
    truth: NDArray[np.bool_],
    classifications: dict[str, NDArray[np.int8]],
    metrics: dict[str, dict[str, float]],
) -> None:
    """Save reference and the three burned/unburned classification results."""
    class_map = ListedColormap(["#c0392b", "#d9d9d9", "#3498db"])
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    panels = [("Reference", np.where(truth, 0, 1))]
    for name, classes in classifications.items():
        score = metrics[name]
        panels.append(
            (
                f"{name}\nIoU={score['iou']:.3f}  F1={score['f1']:.3f}",
                classes,
            )
        )

    for axis, (title, classes) in zip(axes, panels):
        axis.imshow(classes, cmap=class_map, vmin=0, vmax=2, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.legend(
        handles=[
            plt.Line2D([0], [0], color="#c0392b", lw=6, label="Burned"),
            plt.Line2D([0], [0], color="#d9d9d9", lw=6, label="Unburned"),
            plt.Line2D([0], [0], color="#3498db", lw=6, label="Undecided"),
        ],
        loc="lower center",
        ncol=3,
    )
    figure.suptitle(
        "Burned-class comparison (undecided counts as not predicted burned)"
    )
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_growing_fire(
    output: Path,
    scene: SyntheticScene,
    fused_by_exponent: dict[float, NDArray[np.int8]],
    metrics: dict[float, dict[str, float]],
) -> None:
    """Plot final perimeter, growth ring, and k=1/k=2 fusion comparisons."""
    assert scene.growth_ring is not None
    class_map = ListedColormap(["#c0392b", "#d9d9d9", "#3498db"])
    panels = [
        ("Final reference perimeter", np.where(scene.truth, 0, 1)),
        ("Growth ring after optical date", np.where(scene.growth_ring, 0, 1)),
    ]
    for exponent, classes in fused_by_exponent.items():
        panels.append(
            (
                f"Fused k={exponent:g}\n"
                f"overall IoU={metrics[exponent]['overall_iou']:.3f}\n"
                f"ring IoU={metrics[exponent]['growth_ring_iou']:.3f}",
                classes,
            )
        )
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (title, data) in zip(axes, panels):
        axis.imshow(data, cmap=class_map, vmin=0, vmax=2, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    axes[2].contour(scene.growth_ring, levels=[0.5], colors=["cyan"], linewidths=0.8)
    axes[3].contour(scene.growth_ring, levels=[0.5], colors=["cyan"], linewidths=0.8)
    figure.suptitle(
        "Growing fire: optical earlier than SAR; cyan outlines the growth ring"
    )
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _mean_std(values: list[float]) -> dict[str, float]:
    """Summarize repeated metric values using population standard deviation."""
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def run(
    output_dir: Path,
    seeds: Iterable[int] = range(20),
) -> dict[str, object]:
    """Run repeated scenes, aggregate metrics, and save representative plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required.")
    scenes = [simulate_scene(seed) for seed in seed_list]
    first_scene = scenes[0]
    whole_scene_metrics = [
        {
            source: _binary_metrics(
                scene.truth,
                scene.classifications[source] == 0,
            )
            for source in scene.classifications
        }
        for scene in scenes
    ]
    per_scene_iou = {
        source: [run[source]["iou"] for run in whole_scene_metrics]
        for source in first_scene.classifications
    }
    mean_iou = {source: _mean_std(values) for source, values in per_scene_iou.items()}
    regional_runs = [
        _metrics_by_region(scene.truth, scene.cloud, scene.classifications)
        for scene in scenes
    ]
    regional_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for region in ("inside_cloud", "outside_cloud"):
        regional_summary[region] = {}
        for source in first_scene.classifications:
            regional_summary[region][source] = {}
            for metric in (
                "iou",
                "f1",
                "undecided_fraction",
                "reference_burned_pixels",
                "predicted_burned_pixels",
            ):
                regional_summary[region][source][metric] = _mean_std(
                    [
                        run[region][source][metric]
                        for run in regional_runs
                    ]
                )

    growing_scenes = {
        exponent: [simulate_growing_fire(seed, asymmetry_exponent=exponent) for seed in seed_list]
        for exponent in (1.0, 2.0)
    }
    growing_metrics: dict[float, list[dict[str, float]]] = {}
    for exponent, growing_runs in growing_scenes.items():
        growing_metrics[exponent] = []
        for scene in growing_runs:
            assert scene.growth_ring is not None
            growth_truth = scene.truth[scene.growth_ring]
            growth_prediction = scene.classifications["Fused"][scene.growth_ring] == 0
            growing_metrics[exponent].append(
                {
                    "overall_iou": _binary_metrics(
                        scene.truth,
                        scene.classifications["Fused"] == 0,
                    )["iou"],
                    "growth_ring_iou": _binary_metrics(
                        growth_truth,
                        growth_prediction,
                    )["iou"],
                }
            )

    growth_summary = {
        metric: {
            f"k={exponent:g}": _mean_std(
                [run[metric] for run in growing_metrics[exponent]]
            )
            for exponent in (1.0, 2.0)
        }
        for metric in ("overall_iou", "growth_ring_iou")
    }
    growing_first = growing_scenes[1.0][0]
    growing_fused = {
        exponent: growing_scenes[exponent][0].classifications["Fused"]
        for exponent in (1.0, 2.0)
    }
    growing_first_metrics = {
        exponent: growing_metrics[exponent][0] for exponent in (1.0, 2.0)
    }
    _plot_growing_fire(
        output_dir / "synthetic_growing_fire.png",
        growing_first,
        growing_fused,
        growing_first_metrics,
    )

    _plot_evidence(
        output_dir / "synthetic_evidence.png",
        first_scene.truth,
        first_scene.optical_score,
        first_scene.sar_score,
        first_scene.cloud,
        first_scene.optical_date,
        first_scene.sar_date,
        first_scene.temporal_reliability,
    )
    _plot_classification(
        output_dir / "synthetic_classification.png",
        first_scene.truth,
        first_scene.classifications,
        whole_scene_metrics[0],
    )
    date_offset_days = first_scene.date_offset_days
    earlier_source = first_scene.earlier_source
    (output_dir / "synthetic_metrics.json").write_text(
        json.dumps(
            {
                "scene_shape": [100, 100],
                "seeds": seed_list,
                "optical_date": first_scene.optical_date.isoformat(),
                "sar_date": first_scene.sar_date.isoformat(),
                "date_offset_days": date_offset_days,
                "earlier_source": earlier_source,
                "tau_days": 14.0,
                "temporal_reliability": first_scene.temporal_reliability,
                "cloud_pixels": int(np.count_nonzero(first_scene.cloud)),
                "whole_scene_iou_mean_std": mean_iou,
                "regional_metrics_mean_std": regional_summary,
                "growing_fire_iou_mean_std": growth_summary,
                "growing_fire_k2_improves_overall": (
                    growth_summary["overall_iou"]["k=2"]["mean"]
                    > growth_summary["overall_iou"]["k=1"]["mean"]
                ),
                "growing_fire_k2_improves_growth_ring": (
                    growth_summary["growth_ring_iou"]["k=2"]["mean"]
                    > growth_summary["growth_ring_iou"]["k=1"]["mean"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "seeds": seed_list,
        "mean_iou": mean_iou,
        "regional_metrics": regional_summary,
        "optical_date": first_scene.optical_date,
        "sar_date": first_scene.sar_date,
        "date_offset_days": date_offset_days,
        "earlier_source": earlier_source,
        "growing_fire": growth_summary,
        "growing_fire_k2_improves_overall": (
            growth_summary["overall_iou"]["k=2"]["mean"]
            > growth_summary["overall_iou"]["k=1"]["mean"]
        ),
        "growing_fire_k2_improves_growth_ring": (
            growth_summary["growth_ring_iou"]["k=2"]["mean"]
            > growth_summary["growth_ring_iou"]["k=1"]["mean"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/synthetic"),
        help="directory for local plots and synthetic metrics",
    )
    arguments = parser.parse_args()
    summary = run(arguments.output_dir)
    print(
        f"Date offset: {summary['earlier_source']} source is earlier by "
        f"{summary['date_offset_days']} days "
        f"(optical {summary['optical_date']}, SAR {summary['sar_date']})."
    )
    print(
        f"Whole-scene burned-class IoU over {len(summary['seeds'])} seeds "
        "(mean +/- population std; undecided is not burned):"
    )
    for source, score in summary["mean_iou"].items():
        print(f"{source:12s} {score['mean']:.4f} +/- {score['std']:.4f}")
    for region, sources in summary["regional_metrics"].items():
        print(f"{region.replace('_', ' ').title()} (mean +/- population std):")
        for source, measures in sources.items():
            print(
                f"  {source:12s} IoU={measures['iou']['mean']:.4f}"
                f"+/-{measures['iou']['std']:.4f}"
                f" F1={measures['f1']['mean']:.4f}"
                f"+/-{measures['f1']['std']:.4f}"
                f" reference_burned={measures['reference_burned_pixels']['mean']:.1f}"
                f" predicted_burned={measures['predicted_burned_pixels']['mean']:.1f}"
                f" undecided={measures['undecided_fraction']['mean']:.4f}"
                f"+/-{measures['undecided_fraction']['std']:.4f}"
            )
    print("Growing-fire fused IoU over the same seeds (mean +/- population std):")
    for metric in ("overall_iou", "growth_ring_iou"):
        print(f"  {metric.replace('_', ' ')}:")
        for exponent in ("k=1", "k=2"):
            score = summary["growing_fire"][metric][exponent]
            print(f"    {exponent}: {score['mean']:.4f} +/- {score['std']:.4f}")
    print(
        "k=2 improves overall IoU: "
        f"{'yes' if summary['growing_fire_k2_improves_overall'] else 'no'}; "
        "growth-ring IoU: "
        f"{'yes' if summary['growing_fire_k2_improves_growth_ring'] else 'no'}."
    )
    print(f"Plots and metrics saved under {arguments.output_dir}")


if __name__ == "__main__":
    main()
