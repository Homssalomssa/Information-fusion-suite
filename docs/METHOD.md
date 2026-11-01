# Possibility-theory method

Possibility Fusion combines optical and radar evidence about two classes:
**Burned** and **Unburned**. It uses possibility theory, not probability or
Dempster–Shafer theory. For each pixel, an evidence source supplies a
possibility distribution `(pi_B, pi_U)` with values in `[0, 1]` and
`max(pi_B, pi_U) = 1`. A value of 1 means a class is fully possible; values
below 1 express exclusion or reduced compatibility. `(1, 1)` represents total
ignorance.

## Discounting and adaptive fusion

Reliability `alpha` is in `[0, 1]`. Discounting weakens evidence according to

`pi'[c] = max(pi[c], 1 - alpha)`.

Thus `alpha=1` leaves evidence unchanged; `alpha=0` gives total ignorance.
For example, discounting `(1, 0.3)` with `alpha=0.5` yields `(1, 0.5)`.
Temporal reliability is based on the separation between acquisitions,
`alpha_time = exp(-abs(dt) / tau_days)`, and combines with sensor reliability
before discounting.

For two distributions `pi1` and `pi2`, let
`h = max_c min(pi1[c], pi2[c])`. Adaptive fusion combines their normalized
conjunction and a conflict-aware disjunction:

`pi_adaptive[c] = max(min(pi1[c], pi2[c]) / h, min(max(pi1[c], pi2[c]), 1 - h))`.

When `h` is effectively zero, the undefined normalization is omitted and the
disjunctive limit is used. Conflict is `1 - h`: zero means the evidence has a
fully compatible class; values approaching one indicate stronger conflict.

**Worked example** from the design tests: fuse
`pi1=(1, 0.3)` and `pi2=(0.5, 1)`. The classwise minima are `(0.5, 0.3)`, so
`h=0.5`, and the conflict is `0.5`. The normalized conjunction is `(1, 0.6)`;
the conflict-capped disjunction contributes `(0.5, 0.5)`. Taking the
classwise maximum gives adaptive output `(1, 0.6)`.

## Decisions and raster codes

Necessity for the leading class is `1 - runner_up_possibility`. In the worked
example, Burned leads with possibility 1 and Unburned is 0.6, so Burned
necessity is `0.4`. A tied distribution or necessity below the configured
threshold is **Undecided**. The output class codes are `0` Burned, `1`
Unburned, `2` Undecided, and `-1` NoData. NoData propagates rather than being
treated as zero or ignorance. The separate undecided mask uses `1` for
undecided, `0` for decided, and `255` for NoData.

## Checkpoint C validation and caveats

The pre-registered primary fused VV result was compared on 111,191 valid
AOI pixels with a 500 m spatial-block bootstrap (1,000 resamples). Against the
FireDelineation reference, fused IoU was `0.4962` (95% interval
`0.4328–0.5551`); optical-only IoU was `0.8233` (`0.7696–0.8737`), and
VV SAR-only IoU was `0.3777` (`0.3219–0.4281`). Paired fused-minus-optical
IoU was `-0.3271` (`-0.3994` to `-0.2622`); fused-minus-VV-SAR IoU was
`+0.1184` (`+0.0955` to `+0.1456`).

Interpretation is limited: this is one event and one acquisition pairing; the
reference is optical-derived and closely related to dNBR (Spearman `0.90`), so
it favors optical evidence and may understate SAR. The tested positive VV sign
was label-informed, and the SAR null result indicates an unusual background
contrast, not reliable burned-pixel detection. The synthetic cloud ablation
shows behavior under its registered scenario, not generalization to real
events. These results do not establish transferable defaults or universal
sensor performance.
