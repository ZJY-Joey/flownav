# FlowNav Goal-Condition Evaluation Scripts

This folder contains standalone evaluation scripts for checking whether a
pretrained FlowNav checkpoint actually uses the goal image condition.

The central question is:

> If the current observation is fixed but the goal image changes, does the
> predicted trajectory distribution change in a meaningful, goal-consistent way?

The scripts reuse the repository's existing `ViNT_Dataset` loading path. They do
not read raw HDF5 files directly. If `--checkpoint` is omitted, the checkpoint is
resolved from `logs/<load_run>/latest.pth` in `flownav/config/flownav.yaml`.

Chinese documentation: [README_CN.md](README_CN.md).

## Output Layout

By default, outputs are written to `test_logs/<dataset>/<script_name>/`.

Typical layout:

```text
test_logs/
  recon/
    goal_shuffle_quantitative/
    goal_swap_visualization/
      angle10-mmd0p5-emd0p2/
        all_samples/
          heading_filter/
          no_heading_filter/
        anomaly_samples/
          heading_filter/
          no_heading_filter/
    goal_inconsistent_rate/
    goal_mask_sensitivity/
    goal_separation_ratio/
```

For `goal_swap_visualization.py`, the threshold folder name encodes the run
settings:

- `angle10`: left/forward/right classes use a 10 degree goal-position threshold.
- `mmd0p5`: anomaly filtering uses MMD threshold 0.5.
- `emd0p2`: anomaly filtering uses sliced Wasserstein threshold 0.2 as an EMD approximation.

## Shared Matching Logic

`common.py` provides shared utilities for config loading, checkpoint loading,
dataloader construction, model inference, matched-goal search, and plotting.

For matched left/forward/right tests:

- The current observation is fixed.
- Goal images are selected from the same trajectory and same current time.
- Goals are not randomly taken from another batch or unrelated scene.
- Direction labels are assigned from the candidate `goal_pos` bearing in the
  current robot local frame.
- `left`: `goal_pos` angle is greater than `--angle-threshold-deg`.
- `right`: `goal_pos` angle is smaller than `- --angle-threshold-deg`.
- `forward`: absolute `goal_pos` angle is within the threshold.
- Heading filtering additionally checks the goal image time's local trajectory
  heading, avoiding goal images that clearly face the opposite side.

This constructs same-scene, same-observation, different-goal tests instead of
invalid random goal swaps.

## Main Metrics

| Metric | Meaning | Expected if Goal Conditioning Works |
| --- | --- | --- |
| `goal_inconsistent_rate` | Fraction of sampled trajectories whose final direction differs from the GT final direction by more than the threshold. | Lower is better. |
| `mean_angle_diff_deg` | Mean angular error between sampled final direction and GT final direction. | Lower is better. |
| `mean_endpoint_mean_distance` | Mean distance between endpoint means under left/forward/right goals. | Higher means goal changes move the endpoint cluster. |
| `mean_endpoint_rbf_mmd` | Mean RBF-MMD between endpoint distributions under different goals. | Higher means distributions are more separable. |
| `mean_endpoint_sliced_wasserstein` | Sliced Wasserstein distance, used as an EMD approximation. | Higher means endpoint distributions differ more. |
| `mean_s_goal` | Endpoint mean separation normalized by goal-position separation. | Higher means trajectory change tracks goal change better. |
| `mean_traj_dtw` | DTW distance between mean trajectories. | Higher means whole-trajectory shape differs more. |
| `mean_traj_frechet` | Frechet distance between mean trajectories. | Higher means trajectory curves differ more. |
| `mean_goal_separation_ratio` | Inter-goal endpoint separation divided by within-goal spread. | Higher means left/forward/right clusters are cleaner. |

Low MMD/EMD values are suspicious when the selected goals are clearly different:
they indicate that the model may generate nearly the same trajectory
distribution regardless of the goal image.

## 1. Goal Shuffle Quantitative

File: `goal_shuffle_quantitative.py`

### What It Does

This test compares model outputs under the correct goal and a matched
alternative goal. The alternative goal is selected from the same trajectory and
same current time, so it remains scene-compatible.

Procedure:

1. Run the model with the correct goal.
2. Select a same-scene alternative goal with a different direction.
3. Run the model again with that alternative goal.
4. Compare both predictions against the dataset action label.

### Key Outputs

- Summary JSON with correct-goal and alternative-goal metrics.
- Optional three-column visualization for one matched left/forward/right sample.
- Progress logs for batch processing and matched-goal search.

### Key Metrics

- `action_loss_mse`
- `ade`
- `fde`
- `waypoint_cos_sim`
- `trajectory_cos_sim`
- `success_rate_fde`

### Run

```bash
python test_scripts/goal_shuffle_quantitative.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --max-batches 20 \
  --num-samples 8 \
  --min-alternative-angle-diff-deg 10 \
  --max-alternative-angle-diff-deg 90 \
  --max-direction-angle-deg 90 \
  --visualization-samples 16 \
  --output-dir test_logs
```

### What To Inspect

- If correct-goal and alternative-goal metrics are almost identical, the model
  may be weakly conditioned on the goal.
- If alternative goals are too extreme, lower `--max-alternative-angle-diff-deg`.
- If too few matches are found, increase scan batches or relax angle limits.

## 2. Goal Swap Visualization

File: `goal_swap_visualization.py`

### What It Does

This is the main same-scene left/forward/right goal sensitivity test.

For each valid matched sample:

1. Keep the current observation fixed.
2. Select left, forward, and right future goal images from the same trajectory
   and same current time.
3. Sample multiple goal-conditioned trajectories for each goal.
4. Compare endpoint distributions, mean trajectories, headings, DTW/Frechet
   distances, MMD, and sliced Wasserstein distance.
5. Mark a sample as anomalous when different goals produce overly similar
   endpoint distributions.

The script automatically runs two settings:

- `heading_filter`: goal image heading must be compatible with the selected
  left/forward/right class.
- `no_heading_filter`: only `goal_pos` direction is used for class selection.

### Output Structure

```text
test_logs/recon/goal_swap_visualization/angle10-mmd0p5-emd0p2/
  all_samples/
    heading_filter/
      goal_swap_global_endpoints_*.png
      goal_swap_sensitivity_overview_*.png
      goal_swap_visualization_summary_*.json
    no_heading_filter/
      ...
  anomaly_samples/
    heading_filter/
      anomaly_indices.txt
      goal_swap_anomaly_global_endpoints_*.png
      anomaly_00000_*.png
      anomaly_00000_*_endpoints.png
      anomaly_00000_*.json
      goal_swap_visualization_summary_*.json
    no_heading_filter/
      ...
  goal_swap_all_samples_heading_filter_endpoint_comparison_*.png
  goal_swap_all_samples_heading_filter_comparison_*.json
```

### Important Outputs

- `all_samples/`
  - Runs on all matched samples.
  - Does not save per-sample PNG/JSON.
  - Saves global endpoint/goal distributions and summary JSON.

- `anomaly_samples/`
  - Runs anomaly filtering using MMD/EMD thresholds.
  - Saves anomalous sample visualizations, endpoint plots, JSON metadata, and
    `anomaly_indices.txt`.

- `goal_swap_global_endpoints_*.png`
  - Left panel: sampled endpoint distributions for left/forward/right goals.
  - Right panel: matched `goal_pos` distributions for left/forward/right goals.
  - Both panels annotate pairwise MMD and EMD approximation.

- `goal_swap_anomaly_global_endpoints_*.png`
  - Same plot, restricted to anomalous samples.

- `goal_swap_*_heading_filter_endpoint_comparison_*.png`
  - Parent-folder comparison using the full `all_samples` endpoint
    distributions.
  - Uses different marker/color styles for heading and no-heading points.
  - Annotates MMD and EMD approximation between the two settings.
  - If a class has more than 10000 points, the figure is downsampled for
    display only.

- `anomaly_*.png`
  - Three columns: left, forward, and right goal.
  - Same current observation in every column.
  - BEV includes sampled trajectories, robot position, goal position, current
    heading, and goal-image-time heading in the current robot frame.

### Key Metrics

- `mean_endpoint_rbf_mmd`
- `mean_endpoint_sliced_wasserstein`
- `mean_endpoint_mean_distance`
- `mean_s_goal`
- `mean_traj_dtw`
- `mean_traj_frechet`
- `class_tv_distance`
- `endpoint_symmetric_kl`

### Anomaly Rule

A matched sample is marked anomalous if:

```text
mean_endpoint_rbf_mmd <= --anomaly-mmd-threshold
or
mean_endpoint_sliced_wasserstein <= --anomaly-emd-threshold
```

In other words, if left/forward/right goals produce endpoint distributions that
are too similar, the sample is treated as evidence of low goal sensitivity.

### Run

```bash
python test_scripts/goal_swap_visualization.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --scan-batches 200 \
  --num-samples 8 \
  --angle-threshold-deg 10 \
  --max-direction-angle-deg 90 \
  --anomaly-mmd-threshold 0.5 \
  --anomaly-emd-threshold 0.2 \
  --global-endpoint-max-points-per-class 10000 \
  --output-dir test_logs
```

### What To Inspect

- If goal-position distributions are well separated but endpoint distributions
  overlap heavily, the model is seeing distinct goals but not responding
  strongly.
- If goal-position distributions also overlap, matching may be too weak or the
  direction threshold may be too small.
- If `heading_filter` and `no_heading_filter` differ strongly, goal image
  viewpoint consistency materially affects the test.
- If too few anomalies are found, increase `--anomaly-mmd-threshold` or
  `--anomaly-emd-threshold`.
- If too many anomalies are found, lower those thresholds.
- If global plots are too dense, lower `--global-endpoint-max-points-per-class`.
  This only affects plotted points; metrics still use all points.

## 3. Goal-Inconsistent Rate

File: `goal_inconsistent_rate.py`

### What It Does

This test samples N trajectories with the correct goal. It uses the
ground-truth action's final waypoint direction as the target direction and
compares each sampled trajectory's final waypoint direction against it.

If the angular difference exceeds `--angle-threshold-deg`, that sampled
trajectory is counted as goal-direction inconsistent.

### Key Metrics

- `goal_inconsistent_rate`
- `mean_angle_diff_deg`
- `median_angle_diff_deg`
- `p90_angle_diff_deg`
- `p95_angle_diff_deg`

### Run

```bash
python test_scripts/goal_inconsistent_rate.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --max-batches 20 \
  --num-samples 8 \
  --angle-threshold-deg 45 \
  --output-dir test_logs
```

### What To Inspect

- High `goal_inconsistent_rate` means many sampled trajectories point away from
  the GT final direction.
- High `p90_angle_diff_deg` or `p95_angle_diff_deg` means a tail of samples has
  severe directional errors.
- Output filenames include the angle threshold, so different GIR thresholds can
  be compared safely.

## 4. Goal Mask Sensitivity

File: `goal_mask_sensitivity.py`

### What It Does

This test compares trajectories sampled with the original goal image against
trajectories sampled with the goal masked. It checks whether removing goal
information changes the output distribution.

### Key Metrics

- Endpoint mean distance between goal-conditioned and goal-masked outputs.
- Endpoint Chamfer distance between endpoint clouds.
- Matched-sample ADE/FDE between goal-conditioned and masked samples.

### Run

```bash
python test_scripts/goal_mask_sensitivity.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --max-batches 20 \
  --num-samples 16 \
  --output-dir test_logs
```

### What To Inspect

- If masked and unmasked endpoint distributions are almost identical, the model
  may not rely strongly on the goal image.
- If masked outputs degrade substantially, goal conditioning is likely being
  used.

## 5. Goal Separation Ratio

File: `goal_separation_ratio.py`

### What It Does

This test builds many hard left/forward/right triplets from the same trajectory
and same current time. For each goal class, it samples multiple trajectories,
computes endpoint spread within that class, and compares it with endpoint
separation between classes.

The intended signal is:

```text
GSR = between-goal separation / within-goal dispersion
```

### Key Metrics

- `mean_inter_goal_distance`
- `mean_within_goal_dispersion`
- `mean_goal_separation_ratio`
- `median_goal_separation_ratio`
- `p10_goal_separation_ratio`
- `p90_goal_separation_ratio`

### Run

```bash
python test_scripts/goal_separation_ratio.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --scan-batches 50 \
  --max-triplets 100 \
  --num-samples 16 \
  --angle-threshold-deg 25 \
  --max-direction-angle-deg 90 \
  --output-dir test_logs
```

### What To Inspect

- Higher GSR means left/forward/right endpoint clusters are better separated.
- Low GSR means different goal conditions collapse into overlapping trajectory
  clusters.
- If within-goal dispersion is huge, model sampling may be unstable.
- If between-goal separation is tiny, the model may be insensitive to goal
  changes.

## Common Options

Use a specific checkpoint:

```bash
--checkpoint logs/flownav0421/latest.pth
```

Use a specific device:

```bash
--device cuda:0
```

Reduce runtime for smoke tests:

```bash
--batch-size 8 --max-batches 1 --num-samples 4
```

Limit visualization-heavy runs:

```bash
--max-visualizations 20
```

Control left/forward/right matching strictness:

```bash
--angle-threshold-deg 10 --max-direction-angle-deg 90
```

## Summary Figure Generation

File: `generate_summary_figures.py`

This script does not run the model. It reads existing `test_logs` outputs and
creates paper-style summary figures under `test_logs/summary_figure/`.

Run:

```bash
python3 test_scripts/generate_summary_figures.py --log-root test_logs
```

Outputs:

- `fig1_quantitative_summary.png`
  - Multi-panel quantitative summary.
  - Panel A: mean EMD approximation for heading-filter vs no-heading-filter.
  - Panel B: mean MMD for heading-filter vs no-heading-filter.
  - Panel C: matched sample count plus heading-filter retention ratio.
  - Panel D: raw anomaly counts.

- `fig2_global_endpoint_angle10.png`
  - A 3 x 2 collage for `angle=10`.
  - Rows are datasets.
  - Columns are no-heading-filter and heading-filter.
  - Each subplot reuses the existing global endpoint / matched goal-position
    distribution image.

- `fig2_global_endpoint_angle15.png`
  - Same as Figure 2, but for `angle=15`.

- `fig2_all_vs_anomaly_angle10.png`
  - A 3 x 2 collage for `angle=10`.
  - Rows are datasets.
  - Columns are `all samples` and `anomaly samples`.
  - Uses the `no-heading-filter` setting only.

- `fig2_all_vs_anomaly_angle15.png`
  - Same as above, but for `angle=15`.

- `fig3_hard_case_gallery_angle10.png`
  - Diverse hard-case gallery for `angle=10`.
  - Each dataset contributes multiple anomaly cases.
  - Each case shows two images: the swap visualization and its endpoint
    distribution.
  - Cases are selected to be far apart in `dataset_index` when possible, so the
    gallery is not dominated by near-duplicate frames.
  - If one dataset has fewer available anomaly cases than requested, the figure
    automatically uses the maximum shared count available for that angle.

- `fig3_hard_case_gallery_angle15.png`
  - Same as Figure 3, but for `angle=15`.

- `fig4_anomaly_summary.png`
  - Hard-case anomaly count and descriptive anomaly ratio summary.
  - Ratios should be interpreted cautiously because heading-filter and
    no-heading-filter use different matched sample pools.

- `fig5_paired_improvement.png`
  - Strict paired comparison using only matched samples present in both
    heading-filter and no-heading-filter runs.
  - Reports paired EMD, paired MMD, deltas, and paired sample count.
  - This is the fairest plot for judging whether heading filtering improves
    metrics without changing the evaluated sample set.

- `table1_summary.csv` and `table1_summary.md`
  - Multi-dataset summary table with matched counts, MMD, EMD, anomaly counts,
    deltas, and heading-filter retention.

- `table2_paired_improvement.csv` and `table2_paired_improvement.md`
  - Numeric table backing `fig5_paired_improvement.png`.

Useful options:

```bash
python3 test_scripts/generate_summary_figures.py \
  --log-root test_logs \
  --hard-cases-per-dataset 5 \
  --hard-case-min-index-gap 500
```

- `--hard-cases-per-dataset` controls how many hard cases are shown per dataset.
- `--hard-case-min-index-gap` controls how far apart selected cases should be
  in dataset index before the script relaxes the gap.

## Troubleshooting Checklist

- No output for a long time:
  model loading, dataset scanning, and sampling can take time. First run a smoke
  test with smaller `--scan-batches`, `--max-batches`, or `--num-samples`.

- `Could not find a same-trajectory/same-time sample`:
  increase `--scan-batches`, lower `--angle-threshold-deg`, or relax
  `--max-direction-angle-deg`.

- Too few anomalies:
  increase `--anomaly-mmd-threshold` or `--anomaly-emd-threshold`.

- Too many anomalies:
  decrease anomaly thresholds and check whether the goal-position distributions
  are themselves poorly separated.

- Endpoint plot has too many points:
  lower `--global-endpoint-max-points-per-class`. Metrics still use all points;
  this only changes visualization density.

- Goal image appears visually inconsistent with its left/forward/right label:
  compare `heading_filter` against `no_heading_filter`. If the difference is
  large, goal image viewpoint filtering is important for this dataset.

- `python` command not found:
  activate the project conda environment, or use `python3` if that is how the
  environment exposes Python.
