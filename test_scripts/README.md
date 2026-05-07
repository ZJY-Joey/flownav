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

## Maintenance Status

Current active scope:

- `goal_swap_visualization.py`
- `dist_head_backfill.py`
- `topomap_subgoal_analysis.py`
- `recon_head_horizon_summary.py`
- `goal_mask_sensitivity.py`
- `generate_summary_figures.py`
- `common.py`, only for utilities used by the active scripts

The other test scripts are kept in the repository for historical reference, but
they are temporarily archived and no longer actively synchronized with the
current goal-swap evaluation workflow:

- `goal_shuffle_quantitative.py`
- `goal_inconsistent_rate.py`
- `goal_separation_ratio.py`

Use these archived scripts only if you have checked their assumptions and
outputs manually. New changes should target the active goal-swap and
goal-mask-sensitivity pipelines unless there is an explicit reason to revive one
of the archived tests.

## Output Layout

By default, baseline outputs are written to
`test_logs/flownav_baseline/<dataset>/<script_name>/`.

When `goal_swap_visualization.py` is run with
`--trajectory-selection cluster`, outputs are written to
`test_logs/flownav_cluster/<dataset>/<script_name>/`.

Typical layout:

```text
test_logs/
  flownav_baseline/
    recon/
      goal_swap_visualization/
        angle10-mmd0p5-emd0p2/
          all_samples/
            heading_filter/
            no_heading_filter/
          anomaly_samples/
            heading_filter/
            no_heading_filter/
    summary_figure/
  flownav_cluster/
    recon/
      goal_swap_visualization/
        angle10-mmd0p5-emd0p2/
          all_samples/
          anomaly_samples/
    summary_figure/
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

## Goal Images, Subgoals, and Flow-Head Inputs

The goal condition is not the same object in training, offline tests, and
deployment:

| Context | Flow-head condition |
| --- | --- |
| Normal training | A future `goal image` sampled by the dataset. There is no subgoal-selection step. |
| `goal_swap_visualization.py` | Matched left/forward/right future `goal image`s from the same trajectory and same `curr_time`. These are not subgoals. |
| `goal_mask_sensitivity.py` | The with-goal branch uses matched future `goal image`s; the masked branch uses the goal-mask path. These are not subgoals. |
| File-based inference, e.g. `infer_rgb.py` / `benchmark_flownav.py` | The `goal image` explicitly provided by the command or data. |
| Deployment topomap navigation | `dist_pred_net` scores candidate node images in a local topomap window, then the selected local topomap subgoal image is used as the flow-head goal condition. |

The model does not directly consume `(x, y)` goal poses. `goal_pos` /
`goal_pos_metric` are geometric quantities computed by the test scripts from
the trajectory, used for left/forward/right classification, horizon bucketing,
plotting, and metrics.

The deployment-style local subgoal rule is:

```python
closest_idx = argmin(dist_pred(topomap_window_images))
subgoal_idx = closest_idx + int(dists[closest_idx] < close_threshold)
```

`close_threshold` is in the distance-head prediction units, not necessarily
meters. For `recon`, `metric_waypoint_spacing=0.25`; with `waypoint_spacing=1`
and frequent advancement to the next topomap node, the mean selected local
subgoal distance can be near `0.25m`.

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

## Archived: Goal Shuffle Quantitative

File: `goal_shuffle_quantitative.py`

Status: temporarily archived. This script is not part of the currently
maintained evaluation workflow.

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
  --output-dir test_logs/flownav_baseline
```

### What To Inspect

- If correct-goal and alternative-goal metrics are almost identical, the model
  may be weakly conditioned on the goal.
- If alternative goals are too extreme, lower `--max-alternative-angle-diff-deg`.
- If too few matches are found, increase scan batches or relax angle limits.

## Active: Goal Swap Visualization

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

By default, the script runs two settings. Use `--heading-filter-mode` to run
only one setting:

- `heading_filter`: goal image heading must be compatible with the selected
  left/forward/right class.
- `no_heading_filter`: only `goal_pos` direction is used for class selection.
- `both`: run both settings and write the parent-folder heading-filter
  comparison.

Trajectory selection modes:

- `baseline`: keeps the sampled FlowNav trajectories and preserves the previous
  goal-swap behavior.
- `cluster`: samples multiple trajectories per goal, clusters them with the
  shared FlowNav weighted trajectory distance, and uses the medoid of the
  largest cluster for metric/log generation. The selected trajectory is repeated
  internally so distribution metrics keep the same shape as the baseline path.

### Output Structure

```text
test_logs/flownav_baseline/recon/goal_swap_visualization/angle10-mmd0p5-emd0p2/
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
  - Horizon-filtered runs skip this stage automatically when any
    `--min/--max-goal-offset` or `--min/--max-goal-pos-dist` argument is set.
    This keeps horizon sweeps focused on the full matched population.
  - Horizon-filtered runs also allow partial directional sets. If a matched
    observation has only one or two of left/forward/right, the script samples
    the available directions and computes metrics only for available pairs.
    Regular non-horizon `test_logs` swap runs still require all three directions.
  - If a horizon-filtered bucket has no directional goals at all, the script
    writes an empty summary with `scan_error` instead of aborting the sweep.

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
  --heading-filter-mode no_heading_filter \
  --global-endpoint-max-points-per-class 10000 \
  --global-metric-max-points-per-class 5000 \
  --output-dir test_logs
```

For full-population runs that should not write anomaly outputs, add:

```bash
--skip-anomaly-stage
```

Run the clustered trajectory-selection variant:

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
  --trajectory-selection cluster \
  --cluster-threshold 0.35 \
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
  This only affects plotted points.
- `--global-metric-max-points-per-class` caps the points per direction used for
  global pairwise MMD/EMD. The default is 5000 to avoid O(N^2) memory blowups;
  per-sample metrics still use all sampled trajectories.

## Archived: Goal-Inconsistent Rate

File: `goal_inconsistent_rate.py`

Status: temporarily archived. This script is not part of the currently
maintained evaluation workflow.

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
  --output-dir test_logs/flownav_baseline
```

### What To Inspect

- High `goal_inconsistent_rate` means many sampled trajectories point away from
  the GT final direction.
- High `p90_angle_diff_deg` or `p95_angle_diff_deg` means a tail of samples has
  severe directional errors.
- Output filenames include the angle threshold, so different GIR thresholds can
  be compared safely.

## Active: Goal Mask Sensitivity

Files:

- `goal_mask_sensitivity.py`

### What It Does

This test compares no-heading-filter left/forward/right goal-conditioned
trajectories against trajectories sampled with the goal masked.

"With goal" means the same-scene matched goals used by goal swap: left,
forward, and right future goal images are selected from the same trajectory and
same current time, with `filter_goal_heading=False`. "Masked" means the same
observation and goal image are passed through the model's goal-mask path, so the
trajectory is generated without goal conditioning.

### Key Metrics

- `endpoint_mean_l2`: distance between goal-conditioned and masked endpoint means.
- `endpoint_rbf_mmd`: RBF-MMD between goal-conditioned and masked endpoints.
- `endpoint_sliced_wasserstein`: sliced Wasserstein distance, used as an EMD approximation.
- `matched_sample_ade` / `matched_sample_fde`: paired trajectory differences between goal-conditioned and masked samples.
- Direction-pair MMD/EMD for the with-goal endpoints and for the masked endpoints.

### Quantitative Run

```bash
python test_scripts/goal_mask_sensitivity.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --scan-batches 200 \
  --num-samples 16 \
  --angle-threshold-deg 10 \
  --max-direction-angle-deg 90 \
  --output-dir test_logs
```

Run the clustered trajectory-selection variant:

```bash
python test_scripts/goal_mask_sensitivity.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --batch-size 64 \
  --scan-batches 200 \
  --num-samples 16 \
  --angle-threshold-deg 10 \
  --max-direction-angle-deg 90 \
  --trajectory-selection cluster \
  --cluster-threshold 0.35 \
  --output-dir test_logs
```

Outputs:

```text
test_logs/flownav_baseline/recon/goal_mask_sensitivity/angle10-no_heading_filter/
  goal_mask_sensitivity_summary_*.json
  goal_mask_sensitivity_items_*.csv
  goal_mask_sensitivity_endpoints_*.npz
```

### Visualization Run

`goal_mask_sensitivity.py` generates visualization figures during the same run
that produces the JSON/CSV/NPZ outputs. Figures are saved in the same
`goal_mask_sensitivity/<run_tag>/` folder as the metrics.

Main visualization outputs:

- `goal_mask_endpoint_shift_by_direction_*.png`
  - Three panels: left, forward, right.
  - Each panel overlays goal-conditioned endpoints and masked-goal endpoints.
  - Each panel annotates goal-vs-masked MMD, EMD approximation, and endpoint mean shift.

- `goal_mask_direction_distribution_comparison_*.png`
  - Left panel: with-goal endpoint distributions for left/forward/right goals.
  - Middle panel: masked-goal endpoint distributions grouped by the requested goal class.
  - Right panel: matched goal-position distributions.
  - The first two panels annotate direction-pair MMD and EMD approximation.

### What To Inspect

- If with-goal left/forward/right endpoints separate but masked endpoints collapse, the model is using the goal.
- If with-goal and masked endpoints are nearly identical, goal conditioning is weak.
- If masked endpoints still separate by left/forward/right, check whether the observation alone already determines the route or whether matched goals are biased by trajectory position.

## Archived: Goal Separation Ratio

File: `goal_separation_ratio.py`

Status: temporarily archived. This script is not part of the currently
maintained evaluation workflow.

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
  --output-dir test_logs/flownav_baseline
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

Control the local metric distance of matched goals:

```bash
--min-goal-pos-dist 8 --max-goal-pos-dist 12
```

This filters candidates by `norm(goal_pos[:2])` in the current robot frame. It
is different from `--min-goal-offset/--max-goal-offset`, which filters by future
trajectory index/time offset rather than meters.

Any run with `--min/--max-goal-offset` or `--min/--max-goal-pos-dist` is treated
as a horizon-filtered run by `goal_swap_visualization.py`; it writes only the
`all_samples` stage, skips anomaly testing, and does not require all
left/forward/right directions to be present in the same matched observation.

## Head-Level Utilities

## Dist-Head Backfill

File: `dist_head_backfill.py`

This script reuses existing `goal_swap_visualization_summary_*.json` matched
samples and runs only the vision encoder plus `dist_pred_net`. It does not run
the flow head or diffusion sampling again.

For each matched observation with left/forward/right goals, it writes
`dist_head_backfill_summary_*.json` and `dist_head_backfill_items_*.csv` next to
the source goal-swap summary. Before writing, it removes old
`dist_head_backfill_summary_*.json` and `dist_head_backfill_items_*.csv` files in
that same output directory so downstream summary scripts read only the latest
backfill. It does not remove the original goal-swap summaries, images, or other
logs.

The main metrics are:

- `dist_pred_pair_l1` / `dist_pred_pair_l2`: mean absolute / L2 difference
  between `dist_pred` outputs for different matched goal images under the same
  observation. This measures dist-head response magnitude to goal-image changes;
  it is not an error against GT distance.
- `goal_pos_pair_distance`
- `dist_pred_goal_normalized_sensitivity`
- `dist_pred_rank_accuracy`
- `dist_pred_goal_offset_spearman`
- `flow_endpoint_pair_distance`, copied from the existing goal-swap summary
- `flow_goal_normalized_sensitivity`
- `flow_vs_dist_goal_normalized_ratio`
- `flow_goal_direction_alignment`, available for goal-swap summaries generated
  after endpoint mean logging was added

Run for all baseline all-sample goal-swap summaries:

```bash
python3 test_scripts/dist_head_backfill.py \
  --log-root test_logs \
  --variant flownav_baseline \
  --keep-going
```

Run for one existing summary:

```bash
python3 test_scripts/dist_head_backfill.py \
  --summary-path test_logs/flownav_baseline/recon/goal_swap_visualization/angle10-mmd0p2-emd1/all_samples/no_heading_filter/goal_swap_visualization_summary_YYYYMMDD_HHMMSS.json
```

## Focused Head/Goal Response Figures

File: `generate_head_horizon_figures.py`

This script generates focused figures for the head-level goal-response
analysis:

- `fig_dist_pred_by_goal_pos_dist.png`
  - One row with three dataset panels.
  - Each panel plots per-direction `dist_pred` values against local metric goal
    distance `norm(goal_pos[:2])` from the dist-head backfill rows.
  - Use this figure to judge whether `dist_pred` changes with local goal
    distance and whether left/forward/right directions occupy different value
    ranges.

- `fig_endpoint_goal_distribution_by_horizon.png`
  - A dataset x horizon-bucket collage.
  - Rows are datasets; columns are short, mid, and long local goal-distance buckets.
  - Each panel reuses the corresponding horizon swap global endpoint figure,
    which shows endpoint distributions and matched goal-position distributions.
  - Use this figure to inspect whether flow endpoints move with goal offset and
    whether endpoint clusters track the goal-position clusters.

- `fig_endpoint_mmd_emd_by_horizon.png`
  - Two-panel line chart across short, mid, and long local goal-distance
    buckets.
  - Each dataset is one line.
  - Panel A shows mean endpoint RBF-MMD; panel B shows mean endpoint sliced
    Wasserstein.
  - Use this figure to inspect whether endpoint distribution separability
    decreases or changes with longer goal distance.

- `fig_flow_vs_dist_sensitivity_by_horizon.png`
  - Raw response-magnitude comparison. Flow uses endpoint pair distance; dist
    uses `dist_pred` pair difference. These are not in the same units.

- `fig_goal_normalized_flow_vs_dist_sensitivity_by_horizon.png`
  - Goal-space calibrated comparison.
  - Flow sensitivity is endpoint pair distance divided by goal-pose pair
    distance.
  - Dist sensitivity is `dist_pred` pair difference divided by goal-pose pair
    distance.
  - The ratio panel compares those two normalized sensitivities.

It also writes CSV tables for inspection:

- `head_level_summary.csv`
- `dist_pred_goal_pos_dist_items.csv`
- `horizon_mask_summary.csv`
- `horizon_swap_summary.csv`
- `horizon_head_level_summary.csv`
- `endpoint_distribution_images.csv`
- `missing_or_counts.json`

Run:

```bash
python3 test_scripts/generate_head_horizon_figures.py \
  --head-log-root test_logs \
  --horizon-root test_logs_horizon \
  --variant flownav_baseline \
  --output-dir test_logs_horizon/flownav_baseline/head_horizon_summary
```

## Topomap Subgoal Analysis

File: `topomap_subgoal_analysis.py`

This script tests whether different final goals collapse to the same local
topomap subgoal. It uses the same trajectory as the offline topomap and
replays the deployment-style local topomap selection:

1. Build directional final goals in local metric distance buckets: short
   `0-7m`, mid `8-12m`, long `13-19m`. A cell does not require all
   left/forward/right directions to exist; available directions are plotted.
2. For each fixed observation and each final goal, score topomap nodes in a
   local window with `dist_pred_net`.
3. Select the closest node and the next local subgoal using the deployment
   `close_threshold` rule.
4. Aggregate selected subgoals, final goal poses, and available flow endpoint
   means.

Per-cell panel PNGs are written under each dataset/bucket folder:

```text
test_logs_horizon/topomap_subgoal_analysis/<bucket>/flownav_baseline/<dataset>/
```

- `panel_topomap_subgoal_vs_goal_pose.png`
- `panel_flow_endpoint_vs_topomap_subgoal.png`

The final summary figures under
`test_logs_horizon/topomap_subgoal_analysis/flownav_baseline/summary/` are
montages that only stitch existing PNGs:

- `fig_topomap_subgoal_vs_goal_pose.png`
  - Dataset x metric-distance bucket collage.
  - Each cell has selected subgoal distribution on the left and final goal-pose
    distribution on the right.
  - Left/forward/right are colored separately and outlined with covariance
    circles when that direction exists.

- `fig_flow_endpoint_vs_topomap_subgoal.png`
  - Dataset x metric-distance bucket collage.
  - Each cell has flow endpoint-mean distribution on the left and selected
    subgoal distribution on the right.
  - Flow endpoint panels require matching metric-distance `goal_swap` logs under
    `--swap-log-root`.

- `fig_flow_endpoint_vs_goal_pose.png`
  - Dataset x metric-distance bucket collage.
  - Each cell is the existing `goal_swap_global_endpoints_*.png` for that
    dataset and bucket, so it shows flow endpoint distribution and matched goal
    position distribution without redrawing points.

- `fig_endpoint_mmd_emd_by_goal_distance.png`
  - Line chart over short, mid, and long buckets.
  - Panel A is mean endpoint RBF-MMD; panel B is mean endpoint sliced
    Wasserstein.
  - This is read from existing horizon goal-swap summaries.

- `fig_final_goal_dist_pred_error_by_goal_distance.png`
  - Line chart over short, mid, and long buckets.
  - Compares mean final-goal `dist_pred` with mean true local goal distance,
    and plots their mean difference.
  - This requires rerunning `topomap_subgoal_analysis.py`, because
    `final_goal_dist_pred` is recorded during topomap/dist-head inference.

Run after metric-distance goal-swap logs have been generated:

```bash
python3 test_scripts/topomap_subgoal_analysis.py \
  --output-dir test_logs_horizon \
  --swap-log-root test_logs_horizon \
  --scan-batches 200 \
  --batch-size 64 \
  --angle-threshold-deg 10
```

To only rebuild the three summary montages from existing PNGs, without loading
the model or rerunning inference:

```bash
python3 test_scripts/topomap_subgoal_analysis.py \
  --output-dir test_logs_horizon \
  --swap-log-root test_logs_horizon \
  --montage-only
```

## Recon Head Horizon Summary

File: `recon_head_horizon_summary.py`

This is a focused `recon`-only experiment for jointly inspecting final goal
poses, deployment-style local topomap subgoals, and optional
subgoal-conditioned flow endpoints.

Default output directory:

```text
test_logs_horizon/flownav_baseline/head_horizon_summary/
```

Core definitions:

- Direction is assigned by the final goal pose bearing in the current robot
  local frame: `left > 10 deg`, `forward = [-10 deg, 10 deg]`,
  `right < -10 deg`.
- Horizon buckets use Euclidean final-goal distance in the current robot local
  frame: `short=[0,2)m`, `mid=[2,4)m`, `long=[4,max_offset_range]`.
- For the same observation and same direction, the script selects one short,
  one mid, and one long final goal image.
- The local topomap subgoal is selected with the deployment-style dist-head
  rule. It is not a geometric interpolation from the final goal pose.

Default run computes dist-head/local-subgoal outputs only and does not run the
flow head:

```bash
python3 test_scripts/recon_head_horizon_summary.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test
```

Main outputs:

- `fig_recon_head_horizon_distribution.png`
  - 1x3 figure with left / forward / right columns.
  - Overlays final goal-pose and selected local subgoal distributions.
  - Uses different colors for short / mid / long and annotates subgoal-vs-goal
    MMD and EMD approximation.

- `fig_recon_head_horizon_local_subgoals.png`
  - 1x3 figure with only selected local topomap subgoals.
  - Use it to check whether different final-goal buckets collapse to the same
    local subgoal.

- `fig_recon_head_horizon_metrics.png`
  - 2x2 figure.
  - Top row: raw MMD and EMD approximation between subgoals and final goal
    poses.
  - Bottom left: mean selected-subgoal distance to the robot.
  - Bottom right: mean final-goal-pose distance to the robot.

- `recon_head_horizon_items_*.csv`
  - Per observation / direction / bucket selected goal, selected subgoal,
    dist-head prediction, and topomap window information.

- `recon_head_horizon_metrics_*.csv`
  - Per direction / bucket distribution metrics and mean distances.

To force the flow head to use the selected local subgoal image as its goal
condition, explicitly enable GC-only flow sampling:

```bash
python3 test_scripts/recon_head_horizon_summary.py \
  --config flownav/config/flownav.yaml \
  --dataset recon \
  --split test \
  --num-flow-samples 8
```

Additional outputs:

- `fig_recon_head_horizon_subgoal_conditioned_flow.png`
  - 3x2 figure.
  - Rows are left / forward / right.
  - Left column: GC flow endpoint distribution conditioned on the selected
    subgoal image.
  - Right column: corresponding selected subgoal distribution.

- `recon_head_horizon_subgoal_flow_endpoints_*.csv`
  - Endpoint rows for every subgoal-conditioned GC sample.

Note: `--num-flow-samples` defaults to `0` because flow ODE sampling for every
selected subgoal is expensive. Enable it only when the flow figure is needed.
The implementation runs only the GC branch, not the UC branch.

## Summary Figure Generation

File: `generate_summary_figures.py`

This script does not run the model. It reads existing `test_logs` outputs and
creates paper-style summary figures under
`test_logs/<variant>/summary_figure/`.

The script is tolerant of incomplete experiments. If one dataset, angle, stage,
or setting is missing, it still writes the figures that can be produced and
marks missing image panels as `missing`. It also writes:

- `missing_summary_inputs.json`
- `missing_summary_inputs.md`

These files list which swap or mask comparison inputs were unavailable.

Run:

```bash
python3 test_scripts/generate_summary_figures.py \
  --log-root test_logs \
  --variant flownav_baseline
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

- `fig6_goal_mask_direction_distribution_comparison.png`
  - Cross-dataset collage of each dataset's
    `goal_mask_direction_distribution_comparison_*.png`.

- `fig6_goal_mask_mmd_emd_delta.png`
  - Cross-dataset comparison of with-goal minus masked-goal direction-pair MMD/EMD deltas.
  - Bar annotations include the delta and the raw with-goal / masked-goal values.

- `table1_summary.csv` and `table1_summary.md`
  - Multi-dataset summary table with matched counts, MMD, EMD, anomaly counts,
    deltas, and heading-filter retention.

- `table2_paired_improvement.csv` and `table2_paired_improvement.md`
  - Numeric table backing `fig5_paired_improvement.png`.

- `missing_summary_inputs.json` and `missing_summary_inputs.md`
  - Missing dataset / angle / setting records that prevented a full comparison.

Useful options:

```bash
python3 test_scripts/generate_summary_figures.py \
  --log-root test_logs \
  --variant flownav_cluster \
  --hard-cases-per-dataset 5 \
  --hard-case-min-index-gap 500
```

- `--variant` chooses which log namespace to summarize:
  `flownav_baseline` or `flownav_cluster`.
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
