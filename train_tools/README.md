# Train Tools

This folder contains utility scripts related to training data inspection and
preparation. Keep this README and `README_CN.md` updated whenever a script in
`train_tools/` is added, changed, deprecated, or removed.

## Script Status

| Script | Status | Purpose |
| --- | --- | --- |
| `dataset_duration.py` | Maintained | Count trajectory steps from configured FlowNav datasets and estimate dataset duration. |

Status meanings:

- Maintained: expected to work with the current repository layout.
- Deprecated: kept for reference or old workflows; avoid using it for new runs.

## `dataset_duration.py`

Purpose:

- Read dataset definitions from `flownav/config/flownav.yaml`.
- Scan each configured `train` / `test` split's `traj_names.txt`.
- Prefer the split's cached FlowNav index file, `dataset_dist_...pkl`, when it
  exists. The cached `goals_index` already contains one entry per trajectory
  time step.
- Open each trajectory's `traj_data.pkl`.
- Count trajectory steps from `position` first, or `yaw` as a fallback.
- Compute duration from timestamp fields when present.
- If timestamps are absent, compute duration as `total_steps / Hz / 3600`.
- Print each dataset's split rows, each dataset total row, and an `ALL total`
  row across selected datasets.
- Print a `Duration summary` block before the table, listing each selected
  dataset's total hours and the overall `ALL` hours.
- Write the report to `test_tools_logs/datasets_statistics.txt` by default.
- Remove the previous output file before writing a new report, so stale content
  is not kept when regenerating statistics. The new report is written through a
  temporary file and atomically moved into place.

Default command:

```bash
python3 train_tools/dataset_duration.py --hz 4
```

Use dataset-specific rates when datasets were processed at different sampling
rates:

```bash
python3 train_tools/dataset_duration.py \
  --dataset-hz recon=4 \
  --dataset-hz go_stanford=4 \
  --dataset-hz sacson=4 \
  --dataset-hz scand=10
```

Run only selected datasets or splits:

```bash
python3 train_tools/dataset_duration.py --datasets recon sacson --splits train --hz 4
```

Write to a custom output path:

```bash
python3 train_tools/dataset_duration.py --hz 4 --output test_tools_logs/custom_statistics.txt
```

Print only without writing a file:

```bash
python3 train_tools/dataset_duration.py --hz 4 --no-output
```

Force direct `traj_data.pkl` scanning instead of using cached split indexes:

```bash
python3 train_tools/dataset_duration.py --hz 4 --no-cached-index
```

Notes:

- Current FlowNav-style `traj_data.pkl` files often contain only `position` and
  `yaw`, so `--hz` or `--dataset-hz` is required for real hour estimates.
- The cached-index path is faster, but it cannot discover timestamp fields
  inside individual `traj_data.pkl` files. Use `--no-cached-index` if timestamp
  based duration is needed.
- The script is read-only for dataset files. It only writes the statistics text
  file under `test_tools_logs/` unless `--no-output` is used. Existing output
  at the selected path is deleted before the new report is written.
