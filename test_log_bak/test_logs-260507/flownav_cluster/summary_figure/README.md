# Summary Figure Folder

This folder contains paper-style summary figures generated from existing `test_logs` outputs.

## How to Generate

```bash
python3 test_scripts/generate_summary_figures.py --log-root test_logs --hard-cases-per-dataset 5 --hard-case-min-index-gap 500
```

## Figures

- `fig1_quantitative_summary.png`: heading-filter vs no-heading-filter quantitative comparison.
- `fig2_global_endpoint_angle10.png`: no-heading-filter vs heading-filter global endpoint / goal-position collage at angle 10.
- `fig2_global_endpoint_angle15.png`: same collage at angle 15.
- `fig2_all_vs_anomaly_angle10.png`: no-heading-filter all samples vs anomaly samples collage at angle 10.
- `fig2_all_vs_anomaly_angle15.png`: same collage at angle 15.
- `fig3_hard_case_gallery_angle10.png`: diverse hard-case gallery with swap image and endpoint plot for each case at angle 10.
- `fig3_hard_case_gallery_angle15.png`: same gallery at angle 15.
- `fig4_anomaly_summary.png`: raw anomaly counts and descriptive anomaly ratios.
- `fig5_paired_improvement.png`: strict paired comparison using only samples present in both heading-filter and no-heading-filter runs.
- `fig6_goal_mask_direction_distribution_comparison.png`: goal-mask direction-distribution collage across datasets.
- `fig6_goal_mask_mmd_emd_delta.png`: with-goal minus masked-goal direction-pair MMD/EMD deltas across datasets.
- `table1_summary.csv` / `table1_summary.md`: multi-dataset summary table.
- `table2_paired_improvement.csv` / `table2_paired_improvement.md`: numeric values used by `fig5_paired_improvement.png`.

## Notes

- `HF retention = matched_HF / matched_noHF`.
- `fig3` chooses diverse anomaly cases by preferring larger dataset-index gaps when possible.
- `fig2_all_vs_anomaly` uses `no_heading_filter` only.
- `fig5_paired_improvement` uses only shared matched samples present in both heading-filter and no-heading-filter runs.

## Summary Rows

- go_stanford angle10: matched HF=69, matched no-HF=697, mean EMD HF=1.5344, mean EMD no-HF=1.7799
- go_stanford angle15: matched HF=55, matched no-HF=351, mean EMD HF=1.6518, mean EMD no-HF=1.9785
- recon angle10: matched HF=136, matched no-HF=573, mean EMD HF=1.8165, mean EMD no-HF=2.2386
- recon angle15: matched HF=58, matched no-HF=142, mean EMD HF=2.1972, mean EMD no-HF=2.4464
- sacson angle10: matched HF=61, matched no-HF=260, mean EMD HF=1.0258, mean EMD no-HF=1.4870
- sacson angle15: matched HF=37, matched no-HF=126, mean EMD HF=1.4850, mean EMD no-HF=1.6745
