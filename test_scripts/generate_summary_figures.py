import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ["go_stanford", "recon", "sacson"]
SETTING_ORDER = ["heading_filter", "no_heading_filter"]
SETTING_LABELS = {
    "heading_filter": "heading-filter",
    "no_heading_filter": "no-heading-filter",
}
SETTING_COLORS = {
    "heading_filter": "#3b7ddd",
    "no_heading_filter": "#d9822b",
}
VARIANT_ORDER = ["flownav_baseline", "flownav_cluster"]


def latest_path(paths):
    paths = sorted(paths)
    return paths[-1] if paths else None


def parse_run_dir(run_dir: Path):
    name = run_dir.name
    parts = name.split("-")
    angle = None
    for part in parts:
        if part.startswith("angle"):
            angle = float(part.replace("angle", "").replace("p", "."))
    return angle


def mean_metric(summary, key):
    values = [
        row.get(key)
        for row in summary.get("metrics", [])
        if isinstance(row.get(key), (int, float))
    ]
    return float(np.mean(values)) if values else np.nan


def summary_variant(summary_path: Path, summary: dict) -> str:
    if summary.get("output_variant"):
        return summary["output_variant"]
    for part in summary_path.parts:
        if part.startswith("flownav_"):
            return part
    return "flownav_baseline"


def matches_variant(summary_path: Path, summary: dict, variant: str, log_root: Path) -> bool:
    if summary_variant(summary_path, summary) == variant:
        return True
    if variant in summary_path.parts or variant in log_root.parts:
        return True
    return any(variant in part for part in summary_path.parts)


def load_summary_rows(log_root: Path, variant: str):
    rows = []
    for summary_path in sorted(log_root.rglob("goal_swap_visualization_summary_*.json")):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if summary.get("stage") != "all_samples":
            continue
        if not matches_variant(summary_path, summary, variant, log_root):
            continue
        run_dir = summary_path.parents[2]
        setting = "heading_filter" if summary.get("filter_goal_heading") else "no_heading_filter"
        rows.append(
            {
                "dataset": summary.get("dataset"),
                "angle": float(summary.get("angle_threshold_deg")),
                "setting": setting,
                "run_dir": run_dir,
                "summary_path": summary_path,
                "variant": variant,
                "matched": int(summary.get("num_matched_sets", 0)),
                "anomalies": int(get_anomaly_count_from_stage(run_dir, setting)),
                "mean_mmd": mean_metric(summary, "mean_endpoint_rbf_mmd"),
                "mean_emd": mean_metric(summary, "mean_endpoint_sliced_wasserstein"),
                "mean_endpoint_distance": mean_metric(summary, "mean_endpoint_mean_distance"),
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 999,
            r["angle"],
            r["setting"],
        ),
    )


def build_missing_report(rows, mask_rows, args):
    grouped = group_rows(rows)
    missing = {
        "variant": args.variant,
        "swap": [],
        "mask": [],
        "notes": [],
    }
    for angle in [args.main_angle, args.supp_angle]:
        for dataset in DATASET_ORDER:
            group = grouped.get((dataset, float(angle)), {})
            for setting in SETTING_ORDER:
                if setting not in group:
                    missing["swap"].append(
                        {
                            "dataset": dataset,
                            "angle": angle,
                            "setting": setting,
                            "missing": "goal_swap all_samples summary",
                        }
                    )
    mask_by_key = {
        (row["dataset"], float(row["angle"])): row
        for row in mask_rows
    }
    for dataset in DATASET_ORDER:
        key = (dataset, float(args.main_angle))
        row = mask_by_key.get(key)
        if row is None:
            missing["mask"].append(
                {
                    "dataset": dataset,
                    "angle": args.main_angle,
                    "missing": "goal_mask_sensitivity summary",
                }
            )
            continue
        for field in ["direction_distribution_path", "endpoint_shift_path"]:
            path = row.get(field)
            if path is None or not Path(path).exists():
                missing["mask"].append(
                    {
                        "dataset": dataset,
                        "angle": args.main_angle,
                        "missing": field,
                    }
                )
    if not rows:
        missing["notes"].append("No goal_swap summaries were found; only mask figures can be generated.")
    if not mask_rows:
        missing["notes"].append("No goal_mask_sensitivity summaries were found; Fig6 will be skipped.")
    return missing


def write_missing_report(output_dir: Path, missing_report: dict):
    json_path = output_dir / "missing_summary_inputs.json"
    md_path = output_dir / "missing_summary_inputs.md"
    with open(json_path, "w") as f:
        json.dump(missing_report, f, indent=2)
    lines = ["# Missing Summary Inputs", ""]
    for section in ["swap", "mask"]:
        lines.append(f"## {section}")
        items = missing_report.get(section, [])
        if not items:
            lines.append("")
            lines.append("- None")
        else:
            for item in items:
                detail = ", ".join(f"{key}={value}" for key, value in item.items())
                lines.append(f"- {detail}")
        lines.append("")
    if missing_report.get("notes"):
        lines.append("## Notes")
        for note in missing_report["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def get_anomaly_count_from_stage(run_dir: Path, setting: str):
    paths = sorted(
        (run_dir / "anomaly_samples" / setting).glob(
            "goal_swap_visualization_summary_*.json"
        )
    )
    if not paths:
        return 0
    with open(paths[-1], "r") as f:
        summary = json.load(f)
    return int(summary.get("num_anomalies", 0))


def group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["angle"]), {})[row["setting"]] = row
    return grouped


def experiment_label(dataset, angle):
    return f"{dataset}\nangle{int(angle)}"


def save_summary_table(rows, output_dir: Path):
    grouped = group_rows(rows)
    table_rows = []
    for dataset in DATASET_ORDER:
        for angle in sorted({a for d, a in grouped if d == dataset}):
            group = grouped[(dataset, angle)]
            hf = group.get("heading_filter")
            nh = group.get("no_heading_filter")
            if not hf or not nh:
                continue
            table_rows.append(
                {
                    "Dataset": dataset,
                    "Angle": int(angle),
                    "Matched HF": hf["matched"],
                    "Matched no-HF": nh["matched"],
                    "Mean MMD HF": hf["mean_mmd"],
                    "Mean MMD no-HF": nh["mean_mmd"],
                    "Delta MMD": nh["mean_mmd"] - hf["mean_mmd"],
                    "Mean EMD HF": hf["mean_emd"],
                    "Mean EMD no-HF": nh["mean_emd"],
                    "Delta EMD": nh["mean_emd"] - hf["mean_emd"],
                    "Anomalies HF": hf["anomalies"],
                    "Anomalies no-HF": nh["anomalies"],
                    "HF retention": hf["matched"] / nh["matched"] if nh["matched"] else np.nan,
                }
            )

    csv_path = output_dir / "table1_summary.csv"
    columns = (
        list(table_rows[0].keys())
        if table_rows
        else [
            "Dataset",
            "Angle",
            "Matched HF",
            "Matched no-HF",
            "Mean MMD HF",
            "Mean MMD no-HF",
            "Delta MMD",
            "Mean EMD HF",
            "Mean EMD no-HF",
            "Delta EMD",
            "Anomalies HF",
            "Anomalies no-HF",
            "HF retention",
        ]
    )
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in table_rows:
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")

    md_path = output_dir / "table1_summary.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in table_rows:
            values = []
            for col in columns:
                value = row[col]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            f.write("| " + " | ".join(values) + " |\n")
    return table_rows


def load_summary_json(row):
    with open(row["summary_path"], "r") as f:
        return json.load(f)


def metric_key(item):
    return (item.get("dataset_index"), item.get("curr_time"))


def paired_metric_rows(rows):
    grouped = group_rows(rows)
    paired_rows = []
    for dataset in DATASET_ORDER:
        for angle in sorted({a for d, a in grouped if d == dataset}):
            group = grouped[(dataset, angle)]
            hf_row = group.get("heading_filter")
            nh_row = group.get("no_heading_filter")
            if not hf_row or not nh_row:
                continue
            hf_summary = load_summary_json(hf_row)
            nh_summary = load_summary_json(nh_row)
            hf_items = {
                metric_key(item): item
                for item in hf_summary.get("metrics", [])
                if metric_key(item)[0] is not None
            }
            nh_items = {
                metric_key(item): item
                for item in nh_summary.get("metrics", [])
                if metric_key(item)[0] is not None
            }
            shared_keys = sorted(set(hf_items) & set(nh_items))

            def paired_mean(key, items):
                if not shared_keys:
                    return np.nan
                return float(np.mean([items[k][key] for k in shared_keys]))

            emd_hf = paired_mean("mean_endpoint_sliced_wasserstein", hf_items)
            emd_nohf = paired_mean("mean_endpoint_sliced_wasserstein", nh_items)
            mmd_hf = paired_mean("mean_endpoint_rbf_mmd", hf_items)
            mmd_nohf = paired_mean("mean_endpoint_rbf_mmd", nh_items)
            endpoint_hf = paired_mean("mean_endpoint_mean_distance", hf_items)
            endpoint_nohf = paired_mean("mean_endpoint_mean_distance", nh_items)
            paired_rows.append(
                {
                    "dataset": dataset,
                    "angle": angle,
                    "paired_count": len(shared_keys),
                    "emd_hf": emd_hf,
                    "emd_nohf": emd_nohf,
                    "emd_delta": emd_nohf - emd_hf,
                    "mmd_hf": mmd_hf,
                    "mmd_nohf": mmd_nohf,
                    "mmd_delta": mmd_nohf - mmd_hf,
                    "endpoint_hf": endpoint_hf,
                    "endpoint_nohf": endpoint_nohf,
                    "endpoint_delta": endpoint_nohf - endpoint_hf,
                }
            )
    return paired_rows


def save_paired_table(paired_rows, output_dir: Path):
    columns = [
        "Dataset",
        "Angle",
        "Paired count",
        "Mean EMD HF",
        "Mean EMD no-HF",
        "Delta EMD",
        "Mean MMD HF",
        "Mean MMD no-HF",
        "Delta MMD",
        "Endpoint dist HF",
        "Endpoint dist no-HF",
        "Delta endpoint dist",
    ]
    value_getters = [
        lambda row: row["dataset"],
        lambda row: int(row["angle"]),
        lambda row: row["paired_count"],
        lambda row: row["emd_hf"],
        lambda row: row["emd_nohf"],
        lambda row: row["emd_delta"],
        lambda row: row["mmd_hf"],
        lambda row: row["mmd_nohf"],
        lambda row: row["mmd_delta"],
        lambda row: row["endpoint_hf"],
        lambda row: row["endpoint_nohf"],
        lambda row: row["endpoint_delta"],
    ]
    csv_path = output_dir / "table2_paired_improvement.csv"
    md_path = output_dir / "table2_paired_improvement.md"
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in paired_rows:
            values = [getter(row) for getter in value_getters]
            f.write(
                ",".join(
                    f"{value:.6f}" if isinstance(value, float) else str(value)
                    for value in values
                )
                + "\n"
            )
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in paired_rows:
            values = [getter(row) for getter in value_getters]
            f.write(
                "| "
                + " | ".join(
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                    for value in values
                )
                + " |\n"
            )
    return csv_path, md_path


def plot_quantitative_summary(rows, output_dir: Path):
    grouped = group_rows(rows)
    keys = [
        (dataset, angle)
        for dataset in DATASET_ORDER
        for angle in sorted({a for d, a in grouped if d == dataset})
        if "heading_filter" in grouped[(dataset, angle)]
        and "no_heading_filter" in grouped[(dataset, angle)]
    ]
    if not keys:
        return None
    labels = [experiment_label(dataset, angle) for dataset, angle in keys]
    x = np.arange(len(keys))
    width = 0.36

    hf = [grouped[key]["heading_filter"] for key in keys]
    nh = [grouped[key]["no_heading_filter"] for key in keys]
    retention = np.array([h["matched"] / n["matched"] for h, n in zip(hf, nh)])

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes = axes.ravel()

    def grouped_bars(ax, values_hf, values_nh, ylabel, title):
        ax.bar(x - width / 2, values_hf, width, label="heading-filter", color=SETTING_COLORS["heading_filter"])
        ax.bar(x + width / 2, values_nh, width, label="no-heading-filter", color=SETTING_COLORS["no_heading_filter"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    grouped_bars(
        axes[0],
        [r["mean_emd"] for r in hf],
        [r["mean_emd"] for r in nh],
        "mean sliced Wasserstein (EMD approx.)",
        "A. Mean EMD under goal switch",
    )
    grouped_bars(
        axes[1],
        [r["mean_mmd"] for r in hf],
        [r["mean_mmd"] for r in nh],
        "mean RBF-MMD",
        "B. Mean MMD under goal switch",
    )

    ax = axes[2]
    ax.bar(x - width / 2, [r["matched"] for r in hf], width, label="matched HF", color=SETTING_COLORS["heading_filter"])
    ax.bar(x + width / 2, [r["matched"] for r in nh], width, label="matched no-HF", color=SETTING_COLORS["no_heading_filter"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("matched triplet count")
    ax.set_title("C. Matched samples and HF retention")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, retention, color="#202020", marker="o", linewidth=2.2, label="HF retention")
    ax2.set_ylabel("HF retention = matched_HF / matched_noHF")
    ax2.set_ylim(0, max(1.05, float(np.nanmax(retention)) * 1.15))
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, loc="upper right")

    grouped_bars(
        axes[3],
        [r["anomalies"] for r in hf],
        [r["anomalies"] for r in nh],
        "anomaly count",
        "D. Hard-case anomaly counts",
    )
    axes[3].text(
        0.01,
        0.98,
        "Counts are descriptive because HF and no-HF use different matched pools.",
        transform=axes[3].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.suptitle("Goal-Switch Quantitative Summary Across Datasets", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = output_dir / "fig1_quantitative_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_paired_improvement(paired_rows, output_dir: Path):
    if not paired_rows:
        return None
    labels = [experiment_label(row["dataset"], row["angle"]) for row in paired_rows]
    x = np.arange(len(paired_rows))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes = axes.ravel()

    def paired_bars(ax, hf_values, nohf_values, ylabel, title):
        ax.bar(
            x - width / 2,
            hf_values,
            width,
            label="heading-filter",
            color=SETTING_COLORS["heading_filter"],
        )
        ax.bar(
            x + width / 2,
            nohf_values,
            width,
            label="no-heading-filter",
            color=SETTING_COLORS["no_heading_filter"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    paired_bars(
        axes[0],
        [row["emd_hf"] for row in paired_rows],
        [row["emd_nohf"] for row in paired_rows],
        "paired mean sliced Wasserstein",
        "A. Paired EMD on shared matched samples",
    )
    paired_bars(
        axes[1],
        [row["mmd_hf"] for row in paired_rows],
        [row["mmd_nohf"] for row in paired_rows],
        "paired mean RBF-MMD",
        "B. Paired MMD on shared matched samples",
    )

    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].bar(
        x - width / 2,
        [row["emd_delta"] for row in paired_rows],
        width,
        label="Delta EMD = no-HF - HF",
        color="#5f9e6e",
    )
    axes[2].bar(
        x + width / 2,
        [row["mmd_delta"] for row in paired_rows],
        width,
        label="Delta MMD = no-HF - HF",
        color="#8e6bbf",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].set_ylabel("paired delta")
    axes[2].set_title("C. Positive delta means HF is lower")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    axes[3].bar(
        x,
        [row["paired_count"] for row in paired_rows],
        color="#607d8b",
    )
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=25, ha="right")
    axes[3].set_ylabel("shared matched sample count")
    axes[3].set_title("D. Paired comparison sample size")
    axes[3].grid(axis="y", alpha=0.25)
    axes[3].text(
        0.01,
        0.98,
        "Only samples present in both heading-filter and no-heading-filter runs are used.",
        transform=axes[3].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.suptitle("Paired Improvement on Shared Matched Samples", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = output_dir / "fig5_paired_improvement.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def latest_global_endpoint_image(run_dir: Path, setting: str):
    return latest_path(
        (run_dir / "all_samples" / setting).glob("goal_swap_global_endpoints_*.png")
    )


def latest_stage_global_endpoint_image(run_dir: Path, stage: str, setting: str):
    stage_dir = run_dir / stage / setting
    if stage == "anomaly_samples":
        path = latest_path(stage_dir.glob("goal_swap_anomaly_global_endpoints_*.png"))
        if path is not None:
            return path
    return latest_path(stage_dir.glob("goal_swap_global_endpoints_*.png"))


def add_image(ax, path: Path, title: str):
    ax.axis("off")
    if path is None or not path.exists():
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(title)
        return
    ax.imshow(mpimg.imread(path))
    ax.set_title(title, fontsize=11)


def plot_global_endpoint_collage(rows, output_dir: Path, angle: int):
    grouped = group_rows(rows)
    datasets = DATASET_ORDER
    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 5.2 * len(datasets)))
    if len(datasets) == 1:
        axes = np.asarray([axes])
    for row_idx, dataset in enumerate(datasets):
        group = grouped.get((dataset, float(angle)), {})
        for col_idx, setting in enumerate(["no_heading_filter", "heading_filter"]):
            row = group.get(setting)
            path = (
                latest_global_endpoint_image(row["run_dir"], setting)
                if row is not None
                else None
            )
            add_image(
                axes[row_idx, col_idx],
                path,
                f"{dataset}, angle{angle}, {SETTING_LABELS[setting]}",
            )
    fig.suptitle(f"Global Endpoint and Goal-Position Distributions, angle={angle}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_dir / f"fig2_global_endpoint_angle{angle}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_all_vs_anomaly_collage(rows, output_dir: Path, angle: int):
    grouped = group_rows(rows)
    datasets = DATASET_ORDER
    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 5.2 * len(datasets)))
    if len(datasets) == 1:
        axes = np.asarray([axes])
    for row_idx, dataset in enumerate(datasets):
        group = grouped.get((dataset, float(angle)), {})
        row = group.get("no_heading_filter")
        run_dir = row["run_dir"] if row is not None else None
        setting = "no_heading_filter"
        for col_idx, (stage, label) in enumerate(
            [("all_samples", "all samples"), ("anomaly_samples", "anomaly samples")]
        ):
            path = (
                latest_stage_global_endpoint_image(run_dir, stage, setting)
                if run_dir is not None
                else None
            )
            add_image(
                axes[row_idx, col_idx],
                path,
                f"{dataset}, angle{angle}, {label}, no-heading-filter",
            )
    fig.suptitle(
        f"All Samples vs Anomaly Samples Global Distributions (no-heading-filter), angle={angle}",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_dir / f"fig2_all_vs_anomaly_angle{angle}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def anomaly_prefixes(anomaly_dir: Path):
    records = []
    for json_path in sorted(anomaly_dir.glob("anomaly_*.json")):
        if json_path.name.endswith("_summary.json"):
            continue
        stem = json_path.stem
        swap_png = anomaly_dir / f"{stem}.png"
        endpoint_png = anomaly_dir / f"{stem}_endpoints.png"
        if swap_png.exists() and endpoint_png.exists():
            records.append((json_path, swap_png, endpoint_png))
    return records


def read_anomaly_record(record, setting: str):
    json_path, swap_png, endpoint_png = record
    with open(json_path, "r") as f:
        data = json.load(f)
    source = data.get("base_source", {})
    metrics = data.get("anomaly_metrics", {})
    return {
        "json_path": json_path,
        "swap_png": swap_png,
        "endpoint_png": endpoint_png,
        "setting": setting,
        "dataset_index": source.get("dataset_index"),
        "curr_time": source.get("curr_time"),
        "trajectory": source.get("trajectory"),
        "mean_mmd": metrics.get("mean_endpoint_rbf_mmd", np.inf),
        "mean_emd": metrics.get("mean_endpoint_sliced_wasserstein", np.inf),
    }


def collect_anomaly_records(run_dir: Path):
    records = []
    for setting in ["no_heading_filter", "heading_filter"]:
        anomaly_dir = run_dir / "anomaly_samples" / setting
        for record in anomaly_prefixes(anomaly_dir):
            records.append(read_anomaly_record(record, setting))
    return records


def choose_diverse_anomalies(records, count: int, min_index_gap: int):
    valid = [record for record in records if isinstance(record.get("dataset_index"), int)]
    fallback = [record for record in records if not isinstance(record.get("dataset_index"), int)]
    valid = sorted(valid, key=lambda r: (r["mean_emd"], r["mean_mmd"], r["dataset_index"]))
    selected = []
    for gap in [min_index_gap, min_index_gap // 2, min_index_gap // 4, 0]:
        for record in valid:
            if record in selected:
                continue
            if all(
                abs(record["dataset_index"] - chosen["dataset_index"]) >= gap
                for chosen in selected
            ):
                selected.append(record)
            if len(selected) >= count:
                return selected
    return (selected + fallback)[:count]


def hard_case_title(dataset: str, record):
    if record is None:
        return f"{dataset}: missing"
    return (
        f"{dataset} | {SETTING_LABELS[record['setting']]}\n"
        f"idx={record.get('dataset_index')}, t={record.get('curr_time')}, "
        f"EMD={record['mean_emd']:.2f}, MMD={record['mean_mmd']:.2f}"
    )


def plot_hard_case_gallery(
    rows,
    output_dir: Path,
    angle: int,
    cases_per_dataset: int,
    min_index_gap: int,
):
    grouped = group_rows(rows)
    datasets = DATASET_ORDER
    records_by_dataset = {}
    for dataset in datasets:
        row = grouped.get((dataset, float(angle)), {}).get("no_heading_filter")
        records_by_dataset[dataset] = (
            collect_anomaly_records(row["run_dir"]) if row is not None else []
        )
    available_counts = [len(records_by_dataset[dataset]) for dataset in datasets]
    effective_cases = min([cases_per_dataset] + available_counts) if available_counts else 0
    effective_cases = max(effective_cases, 1)
    columns = effective_cases * 2
    fig, axes = plt.subplots(
        len(datasets),
        columns,
        figsize=(5.0 * columns, 4.8 * len(datasets)),
    )
    if len(datasets) == 1:
        axes = np.asarray([axes])

    for row_idx, dataset in enumerate(datasets):
        records = choose_diverse_anomalies(
            records_by_dataset[dataset],
            count=effective_cases,
            min_index_gap=min_index_gap,
        )
        for col_idx in range(effective_cases):
            record = records[col_idx] if col_idx < len(records) else None
            base_col = col_idx * 2
            add_image(
                axes[row_idx, base_col],
                record["swap_png"] if record else None,
                f"{hard_case_title(dataset, record)}\nswap",
            )
            add_image(
                axes[row_idx, base_col + 1],
                record["endpoint_png"] if record else None,
                "endpoint distribution",
            )

    fig.suptitle(
        f"Representative Diverse Hard Cases, angle={angle} "
        f"({effective_cases} cases per dataset)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_dir / f"fig3_hard_case_gallery_angle{angle}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_anomaly_summary(rows, output_dir: Path):
    grouped = group_rows(rows)
    keys = [
        (dataset, angle)
        for dataset in DATASET_ORDER
        for angle in sorted({a for d, a in grouped if d == dataset})
        if "heading_filter" in grouped[(dataset, angle)]
        and "no_heading_filter" in grouped[(dataset, angle)]
    ]
    if not keys:
        return None
    labels = [experiment_label(dataset, angle) for dataset, angle in keys]
    x = np.arange(len(keys))
    width = 0.36
    hf = [grouped[key]["heading_filter"] for key in keys]
    nh = [grouped[key]["no_heading_filter"] for key in keys]
    hf_ratio = [r["anomalies"] / r["matched"] if r["matched"] else np.nan for r in hf]
    nh_ratio = [r["anomalies"] / r["matched"] if r["matched"] else np.nan for r in nh]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(x - width / 2, [r["anomalies"] for r in hf], width, label="heading-filter", color=SETTING_COLORS["heading_filter"])
    axes[0].bar(x + width / 2, [r["anomalies"] for r in nh], width, label="no-heading-filter", color=SETTING_COLORS["no_heading_filter"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("anomaly count")
    axes[0].set_title("A. Raw anomaly counts")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, hf_ratio, width, label="heading-filter", color=SETTING_COLORS["heading_filter"])
    axes[1].bar(x + width / 2, nh_ratio, width, label="no-heading-filter", color=SETTING_COLORS["no_heading_filter"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("anomaly ratio")
    axes[1].set_title("B. Anomaly ratio, descriptive only")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].text(
        0.01,
        0.98,
        "Ratios are not strictly paired because matched pools differ between settings.",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.suptitle("Hard-Case Anomaly Summary", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "fig4_anomaly_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def resolve_existing_path(summary_path: Path, value: str | None):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = summary_path.parent / path.name
    return candidate if candidate.exists() else path


def load_mask_summary_rows(log_root: Path, variant: str, angle: int | None = None):
    rows = []
    for summary_path in sorted(log_root.rglob("goal_mask_sensitivity_summary_*.json")):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if summary.get("test") != "goal_mask_sensitivity":
            continue
        if not matches_variant(summary_path, summary, variant, log_root):
            continue
        if angle is not None and float(summary.get("angle_threshold_deg")) != float(angle):
            continue
        rows.append(
            {
                "dataset": summary.get("dataset"),
                "angle": float(summary.get("angle_threshold_deg")),
                "summary_path": summary_path,
                "direction_distribution_path": resolve_existing_path(
                    summary_path,
                    summary.get("direction_distribution_path"),
                ),
                "endpoint_shift_path": resolve_existing_path(
                    summary_path,
                    summary.get("endpoint_shift_path"),
                ),
                "endpoint_shift_metrics": summary.get("endpoint_shift_metrics", {}),
                "global_goal_direction_pair_metrics": summary.get(
                    "global_goal_direction_pair_metrics", {}
                ),
                "global_masked_direction_pair_metrics": summary.get(
                    "global_masked_direction_pair_metrics", {}
                ),
                "mean_endpoint_mean_l2": summary.get("mean_endpoint_mean_l2"),
                "mean_endpoint_rbf_mmd": summary.get("mean_endpoint_rbf_mmd"),
                "mean_endpoint_sliced_wasserstein": summary.get(
                    "mean_endpoint_sliced_wasserstein"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 999,
            r["angle"],
        ),
    )


def select_mask_rows_by_dataset(mask_rows):
    selected = {}
    for row in mask_rows:
        dataset = row["dataset"]
        if dataset not in selected or row["angle"] < selected[dataset]["angle"]:
            selected[dataset] = row
    return [selected[dataset] for dataset in DATASET_ORDER if dataset in selected]


def plot_mask_direction_distribution_collage(mask_rows, output_dir: Path):
    selected_rows = select_mask_rows_by_dataset(mask_rows)
    if not selected_rows:
        return None
    fig, axes = plt.subplots(
        len(selected_rows),
        1,
        figsize=(16, 5.3 * len(selected_rows)),
    )
    if len(selected_rows) == 1:
        axes = np.asarray([axes])
    for ax, row in zip(axes, selected_rows):
        add_image(
            ax,
            row["direction_distribution_path"],
            f"{row['dataset']}, angle{int(row['angle'])}, goal-mask direction distribution",
        )
    fig.suptitle("Fig6A. Goal-Mask Direction Distribution Comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_dir / "fig6_goal_mask_direction_distribution_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_mask_goal_vs_masked_pair_metric_delta(mask_rows, output_dir: Path):
    selected_rows = select_mask_rows_by_dataset(mask_rows)
    if not selected_rows:
        return None
    pair_keys = ["left_forward", "left_right", "forward_right"]
    labels = [row["dataset"] for row in selected_rows]
    x = np.arange(len(selected_rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    metric_defs = [
        ("endpoint_rbf_mmd", "MMD delta"),
        ("endpoint_sliced_wasserstein", "EMD approx. delta"),
    ]

    def pair_mean(row, source_key, metric_key):
        values = [
            row.get(source_key, {}).get(pair, {}).get(metric_key)
            for pair in pair_keys
        ]
        values = [value for value in values if isinstance(value, (int, float))]
        return float(np.mean(values)) if values else np.nan

    for ax, (metric_key, ylabel) in zip(axes, metric_defs):
        with_values = [
            pair_mean(row, "global_goal_direction_pair_metrics", metric_key)
            for row in selected_rows
        ]
        masked_values = [
            pair_mean(row, "global_masked_direction_pair_metrics", metric_key)
            for row in selected_rows
        ]
        deltas = [
            with_value - masked_value
            if isinstance(with_value, (int, float))
            and isinstance(masked_value, (int, float))
            else np.nan
            for with_value, masked_value in zip(with_values, masked_values)
        ]
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.bar(
            x,
            deltas,
            0.52,
            color="#5f9e6e",
            label="with goal - masked goal",
        )
        for idx, (with_value, masked_value, delta) in enumerate(
            zip(with_values, masked_values, deltas)
        ):
            if not np.isfinite(delta):
                continue
            ax.text(
                idx,
                delta,
                f"Δ={delta:.3f}\nG={with_value:.3f}\nM={masked_value:.3f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_title("A. Direction-pair MMD delta")
    axes[1].set_title("B. Direction-pair EMD approx. delta")

    fig.suptitle(
        "Fig6B. Goal-Mask Direction-Separation Delta",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / "fig6_goal_mask_mmd_emd_delta.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_summary_readme(output_dir: Path, args, table_rows):
    readme_path = output_dir / "README.md"
    lines = [
        "# Summary Figure Folder",
        "",
        "This folder contains paper-style summary figures generated from existing `test_logs` outputs.",
        "",
        "## How to Generate",
        "",
        "```bash",
        "python3 test_scripts/generate_summary_figures.py "
        f"--log-root {args.log_root} "
        f"--hard-cases-per-dataset {args.hard_cases_per_dataset} "
        f"--hard-case-min-index-gap {args.hard_case_min_index_gap}",
        "```",
        "",
        "## Figures",
        "",
        "- `fig1_quantitative_summary.png`: heading-filter vs no-heading-filter quantitative comparison.",
        "- `fig2_global_endpoint_angle10.png`: no-heading-filter vs heading-filter global endpoint / goal-position collage at angle 10.",
        "- `fig2_global_endpoint_angle15.png`: same collage at angle 15.",
        "- `fig2_all_vs_anomaly_angle10.png`: no-heading-filter all samples vs anomaly samples collage at angle 10.",
        "- `fig2_all_vs_anomaly_angle15.png`: same collage at angle 15.",
        "- `fig3_hard_case_gallery_angle10.png`: diverse hard-case gallery with swap image and endpoint plot for each case at angle 10.",
        "- `fig3_hard_case_gallery_angle15.png`: same gallery at angle 15.",
        "- `fig4_anomaly_summary.png`: raw anomaly counts and descriptive anomaly ratios.",
        "- `fig5_paired_improvement.png`: strict paired comparison using only samples present in both heading-filter and no-heading-filter runs.",
        "- `fig6_goal_mask_direction_distribution_comparison.png`: goal-mask direction-distribution collage across datasets.",
        "- `fig6_goal_mask_mmd_emd_delta.png`: with-goal minus masked-goal direction-pair MMD/EMD deltas across datasets.",
        "- `table1_summary.csv` / `table1_summary.md`: multi-dataset summary table.",
        "- `table2_paired_improvement.csv` / `table2_paired_improvement.md`: numeric values used by `fig5_paired_improvement.png`.",
        "",
        "## Notes",
        "",
        "- `HF retention = matched_HF / matched_noHF`.",
        "- `fig3` chooses diverse anomaly cases by preferring larger dataset-index gaps when possible.",
        "- `fig2_all_vs_anomaly` uses `no_heading_filter` only.",
        "- `fig5_paired_improvement` uses only shared matched samples present in both heading-filter and no-heading-filter runs.",
        "",
        "## Summary Rows",
        "",
    ]
    for row in table_rows:
        lines.append(
            f"- {row['Dataset']} angle{row['Angle']}: matched HF={row['Matched HF']}, matched no-HF={row['Matched no-HF']}, "
            f"mean EMD HF={row['Mean EMD HF']:.4f}, mean EMD no-HF={row['Mean EMD no-HF']:.4f}"
        )
    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-style summary figures from existing test_logs."
    )
    parser.add_argument("--log-root", default="test_logs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--variant",
        default="flownav_baseline",
        help="Which test_logs variant to summarize.",
    )
    parser.add_argument("--main-angle", type=int, default=10)
    parser.add_argument("--supp-angle", type=int, default=15)
    parser.add_argument("--hard-cases-per-dataset", type=int, default=5)
    parser.add_argument("--hard-case-min-index-gap", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = Path(args.log_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else log_root / args.variant / "summary_figure"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary_rows(log_root, args.variant)
    mask_rows = load_mask_summary_rows(log_root, args.variant, angle=args.main_angle)
    if not rows and not mask_rows:
        missing_report = {
            "variant": args.variant,
            "swap": [],
            "mask": [],
            "notes": [
                (
                    "No goal_swap or goal_mask_sensitivity summaries found under "
                    f"{log_root}. This is expected if direction/mask/heading were not run."
                )
            ],
        }
        missing_json_path, missing_md_path = write_missing_report(
            output_dir,
            missing_report,
        )
        print(
            f"No goal_swap or goal_mask_sensitivity summaries found for {args.variant} "
            f"under {log_root}; wrote missing report."
        )
        print(missing_json_path)
        print(missing_md_path)
        return

    missing_report = build_missing_report(rows, mask_rows, args)
    missing_json_path, missing_md_path = write_missing_report(output_dir, missing_report)

    table_rows = save_summary_table(rows, output_dir)
    paired_rows = paired_metric_rows(rows)
    save_paired_table(paired_rows, output_dir)
    paths = []
    for path in [
        plot_quantitative_summary(rows, output_dir),
        plot_paired_improvement(paired_rows, output_dir),
        plot_global_endpoint_collage(rows, output_dir, args.main_angle),
        plot_all_vs_anomaly_collage(rows, output_dir, args.main_angle),
        plot_hard_case_gallery(
            rows,
            output_dir,
            args.main_angle,
            args.hard_cases_per_dataset,
            args.hard_case_min_index_gap,
        ),
        plot_anomaly_summary(rows, output_dir),
    ]:
        if path is not None:
            paths.append(path)
    if mask_rows:
        for path in [
            plot_mask_direction_distribution_collage(mask_rows, output_dir),
            plot_mask_goal_vs_masked_pair_metric_delta(mask_rows, output_dir),
        ]:
            if path is not None:
                paths.append(path)
    if any(row["angle"] == args.supp_angle for row in rows):
        for path in [
            plot_global_endpoint_collage(rows, output_dir, args.supp_angle),
            plot_all_vs_anomaly_collage(rows, output_dir, args.supp_angle),
            plot_hard_case_gallery(
                rows,
                output_dir,
                args.supp_angle,
                args.hard_cases_per_dataset,
                args.hard_case_min_index_gap,
            ),
        ]:
            if path is not None:
                paths.append(path)

    readme_path = write_summary_readme(output_dir, args, table_rows)

    print(f"Wrote {len(table_rows)} summary table rows.")
    for path in paths:
        print(path)
    print(output_dir / "table1_summary.csv")
    print(output_dir / "table1_summary.md")
    print(output_dir / "table2_paired_improvement.csv")
    print(output_dir / "table2_paired_improvement.md")
    print(missing_json_path)
    print(missing_md_path)
    print(readme_path)


if __name__ == "__main__":
    main()
