import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = ("go_stanford", "recon", "sacson")
DEFAULT_ANGLES = (10, 15)
HORIZON_BUCKETS = {
    "short": (4, 7),
    "mid": (8, 12),
    "long": (13, 19),
}
PROCESSES = ("direction", "horizon", "mask", "heading", "subgoal")


@dataclass(frozen=True)
class EvalTask:
    name: str
    cmd: list[str]


def log(message: str) -> None:
    print(f"[goal_condition_suite] {message}", flush=True)


def run_command(cmd: list[str], dry_run: bool, label: str | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    prefix = f"{label}: " if label else ""
    if dry_run:
        log(f"DRY-RUN {prefix}{printable}")
        return
    log(f"{prefix}{printable}")
    subprocess.run(cmd, check=True)


def detect_cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        return len(devices)

    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        return 0


def ddp_devices(args) -> list[str]:
    if args.ddp_devices:
        return args.ddp_devices
    if args.device:
        return [args.device]

    count = detect_cuda_device_count()
    if count <= 0:
        return ["cpu"]
    return [f"cuda:{index}" for index in range(count)]


def command_with_device(cmd: list[str], device: str) -> list[str]:
    updated = []
    skip_next = False
    for item in cmd:
        if skip_next:
            skip_next = False
            continue
        if item == "--device":
            skip_next = True
            continue
        updated.append(item)
    return [*updated, "--device", device]


def run_tasks_parallel(tasks: list[EvalTask], args) -> None:
    if not tasks:
        return

    devices = ddp_devices(args)
    max_workers = min(len(tasks), len(devices))
    if args.ddp_max_workers is not None:
        max_workers = min(max_workers, args.ddp_max_workers)
    max_workers = max(1, max_workers)

    log(
        "ddp-eval enabled: process-parallel eval, "
        f"tasks={len(tasks)} devices={','.join(devices)} workers={max_workers}"
    )
    if args.dry_run:
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)]
            run_command(command_with_device(task.cmd, device), True, f"{task.name} [{device}]")
        return

    def worker(index_task: tuple[int, EvalTask]) -> None:
        index, task = index_task
        device = devices[index % len(devices)]
        run_command(command_with_device(task.cmd, device), False, f"{task.name} [{device}]")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in enumerate(tasks)]
        for future in as_completed(futures):
            future.result()


def latest(paths):
    paths = sorted(paths)
    return paths[-1] if paths else None


def angle_tag(angle: float) -> str:
    if float(angle).is_integer():
        return f"angle{int(angle)}"
    return f"angle{str(angle).replace('.', 'p')}"


def threshold_tag(value: float) -> str:
    return str(value).replace(".", "p")


def swap_run_dir(args, dataset: str, angle: float) -> Path:
    tag = (
        f"{angle_tag(angle)}-"
        f"mmd{threshold_tag(args.anomaly_mmd_threshold)}-"
        f"emd{threshold_tag(args.anomaly_emd_threshold)}"
    )
    return (
        Path(args.log_root)
        / args.variant
        / dataset
        / "goal_swap_visualization"
        / tag
    )


def has_swap_summary(args, dataset: str, angle: float, setting: str, stage: str) -> bool:
    run_dir = swap_run_dir(args, dataset, angle)
    return latest(
        (run_dir / stage / setting).glob("goal_swap_visualization_summary_*.json")
    ) is not None


def has_horizon_swap_summary(args, bucket: str, dataset: str) -> bool:
    run_dir = (
        Path(args.horizon_root)
        / bucket
        / args.variant
        / dataset
        / "goal_swap_visualization"
        / (
            f"{angle_tag(args.main_angle)}-"
            f"mmd{threshold_tag(args.anomaly_mmd_threshold)}-"
            f"emd{threshold_tag(args.anomaly_emd_threshold)}"
        )
        / "all_samples"
        / "no_heading_filter"
    )
    return latest(run_dir.glob("goal_swap_visualization_summary_*.json")) is not None


def has_mask_summary(args, root: Path, dataset: str, angle: float) -> bool:
    mask_dir = (
        root
        / args.variant
        / dataset
        / "goal_mask_sensitivity"
        / f"{angle_tag(angle)}-no_heading_filter"
    )
    return latest(mask_dir.glob("goal_mask_sensitivity_summary_*.json")) is not None


def has_recon_subgoal_summary(args) -> bool:
    output_dir = Path(args.horizon_root) / args.variant / "head_horizon_summary"
    return latest(output_dir.glob("recon_head_horizon_summary_*.json")) is not None


def common_model_args(args, device: str | None = None) -> list[str]:
    cmd = ["--config", args.config]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    selected_device = device or args.device
    if selected_device:
        cmd += ["--device", selected_device]
    return cmd


def direction_tasks(args) -> list[EvalTask]:
    tasks = []
    for dataset in args.datasets:
        for angle in args.angles:
            if (
                not args.force
                and has_swap_summary(args, dataset, angle, "no_heading_filter", "all_samples")
                and has_swap_summary(
                    args,
                    dataset,
                    angle,
                    "no_heading_filter",
                    "anomaly_samples",
                )
            ):
                log(f"skip direction dataset={dataset} angle={angle}: existing summaries")
                continue
            tasks.append(
                EvalTask(
                    name=f"direction dataset={dataset} angle={angle}",
                    cmd=[
                sys.executable,
                "test_scripts/goal_swap_visualization.py",
                *common_model_args(args),
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--batch-size",
                str(args.batch_size),
                "--scan-batches",
                str(args.scan_batches),
                "--num-samples",
                str(args.num_samples),
                "--angle-threshold-deg",
                str(angle),
                "--max-direction-angle-deg",
                str(args.max_direction_angle_deg),
                "--anomaly-mmd-threshold",
                str(args.anomaly_mmd_threshold),
                "--anomaly-emd-threshold",
                str(args.anomaly_emd_threshold),
                "--heading-filter-mode",
                "no_heading_filter",
                "--trajectory-selection",
                "baseline",
                "--global-endpoint-max-points-per-class",
                str(args.global_endpoint_max_points_per_class),
                "--global-metric-max-points-per-class",
                str(args.global_metric_max_points_per_class),
                "--output-dir",
                args.log_root,
                    ],
                )
            )
    return tasks


def run_direction_process(args) -> None:
    for task in direction_tasks(args):
        run_command(task.cmd, args.dry_run, task.name)
    generate_summary_figures(args)


def heading_tasks(args) -> list[EvalTask]:
    tasks = []
    for dataset in args.datasets:
        for angle in args.angles:
            if not args.force and has_swap_summary(
                args, dataset, angle, "heading_filter", "all_samples"
            ):
                log(f"skip heading dataset={dataset} angle={angle}: existing summary")
                continue
            tasks.append(
                EvalTask(
                    name=f"heading dataset={dataset} angle={angle}",
                    cmd=[
                sys.executable,
                "test_scripts/goal_swap_visualization.py",
                *common_model_args(args),
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--batch-size",
                str(args.batch_size),
                "--scan-batches",
                str(args.scan_batches),
                "--num-samples",
                str(args.num_samples),
                "--angle-threshold-deg",
                str(angle),
                "--max-direction-angle-deg",
                str(args.max_direction_angle_deg),
                "--anomaly-mmd-threshold",
                str(args.anomaly_mmd_threshold),
                "--anomaly-emd-threshold",
                str(args.anomaly_emd_threshold),
                "--heading-filter-mode",
                "heading_filter",
                "--skip-anomaly-stage",
                "--trajectory-selection",
                "baseline",
                "--global-endpoint-max-points-per-class",
                str(args.global_endpoint_max_points_per_class),
                "--global-metric-max-points-per-class",
                str(args.global_metric_max_points_per_class),
                "--output-dir",
                args.log_root,
                    ],
                )
            )
    return tasks


def run_heading_process(args) -> None:
    for task in heading_tasks(args):
        run_command(task.cmd, args.dry_run, task.name)
    generate_summary_figures(args)


def horizon_tasks(args) -> list[EvalTask]:
    tasks = []
    for bucket, (min_offset, max_offset) in HORIZON_BUCKETS.items():
        for dataset in args.datasets:
            if not args.force and has_horizon_swap_summary(args, bucket, dataset):
                log(f"skip horizon bucket={bucket} dataset={dataset}: existing summary")
                continue
            tasks.append(
                EvalTask(
                    name=f"horizon bucket={bucket} dataset={dataset}",
                    cmd=[
                sys.executable,
                "test_scripts/goal_swap_visualization.py",
                *common_model_args(args),
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--batch-size",
                str(args.batch_size),
                "--scan-batches",
                str(args.horizon_scan_batches),
                "--num-samples",
                str(args.num_samples),
                "--angle-threshold-deg",
                str(args.main_angle),
                "--max-direction-angle-deg",
                str(args.max_direction_angle_deg),
                "--min-goal-offset",
                str(min_offset),
                "--max-goal-offset",
                str(max_offset),
                "--anomaly-mmd-threshold",
                str(args.anomaly_mmd_threshold),
                "--anomaly-emd-threshold",
                str(args.anomaly_emd_threshold),
                "--heading-filter-mode",
                "no_heading_filter",
                "--trajectory-selection",
                "baseline",
                "--global-endpoint-max-points-per-class",
                str(args.global_endpoint_max_points_per_class),
                "--global-metric-max-points-per-class",
                str(args.global_metric_max_points_per_class),
                "--output-dir",
                str(Path(args.horizon_root) / bucket),
                    ],
                )
            )
    return tasks


def run_horizon_postprocess(args) -> None:
    run_command(
        [
            sys.executable,
            "test_scripts/dist_head_backfill.py",
            *common_model_args(args),
            "--log-root",
            args.log_root,
            "--variant",
            args.variant,
            "--angle",
            str(args.main_angle),
            "--setting",
            "no_heading_filter",
            "--keep-going",
        ],
        args.dry_run,
    )
    run_command(
        [
            sys.executable,
            "test_scripts/dist_head_backfill.py",
            *common_model_args(args),
            "--log-root",
            args.horizon_root,
            "--variant",
            args.variant,
            "--angle",
            str(args.main_angle),
            "--setting",
            "no_heading_filter",
            "--keep-going",
        ],
        args.dry_run,
    )
    generate_head_horizon_figures(args)


def run_horizon_process(args) -> None:
    for task in horizon_tasks(args):
        run_command(task.cmd, args.dry_run, task.name)
    run_horizon_postprocess(args)


def mask_tasks(args) -> list[EvalTask]:
    tasks = []
    root = Path(args.log_root)
    for dataset in args.datasets:
        if not args.force and has_mask_summary(args, root, dataset, args.main_angle):
            log(f"skip mask dataset={dataset}: existing summary")
            continue
        tasks.append(
            EvalTask(
                name=f"mask dataset={dataset}",
                cmd=[
            sys.executable,
            "test_scripts/goal_mask_sensitivity.py",
            *common_model_args(args),
            "--dataset",
            dataset,
            "--split",
            args.split,
            "--batch-size",
            str(args.batch_size),
            "--scan-batches",
            str(args.scan_batches),
            "--num-samples",
            str(args.num_samples),
            "--angle-threshold-deg",
            str(args.main_angle),
            "--max-direction-angle-deg",
            str(args.max_direction_angle_deg),
            "--trajectory-selection",
            "baseline",
            "--global-endpoint-max-points-per-class",
            str(args.global_endpoint_max_points_per_class),
            "--output-dir",
            args.log_root,
                ],
            )
        )
    return tasks


def run_mask_process(args) -> None:
    for task in mask_tasks(args):
        run_command(task.cmd, args.dry_run, task.name)
    generate_summary_figures(args)


def subgoal_tasks(args) -> list[EvalTask]:
    if not args.force and has_recon_subgoal_summary(args):
        log("skip subgoal: existing recon_head_horizon_summary")
        return []
    cmd = [
        sys.executable,
        "test_scripts/recon_head_horizon_summary.py",
        *common_model_args(args),
        "--dataset",
        "recon",
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--scan-batches",
        str(args.subgoal_scan_batches),
        "--angle-threshold-deg",
        str(args.main_angle),
        "--num-flow-samples",
        str(args.subgoal_num_flow_samples),
        "--output-dir",
        str(Path(args.horizon_root) / args.variant / "head_horizon_summary"),
    ]
    if args.subgoal_max_sets is not None:
        cmd += ["--max-sets", str(args.subgoal_max_sets)]
    return [EvalTask(name="subgoal dataset=recon", cmd=cmd)]


def run_subgoal_process(args) -> None:
    for task in subgoal_tasks(args):
        run_command(task.cmd, args.dry_run, task.name)


def generate_summary_figures(args) -> None:
    run_command(
        [
            sys.executable,
            "test_scripts/generate_summary_figures.py",
            "--log-root",
            args.log_root,
            "--variant",
            args.variant,
            "--main-angle",
            str(args.main_angle),
            "--supp-angle",
            str(args.supp_angle),
            "--hard-cases-per-dataset",
            str(args.hard_cases_per_dataset),
            "--hard-case-min-index-gap",
            str(args.hard_case_min_index_gap),
        ],
        args.dry_run,
    )


def generate_head_horizon_figures(args) -> None:
    run_command(
        [
            sys.executable,
            "test_scripts/generate_head_horizon_figures.py",
            "--head-log-root",
            args.log_root,
            "--horizon-root",
            args.horizon_root,
            "--variant",
            args.variant,
            "--angle",
            str(args.main_angle),
            "--output-dir",
            str(Path(args.horizon_root) / args.variant / "head_horizon_summary"),
        ],
        args.dry_run,
    )


def run_ddp_eval(args, processes: list[str]) -> None:
    task_builders = {
        "direction": direction_tasks,
        "horizon": horizon_tasks,
        "mask": mask_tasks,
        "heading": heading_tasks,
        "subgoal": subgoal_tasks,
    }

    eval_tasks = []
    needs_summary_figures = False
    needs_horizon_postprocess = False

    for process in processes:
        log(f"collect process={process}")
        eval_tasks.extend(task_builders[process](args))
        if process in {"direction", "mask", "heading"}:
            needs_summary_figures = True
        if process == "horizon":
            needs_horizon_postprocess = True

    run_tasks_parallel(eval_tasks, args)

    if needs_horizon_postprocess:
        log("start horizon postprocess")
        run_horizon_postprocess(args)
        log("done horizon postprocess")

    if needs_summary_figures:
        log("start summary figures")
        generate_summary_figures(args)
        log("done summary figures")

    for process in processes:
        log(f"done process={process}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the maintained FlowNav goal-condition evaluation pipeline as "
            "five selectable processes. Existing outputs are reused by default."
        )
    )
    parser.add_argument("--config", default="flownav/config/flownav.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--log-root",
        "--output-dir",
        dest="log_root",
        default="test_logs",
        help=(
            "Root directory for direction/mask/heading outputs. "
            "`--output-dir` is an alias for this option."
        ),
    )
    parser.add_argument(
        "--horizon-root",
        "--horizon-output-dir",
        dest="horizon_root",
        default="test_logs_horizon",
        help=(
            "Root directory for horizon and subgoal outputs. "
            "`--horizon-output-dir` is an alias for this option."
        ),
    )
    parser.add_argument("--variant", default="flownav_baseline")
    parser.add_argument(
        "--process",
        action="append",
        choices=PROCESSES,
        help=(
            "Process to run. Repeat to run multiple. If omitted, runs all five: "
            "direction, horizon, mask, heading, subgoal."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument("--angles", nargs="+", type=float, default=list(DEFAULT_ANGLES))
    parser.add_argument("--main-angle", type=float, default=10)
    parser.add_argument("--supp-angle", type=float, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scan-batches", type=int, default=200)
    parser.add_argument("--horizon-scan-batches", type=int, default=200)
    parser.add_argument("--subgoal-scan-batches", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--subgoal-num-flow-samples", type=int, default=8)
    parser.add_argument("--subgoal-max-sets", type=int, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--anomaly-mmd-threshold", type=float, default=0.2)
    parser.add_argument("--anomaly-emd-threshold", type=float, default=1.0)
    parser.add_argument("--global-endpoint-max-points-per-class", type=int, default=10000)
    parser.add_argument("--global-metric-max-points-per-class", type=int, default=5000)
    parser.add_argument("--hard-cases-per-dataset", type=int, default=5)
    parser.add_argument("--hard-case-min-index-gap", type=int, default=500)
    parser.add_argument(
        "--ddp-eval",
        action="store_true",
        help=(
            "Run independent eval jobs in parallel across CUDA devices. "
            "This is process-level multi-GPU evaluation, not torch.distributed DDP."
        ),
    )
    parser.add_argument(
        "--ddp-devices",
        nargs="+",
        default=None,
        help=(
            "Devices used by --ddp-eval, for example: "
            "--ddp-devices cuda:0 cuda:1 cuda:2. Defaults to all visible GPUs."
        ),
    )
    parser.add_argument(
        "--ddp-max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent eval subprocesses under --ddp-eval.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processes = args.process or list(PROCESSES)
    dispatch = {
        "direction": run_direction_process,
        "horizon": run_horizon_process,
        "mask": run_mask_process,
        "heading": run_heading_process,
        "subgoal": run_subgoal_process,
    }
    log(f"selected processes: {', '.join(processes)}")
    log("cluster trajectory-selection is intentionally disabled in this suite.")
    if args.ddp_eval:
        run_ddp_eval(args, processes)
        return

    for process in processes:
        log(f"start process={process}")
        dispatch[process](args)
        log(f"done process={process}")


if __name__ == "__main__":
    main()
