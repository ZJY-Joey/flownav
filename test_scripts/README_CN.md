# FlowNav Goal 条件评测脚本说明

这个文件夹下的脚本用于检查一个核心问题：

> FlowNav 是否真的使用了 goal image？当当前 observation 不变、goal image 改变时，模型生成的轨迹分布是否会随 goal 发生合理变化？

这些脚本都复用仓库已有的 `ViNT_Dataset` 数据加载逻辑，不直接读取原始 HDF5。默认配置文件是
`flownav/config/flownav.yaml`。如果不显式传入 `--checkpoint`，脚本会根据配置中的 `load_run`
自动解析 `logs/<load_run>/latest.pth`。

英文版文档见 [README.md](README.md)。

## 当前维护状态

当前主动维护范围：

- `goal_swap_visualization.py`
- `goal_mask_sensitivity.py`
- `generate_summary_figures.py`
- `common.py` 中被上述脚本使用的公共工具

其他测试脚本暂时保留在仓库中作为历史参考，但已经不再作为当前主要实验流程的一部分，也不再保证和最新的 goal-swap 逻辑同步更新：

- `goal_shuffle_quantitative.py`
- `goal_inconsistent_rate.py`
- `goal_separation_ratio.py`

如果之后要重新使用这些脚本，需要先重新检查它们的样本构造假设、输出路径和指标解释。除非明确需要恢复某个 archived test，后续改动默认只围绕 active 的 goal-swap 和 goal-mask-sensitivity pipeline。

## 输出目录结构

默认 baseline 输出到 `test_logs/flownav_baseline/<dataset>/<script_name>/`。

当 `goal_swap_visualization.py` 使用 `--trajectory-selection cluster` 运行时，
输出到 `test_logs/flownav_cluster/<dataset>/<script_name>/`。

典型结构如下：

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

以 `goal_swap_visualization.py` 为例，文件夹名：

```text
angle10-mmd0p5-emd0p2
```

含义是：

- `angle10`：left / forward / right 的 goal 分类阈值是 10 度。
- `mmd0p5`：异常样本筛选使用的 MMD 阈值是 0.5。
- `emd0p2`：异常样本筛选使用的 EMD 近似阈值是 0.2。

## 共同的样本匹配逻辑

`common.py` 负责加载配置、加载 checkpoint、构建 dataloader、运行模型、查找匹配 goal、画图等公共逻辑。

对于 left / forward / right 这类 matched-goal 测试：

- 当前 observation 固定不变。
- goal image 从同一条 trajectory、同一个当前时刻 `curr_time` 的未来目标中选择。
- 不再从其他 batch 或其他场景里随机拿 goal，避免 observation 和 goal 脱节。
- left / forward / right 的分类依据是 candidate `goal_pos` 在当前机器人局部坐标系下的角度。
- `left`：`goal_pos` 角度大于 `--angle-threshold-deg`。
- `right`：`goal_pos` 角度小于 `- --angle-threshold-deg`。
- `forward`：`goal_pos` 角度绝对值在阈值以内。
- heading filter 会额外检查 goal image 对应时间点的轨迹朝向，避免 goal image 明显朝向相反方向。

这套机制的目标是：构造“同一场景、同一当前 observation、不同方向 goal”的测试，而不是把不相关场景随机拼在一起。

## 主要指标总览

| 指标 | 含义 | 更理想的现象 |
| --- | --- | --- |
| `goal_inconsistent_rate` | 采样轨迹最终方向和 GT 最终方向差异超过阈值的比例。 | 越低越好。 |
| `mean_angle_diff_deg` | 采样轨迹最终方向和 GT 最终方向的平均角度差。 | 越低越好。 |
| `mean_endpoint_mean_distance` | left / forward / right 不同 goal 下 endpoint 均值之间的平均距离。 | 越高说明不同 goal 能推动 endpoint 改变。 |
| `mean_endpoint_rbf_mmd` | 不同 goal 下 endpoint 分布的 RBF-MMD。 | 越高说明分布越容易区分。 |
| `mean_endpoint_sliced_wasserstein` | sliced Wasserstein distance，作为 EMD 的近似。 | 越高说明分布差异越大。 |
| `mean_s_goal` | endpoint 均值差异除以 goal position 差异。 | 越高说明轨迹变化更跟随 goal 变化。 |
| `mean_traj_dtw` | 不同 goal 下平均轨迹的 DTW 距离。 | 越高说明整体轨迹形状差异越明显。 |
| `mean_traj_frechet` | 不同 goal 下平均轨迹的 Frechet 距离。 | 越高说明轨迹曲线差异越明显。 |
| `mean_goal_separation_ratio` | 不同 goal 间 endpoint 分离度 / 同一 goal 内 endpoint 离散度。 | 越高说明 left / forward / right 簇分得越开。 |

对于 MMD / EMD 这类分布指标，如果 goal 明显不同但数值很低，说明模型可能对 goal 不敏感，因为它在不同 goal 下生成了非常接近的轨迹分布。

## 已归档：Goal Shuffle Quantitative

脚本：`goal_shuffle_quantitative.py`

状态：暂时弃用，不属于当前主动维护的评测流程。

### 做了什么

这个脚本比较“正确 goal”和“同场景 alternative goal”下的模型预测差异。

流程：

1. 使用正确 goal 跑模型。
2. 从同一条 trajectory、同一个当前时刻中选择一个方向不同但仍然场景匹配的 alternative goal。
3. 使用 alternative goal 再跑一次模型。
4. 比较两组预测和数据集 GT action 的误差。

它不是随机从别的 batch 拿 goal，而是尽量保证 goal 和 observation 属于同一环境上下文。

### 主要输出

- summary JSON：保存 correct-goal 和 alternative-goal 的定量指标。
- 可选三列可视化图：同一个 observation，对应 left / forward / right 三种 goal。
- 终端日志：显示 batch 进度和匹配进度。

### 重点指标

- `action_loss_mse`
- `ade`
- `fde`
- `waypoint_cos_sim`
- `trajectory_cos_sim`
- `success_rate_fde`

### 运行方式

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

### 排查提示

- 如果 correct-goal 和 alternative-goal 的指标几乎一样，说明模型可能没有充分使用 goal。
- 如果 alternative goal 太极端，可以降低 `--max-alternative-angle-diff-deg`。
- 如果匹配不到样本，可以增加 `--scan-batches` 或放宽角度限制。

## 主动维护：Goal Swap Visualization

脚本：`goal_swap_visualization.py`

### 做了什么

这是当前最核心的同场景 left / forward / right goal sensitivity 测试。

对每个有效样本：

1. 固定当前 observation。
2. 从同一条 trajectory、同一个当前时刻选择 left / forward / right 三张 future goal image。
3. 每个 goal 采样多条 goal-conditioned trajectory。
4. 比较不同 goal 下的 endpoint 分布、平均轨迹、heading、DTW、Frechet、MMD、EMD 近似等。
5. 如果不同 goal 下的 endpoint 分布过于相似，则把该样本标为异常样本。

脚本会自动跑两套设置：

- `heading_filter`：goal image 的视角/轨迹朝向也要和 left / forward / right 类别相容。
- `no_heading_filter`：只根据 `goal_pos` 分类，不额外筛 goal image heading。

trajectory selection 模式：

- `baseline`：保留原始 sampled FlowNav trajectories，也就是之前 goal-swap 的行为。
- `cluster`：每个 goal 先采样多条 trajectory，然后使用 FlowNav 共享的加权轨迹距离聚类，
  选择最大簇的 medoid 作为该 goal 的轨迹用于指标和日志生成。内部会把选中的轨迹重复成
  和 baseline 相同的 shape，保证现有分布指标逻辑不需要改写。

### 输出结构

示例：

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

### 每类输出看什么

- `all_samples/`
  - 全量匹配样本测试。
  - 不输出单样本 PNG / JSON。
  - 主要看全局 endpoint 分布和全局 summary。

- `anomaly_samples/`
  - 先根据 MMD / EMD 阈值筛出异常样本。
  - 再输出异常样本的三列 goal swap 图、endpoint 分布图、JSON 详情和 `anomaly_indices.txt`。

- `goal_swap_global_endpoints_*.png`
  - 左图：left / forward / right 三种 goal 下的 sampled endpoint 分布。
  - 右图：对应 matched `goal_pos` 的分布。
  - 两张图都会标注 left-forward、left-right、forward-right 的 MMD 和 EMD 近似。

- `goal_swap_anomaly_global_endpoints_*.png`
  - 只统计异常样本的 endpoint 和 goal_pos 分布。

- `goal_swap_*_heading_filter_endpoint_comparison_*.png`
  - 父文件夹中的比较图使用 `all_samples` 全量 endpoint 分布。
  - heading 和 no-heading 使用不同颜色和 marker。
  - 图中标注两种设置之间的 MMD 和 EMD 近似。
  - 如果某一类超过 10000 个点，只对可视化下采样，不影响指标计算。

- `anomaly_*.png`
  - 三列分别是 left / forward / right goal。
  - 同一个当前 observation，不同 goal image。
  - BEV 图中包含采样轨迹、机器人当前位置、goal 位置、当前朝向和 goal image time 的局部轨迹朝向。

### 重点指标

- `mean_endpoint_rbf_mmd`
- `mean_endpoint_sliced_wasserstein`
- `mean_endpoint_mean_distance`
- `mean_s_goal`
- `mean_traj_dtw`
- `mean_traj_frechet`
- `class_tv_distance`
- `endpoint_symmetric_kl`

### 异常样本判断

满足下面任一条件就标为异常：

```text
mean_endpoint_rbf_mmd <= --anomaly-mmd-threshold
或
mean_endpoint_sliced_wasserstein <= --anomaly-emd-threshold
```

也就是说，不同 goal 下的 endpoint 分布太相似，就认为这个样本体现出 goal sensitivity 低。

### 运行方式

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

运行聚类 trajectory-selection 版本：

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

### 排查提示

- 如果 goal_pos 分布分得很开，但 endpoint 分布重叠严重，说明 goal 明确不同，但模型输出没有明显响应。
- 如果 goal_pos 分布本身就重叠，说明样本匹配或方向阈值可能太宽松，需要调整 `--angle-threshold-deg`。
- 如果 heading_filter 和 no_heading_filter 差异很大，说明 goal image 视角筛选对这个数据集很重要。
- 如果异常样本太少，可以增大 `--anomaly-mmd-threshold` 或 `--anomaly-emd-threshold`。
- 如果异常样本太多，可以降低这两个阈值。
- 如果全局图点太多，降低 `--global-endpoint-max-points-per-class`。这只影响画图下采样，不影响指标计算。

## 已归档：Goal-Inconsistent Rate

脚本：`goal_inconsistent_rate.py`

状态：暂时弃用，不属于当前主动维护的评测流程。

### 做了什么

这个脚本使用正确 goal 采样 N 条轨迹，然后用 GT action 的最终 waypoint 方向作为目标方向，计算每条采样轨迹最终 waypoint 方向和目标方向的角度差。

如果角度差超过 `--angle-threshold-deg`，这条 sampled trajectory 就被认为是 goal-direction inconsistent。

### 重点指标

- `goal_inconsistent_rate`
- `mean_angle_diff_deg`
- `median_angle_diff_deg`
- `p90_angle_diff_deg`
- `p95_angle_diff_deg`

### 运行方式

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

### 排查提示

- `goal_inconsistent_rate` 高，说明很多采样轨迹最终方向偏离 GT 方向。
- `p90_angle_diff_deg` 或 `p95_angle_diff_deg` 高，说明尾部样本存在严重方向错误。
- 输出文件名会包含角度阈值，例如 `goal_inconsistent_rate_angle45_*.json`，方便比较不同阈值。

## 主动维护：Goal Mask Sensitivity

脚本：

- `goal_mask_sensitivity.py`

### 做了什么

这个测试比较 no-heading-filter 的 left / forward / right goal-conditioned
轨迹，和 goal-masked 轨迹之间的分布差异。

这里的“有 goal”指 goal-swap 使用的同场景 matched goals：从同一条 trajectory、
同一个当前时刻选择 left / forward / right 三张未来 goal image，并且
`filter_goal_heading=False`。这里的“无 goal”指同一个 observation 和 goal image
走模型的 goal-mask 路径，生成不带 goal conditioning 的轨迹。

### 重点指标

- `endpoint_mean_l2`：goal-conditioned 和 masked endpoint 均值距离。
- `endpoint_rbf_mmd`：goal-conditioned 和 masked endpoint 的 RBF-MMD。
- `endpoint_sliced_wasserstein`：sliced Wasserstein distance，作为 EMD 近似。
- `matched_sample_ade` / `matched_sample_fde`：有 goal 和无 goal 采样轨迹的 paired 差异。
- 有 goal endpoints 内部 left / forward / right 的 direction-pair MMD/EMD，以及 masked endpoints 内部对应指标。

### 定量运行方式

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

运行聚类 trajectory-selection 版本：

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

输出：

```text
test_logs/flownav_baseline/recon/goal_mask_sensitivity/angle10-no_heading_filter/
  goal_mask_sensitivity_summary_*.json
  goal_mask_sensitivity_items_*.csv
  goal_mask_sensitivity_endpoints_*.npz
```

### 可视化运行方式

`goal_mask_sensitivity.py` 会在同一次测试运行中直接生成可视化图片。
图片和 JSON / CSV / NPZ 输出保存在同一个 `goal_mask_sensitivity/<run_tag>/`
目录下。

主要可视化输出：

- `goal_mask_endpoint_shift_by_direction_*.png`
  - 三个 panel 分别是 left / forward / right。
  - 每个 panel 叠加有 goal endpoint 和无 goal endpoint。
  - 每个 panel 标注有 goal vs 无 goal 的 MMD、EMD 近似和 endpoint mean shift。

- `goal_mask_direction_distribution_comparison_*.png`
  - 左图：有 goal 时 left / forward / right 的 endpoint 分布。
  - 中图：无 goal 时，按请求的 goal class 分组后的 endpoint 分布。
  - 右图：matched goal position 分布。
  - 前两张图会标注 direction-pair MMD 和 EMD 近似。

### 排查提示

- 如果有 goal 的 left / forward / right endpoints 分得开，但无 goal endpoints collapse，说明模型确实使用 goal。
- 如果有 goal 和无 goal endpoints 几乎一样，说明 goal conditioning 弱。
- 如果无 goal endpoints 仍然按 left / forward / right 分开，需要检查 observation 本身是否已经决定路线，或者 matched goals 是否受轨迹位置偏置影响。

## 已归档：Goal Separation Ratio

脚本：`goal_separation_ratio.py`

状态：暂时弃用，不属于当前主动维护的评测流程。

### 做了什么

这个脚本构造大量同场景 hard triplet：同一个当前 observation，对应 left / forward / right 三种 goal。

对每种 goal 采样多条轨迹后：

1. 计算不同 goal 之间 endpoint 均值的距离，也就是 inter-goal distance。
2. 计算同一 goal 内部 sampled endpoint 的离散程度，也就是 within-goal dispersion。
3. 用两者相除得到 Goal Separation Ratio。

公式可以理解为：

```text
GSR = between-goal separation / within-goal dispersion
```

### 重点指标

- `mean_inter_goal_distance`
- `mean_within_goal_dispersion`
- `mean_goal_separation_ratio`
- `median_goal_separation_ratio`
- `p10_goal_separation_ratio`
- `p90_goal_separation_ratio`

### 运行方式

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

### 排查提示

- GSR 越高，说明 left / forward / right 轨迹 endpoint 簇分得越开。
- GSR 很低，说明不同 goal 下轨迹簇重叠严重。
- 如果 within-goal dispersion 很大，说明模型采样本身可能不稳定。
- 如果 inter-goal distance 很小，说明模型可能对 goal 变化不敏感。

## 常用参数

指定 checkpoint：

```bash
--checkpoint logs/flownav0421/latest.pth
```

指定设备：

```bash
--device cuda:0
```

快速 smoke test：

```bash
--batch-size 8 --max-batches 1 --num-samples 4
```

限制可视化样本数量：

```bash
--max-visualizations 20
```

控制 left / forward / right 匹配严格程度：

```bash
--angle-threshold-deg 10 --max-direction-angle-deg 90
```

## Summary Figure 汇总图生成

脚本：`generate_summary_figures.py`

这个脚本不会重新跑模型。它只读取已有的 `test_logs` 输出，并在
`test_logs/<variant>/summary_figure/` 下生成论文风格的汇总图和表格。

这个脚本支持不完整实验的容错。如果某个数据集、角度、stage 或 setting
还没跑完，脚本仍会生成当前能生成的图；缺失的图片 panel 会标成
`missing`。同时会输出：

- `missing_summary_inputs.json`
- `missing_summary_inputs.md`

这两个文件会列出缺少哪些 swap 或 mask 对比输入。

运行方式：

```bash
python3 test_scripts/generate_summary_figures.py \
  --log-root test_logs \
  --variant flownav_baseline
```

输出内容：

- `fig1_quantitative_summary.png`
  - 多面板总体统计图。
  - Panel A：heading-filter 和 no-heading-filter 的 mean EMD 近似对比。
  - Panel B：heading-filter 和 no-heading-filter 的 mean MMD 对比。
  - Panel C：matched sample 数量和 heading-filter retention ratio。
  - Panel D：异常样本数量。

- `fig2_global_endpoint_angle10.png`
  - `angle=10` 的 3 x 2 全局分布拼图。
  - 行是数据集。
  - 列是 no-heading-filter 和 heading-filter。
  - 每个子图复用已有的 global endpoint / matched goal-position 分布图。

- `fig2_global_endpoint_angle15.png`
  - `angle=15` 的补充版本。

- `fig2_all_vs_anomaly_angle10.png`
  - `angle=10` 的 3 x 2 拼图。
  - 行是数据集。
  - 列是 `all samples` 和 `anomaly samples`。
  - 只使用 `no-heading-filter` 结果。

- `fig2_all_vs_anomaly_angle15.png`
  - `angle=15` 的补充版本。

- `fig3_hard_case_gallery_angle10.png`
  - `angle=10` 的多 hard-case gallery。
  - 每个数据集展示多个异常样本。
  - 每个 hard case 保留两张图：swap 三列可视化图和对应 endpoint 分布图。
  - 选择样本时会尽量让 `dataset_index` 间隔较远，避免连续帧导致 case 过于相似。
  - 如果某个数据集在当前 angle 下可用异常样本少于请求数量，图会自动使用该 angle 下所有数据集共同可用的最大 case 数，避免出现 missing。

- `fig3_hard_case_gallery_angle15.png`
  - `angle=15` 的补充版本。

- `fig4_anomaly_summary.png`
  - 异常样本数量和描述性异常比例统计图。
  - 注意：heading-filter 和 no-heading-filter 的 matched sample pool 不同，因此比例只作为描述性参考。

- `fig5_paired_improvement.png`
  - 严格 paired comparison，只使用 heading-filter 和 no-heading-filter 两边都存在的 matched sample。
  - 展示 paired EMD、paired MMD、delta 和 paired sample count。
  - 这是判断 heading-filter 是否真正改善指标时最公平的图，因为它固定了评测样本集合。

- `fig6_goal_mask_direction_distribution_comparison.png`
  - 不同数据集的 `goal_mask_direction_distribution_comparison_*.png` 拼图。

- `fig6_goal_mask_mmd_emd_delta.png`
  - 不同数据集上有 goal 减无 goal 的 direction-pair MMD/EMD delta。
  - bar 标注中会包含 delta，以及有 goal / 无 goal 的原始值。

- `table1_summary.csv` 和 `table1_summary.md`
  - 多数据集 summary table，包含 matched 数量、MMD、EMD、异常数量、delta 和 heading-filter retention。

- `table2_paired_improvement.csv` 和 `table2_paired_improvement.md`
  - `fig5_paired_improvement.png` 对应的数值表格。

- `missing_summary_inputs.json` 和 `missing_summary_inputs.md`
  - 记录缺少哪些 dataset / angle / setting，导致无法生成完整比较。

常用参数：

```bash
python3 test_scripts/generate_summary_figures.py \
  --log-root test_logs \
  --variant flownav_cluster \
  --hard-cases-per-dataset 5 \
  --hard-case-min-index-gap 500
```

- `--variant` 选择要汇总的日志命名空间：`flownav_baseline` 或 `flownav_cluster`。
- `--hard-cases-per-dataset` 控制每个数据集展示多少个 hard case。
- `--hard-case-min-index-gap` 控制选中的 hard case 在 `dataset_index` 上尽量相隔多远；如果样本不够，脚本会自动放宽间隔。

## 常见问题排查

- 长时间没有输出：
  模型加载、数据扫描和采样都需要时间。先用更小的 `--scan-batches`、`--max-batches`、`--num-samples` 做 smoke test。

- 找不到 same-trajectory / same-time 样本：
  增大 `--scan-batches`，降低 `--angle-threshold-deg`，或放宽 `--max-direction-angle-deg`。

- 异常样本太少：
  增大 `--anomaly-mmd-threshold` 或 `--anomaly-emd-threshold`。

- 异常样本太多：
  降低异常阈值，并检查 goal_pos 分布是否本身就没有分开。

- endpoint 图点太密：
  降低 `--global-endpoint-max-points-per-class`。该参数只影响可视化，不影响指标计算。

- goal image 看起来和 left / forward / right 标签不一致：
  对比 `heading_filter` 和 `no_heading_filter`。如果差异很大，说明 goal image 的视角筛选非常关键。

- `python` 命令找不到：
  进入项目对应 conda 环境，或者使用该环境提供的 `python3`。
