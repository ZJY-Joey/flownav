# Train Tools 工具说明

这个文件夹用于维护训练数据检查和准备相关的辅助脚本。之后每次新增、修改、弃用或删除
`train_tools/` 下的脚本时，都需要同步维护本文件和英文版 `README.md`。

## 脚本状态

| 脚本 | 状态 | 目的 |
| --- | --- | --- |
| `dataset_duration.py` | 维护中 | 统计 FlowNav 配置中的轨迹步数，并估算每个数据集有多少小时。 |

状态含义：

- 维护中：适配当前仓库结构，推荐继续使用。
- 弃用：仅为兼容旧流程或历史参考保留，不建议新流程继续使用。

## `dataset_duration.py`

目的：

- 读取 `flownav/config/flownav.yaml` 中的数据集配置。
- 扫描每个数据集 `train` / `test` split 下的 `traj_names.txt`。
- 优先使用 split 目录下训练代码生成的缓存索引 `dataset_dist_...pkl`；其中的
  `goals_index` 已经为每个轨迹时间步保存了一条记录。
- 逐个打开对应轨迹目录中的 `traj_data.pkl`。
- 优先用 `position` 长度统计轨迹步数；没有 `position` 时用 `yaw`。
- 如果 `traj_data.pkl` 中有时间戳字段，则直接用时间戳计算时长。
- 如果没有时间戳，则用 `total_steps / Hz / 3600` 换算小时数。
- 输出每个数据集的 split 行、每个数据集总计行，以及所有选中数据集的
  `ALL total` 总计行。
- 表格前会输出 `Duration summary` 汇总块，直接列出每个选中数据集的总小时数和
  `ALL` 总小时数。
- 默认把统计结果写入 `test_tools_logs/datasets_statistics.txt`。
- 每次生成新统计前会先删除旧输出文件，避免旧内容残留。
  新报告会先写入临时文件，再原子替换到目标路径。

默认运行命令：

```bash
python3 train_tools/dataset_duration.py --hz 4
```

如果不同数据集处理时的采样频率不同，使用每个数据集单独的 Hz：

```bash
python3 train_tools/dataset_duration.py \
  --dataset-hz recon=4 \
  --dataset-hz go_stanford=4 \
  --dataset-hz sacson=4 \
  --dataset-hz scand=10
```

只统计指定数据集或 split：

```bash
python3 train_tools/dataset_duration.py --datasets recon sacson --splits train --hz 4
```

写入自定义输出路径：

```bash
python3 train_tools/dataset_duration.py --hz 4 --output test_tools_logs/custom_statistics.txt
```

只打印，不写入文件：

```bash
python3 train_tools/dataset_duration.py --hz 4 --no-output
```

强制不使用 split 缓存索引，而是逐个读取 `traj_data.pkl`：

```bash
python3 train_tools/dataset_duration.py --hz 4 --no-cached-index
```

注意：

- 当前 FlowNav 格式的 `traj_data.pkl` 通常只有 `position` 和 `yaw`，没有时间戳；
  这种情况下必须提供 `--hz` 或 `--dataset-hz`，才能得到真实小时数。
- 缓存索引路径速度更快，但不会检查每个 `traj_data.pkl` 内部是否有时间戳字段；
  如果需要基于时间戳计算时长，使用 `--no-cached-index`。
- 脚本对数据集文件是只读的；除非使用 `--no-output`，否则只会在
  `test_tools_logs/` 下写统计文本文件。写入前会先删除所选路径上的旧输出文件。
