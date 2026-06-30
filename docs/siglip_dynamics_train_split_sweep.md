# 固定验证集/测试集的数据量扫描训练评测说明（train_split_sweep）

入口：
- [train_split_sweep.py](python -m siglip_dynamics.train_split_sweep)
- 实际实现：[train_split_sweep.py](src/siglip_dynamics/train_split_sweep.py)

本脚本用于回答：
- 在 `mlp / transformer / dit` 三种结构下，训练数据量增加时，预测 `delta_z` 的能力如何变化？
- 到底需要用到多少训练数据，测试集上的余弦相似度才达到较稳定、较高的水平？

## 1. 当前实验逻辑

这版脚本已经改成更严格、更公平的切分方式：

1. 先按 `split_seed` 随机打乱全部 episode 顺序
2. 固定划分出：
   - `test episodes`
   - `val episodes`
   - `train pool episodes`
3. 对于每个 `train_ratio`：
   - 只从固定的 `train pool` 里取前 `x%` episode 做真正训练
   - `val` 和 `test` 全程不变
4. 如果传多个 `split_seed`：
   - 会重复整套固定切分实验
   - 最后自动输出跨 seed 的平均结果和标准差

也就是说，这个脚本不是：
- 小 ratio 训完后继续拿同一个模型增量训练

而是：
- 每个 ratio 都从头初始化一个新模型
- 在相同的 `val/test` 上独立训练和评测

这样更适合做论文里的“数据需求曲线”。

## 2. train / val / test 怎么划分

先固定 episode 级切分：

- `--test-episode-ratio`
  - 固定测试集比例
- `--val-episode-ratio`
  - 固定验证集比例
- 剩下的 episode 组成固定 `train pool`

然后对每个 `train_ratio`：

- `train_ratio=0.1`
  - 使用固定 `train pool` 的前 10%
- `train_ratio=0.5`
  - 使用固定 `train pool` 的前 50%
- `train_ratio=1.0`
  - 使用整个固定 `train pool`

所以这里的 `train_ratio` 含义是：
- 相对于固定 `train pool` 的比例
- 不是相对于全数据集的比例

脚本会同时输出：
- `actual_train_pool_ratio`
- `actual_train_ratio`
- `actual_val_ratio`
- `actual_test_ratio`

方便你区分“占 train pool 的比例”和“占全数据集的比例”。

## 3. 主评估指标

默认重点看测试集上的：
- `test_cosine_mean`
- `test_cosine_median`

同时也会输出：
- `test_mse`
- `test_cosine_std`
- `test_cosine_p25 / p75`
- `test_pred_norm / test_target_norm`

其中：
- 余弦相似度越接近 `1.0` 越好

## 4. 保存的结果

输出目录结构示意：

```text
<output_dir>/
  split_manifest.json
  summary.json
  summary.csv
  summary_aggregated.json
  summary_aggregated.csv
  summary_plots/
    mlp_cosine_vs_ratio.png
    mlp_mse_vs_ratio.png
    transformer_cosine_vs_ratio.png
    dit_cosine_vs_ratio.png
    ...
  mlp/
    seed_42/
      r0.1/
        best_model.pt
        history.json
        run_config.json
        result_metrics.json
        test_tsne_subset.npz
        test_tsne.png
        test_tsne_points.csv
      r0.25/
    seed_123/
      ...
  transformer/
  dit/
```

### 关键文件说明

- `split_manifest.json`
  - 记录每个 `split_seed` 对应的固定 `train pool / val / test` episode

- `summary.csv`
  - 每次单独实验一行
  - 如果有多个 `split_seed`，这里会有多行同 ratio 结果

- `summary_aggregated.csv`
  - 按 `(model_type, train_ratio)` 聚合后的均值、方差、最小值、最大值
  - 这是最适合后续论文作图和汇总的总表

- `test_tsne_subset.npz`
  - 默认保存的轻量 t-SNE 子集
  - 只包含画 t-SNE 需要的采样预测/真值特征
  - 包含：
    - `pred`
    - `target`
    - `sample_index`
    - `episode_id`

- `test_predictions.npz`
  - 默认不保存
  - 只有显式传入 `--save-full-predictions-npz` 才会写出
  - 会非常大，因为它包含整份测试集的预测和真值张量

- `test_tsne.png`
  - 当前 ratio 测试集的预测/真值 t-SNE 可视化
  - 默认先对 token 维做 mean pooling，再做 PCA+t-SNE，保证 CPU 也能跑得动

## 5. 依赖建议

训练主流程依赖：
- `torch`
- `numpy`

如果你想自动出图，还需要：
- `matplotlib`
- `scikit-learn`
- `tqdm`

如果环境缺少 `matplotlib` 或 `scikit-learn`：
- 训练、验证、测试预测保存仍然会正常完成
- 只是自动跳过 `t-SNE` 和 `summary_plots`

## 5.1 大文件说明

当前脚本已经改成：
- 默认不再保存全量 `test_predictions.npz`
- 默认只保存 `test_tsne_subset.npz`、`test_tsne.png`、`test_tsne_points.csv` 和 summary 日志

因此你直接运行：

```bash
python train_split_sweep.py ...
```

如果没有额外传：

```bash
--save-full-predictions-npz
```

就不会再生成几十 GB 的预测缓存文件。

## 6. 常用参数

- `--dataset-path`
  - 已提取缓存目录，里面要有 `metadata.json / z_q.npy / z_scale.npy / actions.npy`

- `--output-dir`
  - 输出目录

- `--model-types`
  - 可一次传多个：`mlp transformer dit`

- `--train-ratios`
  - 相对于固定 train pool 的比例，例如：`0.1 0.25 0.5 0.75 1.0`

- `--val-episode-ratio`
  - 固定验证集比例

- `--test-episode-ratio`
  - 固定测试集比例

- `--min-val-episodes`
  - 固定验证集至少保留多少个 episode

- `--min-test-episodes`
  - 固定测试集至少保留多少个 episode

- `--min-train-episodes`
  - 每个 ratio 至少使用多少个训练 episode

- `--split-seeds`
  - 多随机种子重复实验，例如：`42 123 999`

- `--k-step`
  - 预测 `t -> t+k`

- `--mlp-h-window`
  - MLP 历史窗口，只能是 `1`

- `--seq-h-window`
  - Transformer / DiT 的历史窗口

- `--save-full-predictions-npz`
  - 显式保存全量 `test_predictions.npz`
  - 默认关闭
  - 只在确实需要逐样本全量后处理时再打开

## 7. 使用示例

### 7.1 只跑 MLP，固定 val/test，扫描 train pool 比例

```bash
cd <REPO_ROOT>
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --output-dir <SAFETY_CHECKPOINT_DIR>/sweep_mlp_metaworld_pi05 \
  --model-types mlp \
  --train-ratios 0.1 0.25 0.5 0.75 1.0 \
  --val-episode-ratio 0.1 \
  --test-episode-ratio 0.1 \
  --min-val-episodes 5 \
  --min-test-episodes 5 \
  --min-train-episodes 5 \
  --split-seeds 42 \
  --k-step 10 \
  --mlp-h-window 1 \
  --batch-size 512 \
  --eval-batch-size 512 \
  --epochs 30 \
  --lr 1e-3 \
  --train-loss-type cosine
```

### 7.2 一次同时跑三种结构

```bash
cd <REPO_ROOT>
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --output-dir <SAFETY_CHECKPOINT_DIR>/sweep_all_metaworld_pi05 \
  --model-types mlp transformer dit \
  --train-ratios 0.1 0.25 0.5 0.75 1.0 \
  --val-episode-ratio 0.1 \
  --test-episode-ratio 0.1 \
  --min-val-episodes 5 \
  --min-test-episodes 5 \
  --min-train-episodes 5 \
  --split-seeds 42 \
  --k-step 10 \
  --mlp-h-window 1 \
  --seq-h-window 5 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --epochs 30 \
  --lr 3e-4 \
  --train-loss-type both
```

### 7.3 多随机种子重复实验

```bash

cd <REPO_ROOT>
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --output-dir <SAFETY_CHECKPOINT_DIR>/sweep_all_metaworld_pi05_multiseed \
  --model-types mlp \
  --train-ratios 0.8 1.0 \
  --val-episode-ratio 0.1 \
  --test-episode-ratio 0.1 \
  --min-val-episodes 5 \
  --min-test-episodes 5 \
  --min-train-episodes 5 \
  --split-seeds 42 \
  --k-step 10 \
  --lr 3e-4 \
  --train-loss-type mse \
  --mlp-h-window 1 \
  --seq-h-window 1
```

## 8. 如何解读结果

优先看：
- `summary_aggregated.csv`
- `summary_plots/*_cosine_vs_ratio.png`

如果随着 `train_ratio` 变大：
- `test_cosine_mean_mean` 提升
- `test_mse_mean` 下降
- 且跨 seed 的 `std` 不大

说明：
- 模型确实从更多训练数据中获益
- 当前结构在这个数据规模下是有效的

如果曲线很早就饱和：
- 继续增加训练数据收益有限
- 可能受模型容量、特征质量或 `k_step` 限制

如果不同 seed 波动很大：
- 说明当前数据量还不稳定
- 或者当前 split 难度差异较大
- 应优先增大数据量、增加 seed 数或重新检查特征抽取质量
