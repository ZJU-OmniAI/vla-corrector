# SigLIP Dynamics 多后端提取/训练说明（LIBERO + MetaWorld）

本目录已支持：
- 视觉编码后端：`pi05` / `smolvla` / `xvla`
- 数据环境：`libero` / `metaworld`
- 动力学模型：`mlp` / `transformer` / `dit`

运行环境（重要）：
- 提取/校验请使用：`python`（Python 3.12）
- 不要使用 `openpi/.venv` 的 Python 3.11 去跑 `pi05/smolvla/xvla` 提取

## 1. 关键预处理对齐（本项目 vs 官方）

### PI05
- 本项目实现：`siglip_dynamics/extract.py`
  - 输入统一与布局转换：`_to_bchw_float`（行 44）
  - 值域归一：`_normalize_to_zero_one`（行 63）
  - PI05预处理：`Pi05Extractor._prepare_image`
  - resize_with_pad 等价实现：`Pi05Extractor._resize_with_pad_zero_one`
  - 模型加载：`PI05Policy.from_pretrained`（lerobot）
- 官方实现：
  - `lerobot/policies/pi05/modeling_pi05.py:1142` `_preprocess_images`
  - `lerobot/policies/pi05/modeling_pi05.py:151` `resize_with_pad_torch`

对应关系：
- 形状处理（BCHW/BHWC）: 对齐
- resize + pad 策略: 对齐
- `[0,1] -> [-1,1]` 归一化: 对齐

### SmolVLA
- 本项目实现：`siglip_dynamics/extract.py`
  - 预处理：`SmolVLAExtractor._prepare_image`
  - 视觉embedding：`SmolVLAExtractor.encode_single_image`
- 官方实现：
  - `lerobot/policies/smolvla/modeling_smolvla.py:404` `prepare_images`
  - `lerobot/policies/smolvla/smolvlm_with_expert.py:179` `embed_image`

对应关系：
- resize_with_pad(pad=0): 对齐
- `[0,1] -> [-1,1]`: 对齐
- 调用视觉塔 embedding 路径: 对齐

### XVLA
- 本项目实现：`siglip_dynamics/extract.py`
  - 预处理：`XVLAExtractor._prepare_image`
  - 视觉embedding：`XVLAExtractor.encode_single_image`
- 官方实现：
  - `lerobot/policies/xvla/processor_xvla.py:275` `XVLAImageToFloatProcessorStep`
  - `lerobot/policies/xvla/processor_xvla.py:349` `XVLAImageNetNormalizeProcessorStep`
  - `lerobot/policies/xvla/modeling_xvla.py:306` `_prepare_images`
  - `lerobot/policies/xvla/modeling_xvla.py:157` `forward_vlm`（内部 `vlm._encode_image`）

对应关系：
- `[0,255]/[0,1] -> [0,1]`: 对齐
- ImageNet normalize: 对齐
- optional resize_with_pad: 对齐
- 视觉编码入口: 对齐

## 2. 提取命令（extract）

> 入口：`python -m siglip_dynamics.extract`

推荐增加（防止污染源目录）：
```bash
  --dataset-loader parquet
```
**参数统一说明：**
- 所有后端（PI05/SmolVLA/XVLA）现在统一使用：
  - `--encoder-policy-path` 指定模型路径
  - `--encoder-local-files-only` 仅使用本地文件
  - `--encoder-revision` 指定模型版本
- PI05 兼容旧参数 `--local-pi0-checkpoint-dir`（优先使用新参数）

仅当你明确希望自动补齐下载缺失分片时，才加：
```bash
  --allow-dataset-download
```

### 2.1 PI05 + LIBERO
```bash
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR>/lerobot_libero \
  --dataset-repo-id lerobot/lerobot_libero \
  --dataset-loader parquet \
  --dataset-format libero \
  --output-path <EXTRACTED_CACHE_DIR>/extracted_libero_pi05 \
  --encoder-backend pi05 \
  --encoder-policy-path <POLICY_CHECKPOINT_DIR>/pi05_libero_base \
  --encoder-local-files-only
```

### 2.2 PI05 + MetaWorld
```bash
export CUDA_VISIBLE_DEVICES=4
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR>/mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --output-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --encoder-backend pi05 \
  --use-normalized-delta-action \
  --encoder-policy-path <POLICY_CHECKPOINT_DIR>/pi05_metaworld/060000 \
  --encoder-local-files-only
```

### 2.3 SmolVLA + LIBERO / MetaWorld
```bash
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR>/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --output-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_smolvla \
  --encoder-backend smolvla \
  --use-normalized-delta-action \
  --encoder-policy-path <POLICY_CHECKPOINT_DIR>/smolvla_base \
  --encoder-local-files-only
```

### 2.4 XVLA + LIBERO / MetaWorld
```bash
export CUDA_VISIBLE_DEVICES=4,5
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR>/mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --output-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_xvla \
  --encoder-backend xvla \
  --use-normalized-delta-action \
  --encoder-policy-path <POLICY_CHECKPOINT_DIR>/xvla_mt50 \
  --encoder-local-files-only
```

## 3. 提取结果校验（verify_extraction）

> 入口：`python -m siglip_dynamics.verify_extraction`

### 3.1 结构校验（不重算embedding）
```bash
python -m siglip_dynamics.verify_extraction \
  --dataset-path <DATASET_DIR>/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --extracted-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --num-samples 0
```

### 3.2 严格一致性校验（重算embedding）
- 默认从 `metadata.json` 读取 backend 和模型信息；也可显式覆盖。

```bash
python -m siglip_dynamics.verify_extraction \
  --dataset-path <DATASET_DIR>/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --extracted-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --num-samples 8 \
  --encoder-backend pi05 \
  --local-pi0-checkpoint-dir <POLICY_CHECKPOINT_FILE>
```

## 4. 训练命令（MLP / Transformer / DiT）

> 入口：`python -m siglip_dynamics.train`

说明：训练会自动读取 `metadata.json` 对齐 `token_dim/action_dim`，不用手动改维度。

### 4.1 DiT
```bash
export CUDA_VISIBLE_DEVICES=3,4,5
torchrun --nproc_per_node=2 -m siglip_dynamics.train \
  --model-type dit \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --k-step-list 10 \
  --h-window-list 5 \
  --batch-size 64 \
  --epochs 30 \
  --train-loss-type cosine \
  --log-shapes-every-epoch \
  --checkpoint-dir <SAFETY_CHECKPOINT_DIR>/my_dit_metaworld \
  --wandb-project siglip-dynamics \
  --wandb-run-name dit_metaworld
```

### 4.2 MLP
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29502 -m siglip_dynamics.train \
  --model-type mlp \
  --h-window 1 \
  --k-step-list 10 \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --batch-size 512 \
  --epochs 30 \
  --train-loss-type cosine \
  --checkpoint-dir <SAFETY_CHECKPOINT_DIR>/my_mlp_pi05_metaworld

```

### 4.3 Transformer
```bash
torchrun --nproc_per_node=4 -m siglip_dynamics.train \
  --model-type transformer \
  --h-window-list 3 5 \
  --k-step-list 10 20 \
  --dataset-path <EXTRACTED_CACHE_DIR>/extracted_metaworld_pi05 \
  --batch-size 64 \
  --epochs 30 \
  --checkpoint-dir <SAFETY_CHECKPOINT_DIR>/my_transformer_metaworld
```

## 5. 日志检查要点

- 提取阶段会打印：
  - `Resolved layout ...`（键名解析结果）
  - `Target cache shape: frames=... L=... D=... action_dim=...`
- 校验阶段会打印：
  - `Cache summary: frames/L/D/action_dim`
  - 每个样本 `cosine` 和 `mae`
- 训练阶段会打印：
  - `Dataset metadata ...`
  - `Auto-align action_dim/token_dim ...`
  - `Model params ...`

## 6. 输出格式

提取输出目录包含：
- `z_q.npy`（int8, [N,L,D]）
- `z_scale.npy`（float16, [N,L]）
- `actions.npy`（float16, [N,A]）
- `episode_index.npy`（int32, [N]）
- `metadata.json`（含 backend、预处理、官方对齐引用、维度信息）
