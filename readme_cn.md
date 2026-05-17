# Quick Start

## Demo

HunyuanVideo 1.5
<video src="assets/hy_demo.mp4" controls width="100%"></video>

## 1. Install

```bash
conda create -n minwm python=3.10 -y 
conda activate minwm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
```

## 2. Download Inference Checkpoints

只下你要跑的那一条。所有权重统一落到 `./ckpts/`。

### 2.1 HY1.5 base（HY Action2V / HY TI2V 都需要）

```bash
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm
hf download google/byt5-small           --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir ./ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2
hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip --token <your_hf_token>
```

### 2.2 Wan2.1 base

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B 

# 代码硬编码加载路径，做软链
mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B
```

### 2.3 Stage ckpts（按推理任务挑一条）

```bash
# HY Action2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/dmd/*"

# HY TI2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/TI2V/dmd/*"

# Wan Action2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/dmd/*"
```

## 3. Inference

### 3.1 HY Action2V（4-step DMD，相机控制）

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_action2v \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

> 相机轨迹在 `assets/example.json` 中通过 `"trajectory"` 字段逐样本指定。格式：`w/s/a/d` 键位 + `*N` 重复次数，逗号分隔。示例：`"a*4,w*8,s*7"`。脚本自动读取每个样本的轨迹。

### 3.2 HY TI2V（4-step DMD）

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

### 3.3 Wan Action2V（4-step DMD，相机控制）

```bash 
OUTPUT_FOLDER=./outputs/quickstart_wan_action2v \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```
---

## 4. Training

> 训练流程覆盖三大模型：HY Action2V、HY TI2V、Wan Action2V。
> 每个模型分两个 Phase：**Phase 1 Bidirectional SFT**（双向多步基座）和 **Phase 2 Causal Forcing**（蒸馏到因果少步）。
> Phase 2 包含 4 个 Stage：Stage 1 Teacher Forcing AR Diffusion、Stage 2(a) Causal ODE Distillation Initialization、Stage 2(b) Causal Consistency Distillation、Stage 3 Asymmetric DMD with Self Rollout。
> 每个小节统一组织为：**(1) 模型预下载**、**(2) 数据准备**、**(3) 训练脚本**、**(4) 结果验证**。

### 4.1 HY Action2V

#### 4.1.1 Phase 1: Bidirectional SFT

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**


```bash
hf download tencent/HunyuanVideo-1.5 \
    --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"

hf download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm

hf download google/byt5-small \
    --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small

modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir ./ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2

hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip \
    --token <your_hf_token>
```

**(2) 数据准备**

二选一。所有数据统一落到 `./dataset/`。

**Option A：下载 minWM-dataset**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "preencode_input.json" "videos/**"
    
```

下完后磁盘布局：

```
./dataset/
├── preencode_input.json
└── videos/
    ├── 000000_right8a11/gen.mp4
    ├── 000001_w10d9/gen.mp4
    └── ...
```

**Option B：自备视频与轨迹**

按 Option A 的布局自备 `preencode_input.json` 与 `videos/` 目录。`preencode_input.json` 是 list，每条至少含 `image_path` / `caption` / `pose_str`：

```json
[
    {
        "image_path": "/abs/path/to/image1.png",
        "caption": "A scenic mountain view",
        "pose_str": "right-8, a-11"
    }
]
```

视频目录命名规则：`{i:06d}_{slug(pose_str)}/gen.mp4`，`i` = JSON 顺序下标，`slug` = `pose_str` 转小写后去掉非字母数字字符。

**Final: 编码（两种 Option 共用）**

脚本默认 `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`、`INPUT_DIR=./dataset`、`OUTPUT_DIR=./dataset/HY15/Action2V`：

```bash
bash HY15/scripts/data_preprocessing/run_preencode_downloaded_camera_video.sh
```

CFG 训练用的 negative prompt embeddings 单独下载：

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"
```

最终布局：

```
./dataset/
├── HY15/Action2V/                       # 编码产物
│   ├── latents/
│   └── train_index.json
└── others/HY/Action2V/                  # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) 训练脚本**

Bidirectional + camera (ProPE) 训练。模型类 `HunyuanTransformer3DARActionProPEModel`，使用带 viewmats/Ks 的相机数据加载器，`flow matching` MSE loss：

```bash
bash HY15/scripts/training/hyvideo15/run_bi_camera_multinode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/Action2V/bidirectional/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容（`diffusion_pytorch_model.safetensors` + `config.json` 等）提到 `./ckpts/HY15/Action2V/bidirectional/`，与 §2.3 / §4.1.2 Stage 1 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/Action2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/bidirectional/
```

**(4) 结果验证**

复用 §3.1 推理脚本，但用 50-step bidirectional 模式。脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional`（即本阶段产物，也是 §4.1.2 Stage 1 的预下载路径），用环境变量覆盖即可指向其他 ckpt：

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_camera \
    bash HY15/scripts/inference/run_infer_bidirectional_camera.sh
```

</details>

#### 4.1.2 Phase 2: Causal Forcing

##### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/bidirectional/**"
```


**(2) 数据准备**

同 §4.1.1 (2)。复用同一份 `./dataset/HY15/Action2V/{latents, train_index.json}` 与 `./dataset/others/HY/Action2V/`。

**(3) 训练脚本**

把 Phase 1 的双向模型转为因果 + teacher forcing AR。复用相同的 ProPE 模型类与相机数据加载器，loss 仍是 flow matching MSE：

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_mem_multinode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/Action2V/ar_diffusion_tf/`，与 §4.1.2 Stage 2(a) (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/ar_diffusion_tf/
```

**(4) 结果验证**

50-step AR rollout 模式。脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf`（即本阶段产物，也是 Stage 2(a) 的预下载路径），其他变量同样支持环境变量覆盖：

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_camera \
    bash HY15/scripts/inference/run_infer_ar_diffusion_camera.sh
```

</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/ar_diffusion_tf/**"
```

**(2) 数据准备**

二选一。所有数据落到 `./dataset/HY15/Action2V_ode/`，与 §4.1.1 的 SFT latents (`./dataset/HY15/Action2V/`) 平级、不互相覆盖。Negative prompts 复用 §4.1.1 已下载的 `./dataset/others/HY/Action2V/`。

**Option A：下载 minWM-dataset 预生成 ODE latents**

```bash
# 1) 从 HF 下载（仓库结构落到 ODE_data/）
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/Action2V/**"

# 2) 搬到统一布局 ./dataset/HY15/Action2V_ode/
mkdir -p ./dataset/HY15/Action2V_ode
mv ./dataset/ODE_data/HY15/Action2V/latents ./dataset/HY15/Action2V_ode/latents


# 3) 在新位置重新生成绝对路径 index（落到顶层 ./dataset/HY15/Action2V_ode/train_index.json）
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

**Option B：从 Stage 1 ckpt 自己采样**

需要 §4.1.2 Stage 1 训练完成的 ckpt（落点 `./ckpts/HY15/Action2V/ar_diffusion_tf/`），再用 SFT 编码数据 (§4.1.1) 跑 48 步 CFG 采样：

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/Action2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/Action2V_ode/latents \
    bash HY15/scripts/training/hy15_camera/ode_sampling.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

最终布局：

```
./dataset/
├── HY15/
│   ├── Action2V/                        # SFT latents（来自 §4.1.1）
│   │   ├── latents/
│   │   └── train_index.json
│   └── Action2V_ode/                    # ODE latents（本节）
│       ├── latents/
│       └── train_index.json
└── others/HY/Action2V/                  # neg prompts（复用 §4.1.1）
```

**(3) 训练脚本**

ODE regression：在 §4.1.2 Stage 2(a) (2) 准备好的 ODE latents (`./dataset/HY15/Action2V_ode/`) 上，让模型在关键 timestep [0, 12, 24, 36, -2, -1]（对应 [1000, 750, 500, 250] 等）上直接回归 ODE 求解器输出：

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_ode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/Action2V/causal_ode/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/Action2V/causal_ode/`，与 §4.1.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_ode/
```

**(4) 结果验证**

4-step DMD 推理脚本（`run_infer_causal_camera.sh`）被 Stage 2(a) / 2(b) / 3 共用，默认 `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd`（最终产物，与 §3.1 quickstart 一致）。本阶段验证用 env 切到 causal_ode：

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```


</details>

##### Stage 2(b): Causal Consistency Distillation

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

同 Stage 2(a).

**(2) 数据准备**

同 §4.1.1 (2)。复用同一份编码后的数据。

**(3) 训练脚本**

Consistency distillation：引入冻结的教师模型与 EMA 目标网络，在相邻 timestep (t, t_next) 上强制学生输出一致。`trainer: consistency_distillation`：

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_cd.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/Action2V/causal_cd/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/Action2V/causal_cd/`，与 §4.1.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_cd/
```

**(4) 结果验证**

同 §4.1.2 Stage 2(a) (4)，env 切到本阶段产物 causal_cd：

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

##### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/causal_ode/**" # Or causal_cd
```

**(2) 数据准备**

同 §4.1.1 (2)。复用同一份 §4.1.1 编码后的 SFT latents `./dataset/HY15/Action2V/{latents, train_index.json}` 与 `./dataset/others/HY/Action2V/`。Stage 3 走 self rollout，DMD 训练只消耗 `train_index.json` 里的条件部分（首帧 image + caption + pose），不再监督真实视频 latents，也不消耗 §4.1.2 Stage 2(a) 的 ODE latents。

**(3) 训练脚本**

Asymmetric DMD with self rollout：score distillation，仅消耗条件不再监督真实视频 latents。学生网络在自身 rollout 上对齐 teacher 的 score field：

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_dmd.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/Action2V/dmd/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/Action2V/dmd/`，与 §3.1 quickstart / §4.1.2 Stage 3 (4) 的推理路径对齐：

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/dmd/
```

**(4) 结果验证**

直接 §3.1 quickstart 即可（脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd`，正是本阶段产物）：

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

### 4.2 HY TI2V

#### 4.2.1 Phase 1: Bidirectional SFT

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

同 4.1.1.

**(2) 数据准备**

二选一。所有数据统一落到 `./dataset/`。

**Option A：下载 minWM-dataset 预编码 latents**

> HF 远程目录名叫 `ODE_data/HY15/TI2V/`（toy 数据，约 5K，与 Causal ODE 阶段共享，故仓库就这么命名），下载后 mv 到本地 SFT 布局 `./dataset/HY15/TI2V/`。

```bash
# 1) 从 HF 下载（仓库结构落到 ODE_data/）
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/TI2V/**"

# 2) 搬到统一布局 ./dataset/HY15/TI2V/
mkdir -p ./dataset/HY15/TI2V
mv ./dataset/ODE_data/HY15/TI2V/latents ./dataset/HY15/TI2V

# 3) 在新位置重新生成绝对路径 index（落到顶层 ./dataset/HY15/TI2V/train_index.json）
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V/latents \
    -o ./dataset/HY15/TI2V/train_index.json
```

**Option B：自备视频 + 文本（从原始视频自行编码）**

输入 `./dataset/videos.json`（路径自行替换）：

```json
[
    {"video_path": "/abs/path/to/video1.mp4", "caption": "A cat playing"},
    {"video_path": "/abs/path/to/video2.mp4", "caption": "A sunset"}
]
```

脚本默认 `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`、`INPUT_JSON=./dataset/videos.json`、`OUTPUT_DIR=./dataset/HY15/TI2V`：

```bash
bash HY15/scripts/data_preprocessing/run_preencode_video.sh
```

**Final: CFG negative prompt embeddings（两种 Option 共用）**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/TI2V/**"
```

最终布局（统一为 `./dataset/HY15/TI2V/`，Option A/B 一致）：

```
./dataset/
├── HY15/TI2V/                           # 编码 latents（A：HF 下载后搬运；B：本地编码产出）
│   ├── latents/
│   └── train_index.json
└── others/HY/TI2V/                      # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) 训练脚本**

Bidirectional TI2V SFT。模型类 `HunyuanTransformer3DARActionModel`，全局（非因果）注意力，flow matching MSE loss：

```bash
bash HY15/scripts/training/hyvideo15/run_bi_hunyuan_mem_multinode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/TI2V/bidirectional/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容（`diffusion_pytorch_model.safetensors` + `config.json` 等）提到 `./ckpts/HY15/TI2V/bidirectional/`，与 §4.2.2 Stage 1 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/TI2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/bidirectional/
```

**(4) 结果验证**

复用 §3.2 推理脚本的 50-step bidirectional 模式。脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional`（即本阶段产物，也是 §4.2.2 Stage 1 的预下载路径），用环境变量覆盖即可指向其他 ckpt：

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_ti2v \
    bash HY15/scripts/inference/run_infer_bidirectional.sh
```

</details>

#### 4.2.2 Phase 2: Causal Forcing

##### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt (TI2V 可直接用官方双向模型初始化，不过这里也给出一个我们微调的版本，以使完整):

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/bidirectional/**"
```

**(2) 数据准备**

同 §4.2.1 (2)。复用同一份 `./dataset/HY15/TI2V/{latents, train_index.json}` 与 `./dataset/others/HY/TI2V/`。

**(3) 训练脚本**

把 Phase 1 的双向 TI2V 模型转为因果 + teacher forcing AR。loss 仍是 flow matching MSE：

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_mem_multinode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/TI2V/ar_diffusion_tf/`，与 §4.2.2 Stage 2(a) (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/ar_diffusion_tf/
```

**(4) 结果验证**

50-step AR rollout 模式。脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf`（即本阶段产物，也是 Stage 2(a) 的预下载路径），其他变量同样支持环境变量覆盖：

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_ti2v \
    bash HY15/scripts/inference/run_infer_ar_diffusion.sh
```

</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt :

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/ar_diffusion_tf/**"
```

**(2) 数据准备**

二选一。Option A 直接复用 §4.2.1 (2) 已下载的 toy ODE 数据；Option B 用 Stage 1 ckpt 自采 ODE 轨迹（落到独立目录 `./dataset/HY15/TI2V_ode/`，避免与 §4.2.1 混合）。Negative prompts 复用 §4.2.1 已下载的 `./dataset/others/HY/TI2V/`。

**Option A：复用 §4.2.1 已下载的 ODE 数据**

§4.2.1 Option A 下载的 `ODE_data/HY15/TI2V/`（HF 仓库就以此命名，是与本阶段共享的 toy ODE 数据）已经被 mv 到 `./dataset/HY15/TI2V/{latents, train_index.json}`，本阶段直接复用即可，无需额外动作。

> 与 §4.1.2 Stage 2(a)（HY Action2V）的差异：Action2V 的 SFT latents 与 ODE latents 是两份独立数据（分别落 `./dataset/HY15/Action2V/` 与 `./dataset/HY15/Action2V_ode/`）；TI2V 因为 toy dataset 本身就是 ODE 数据，§4.2.1 Option A 已经把它当作 SFT latents 用，本阶段直接共享同一份。

**Option B：从 Stage 1 ckpt 自己采样**

需要 §4.2.2 Stage 1 训练完成的 ckpt（落点 `./ckpts/HY15/TI2V/ar_diffusion_tf/`），再用 SFT 编码数据 (§4.2.1) 跑 48 步 CFG 采样：

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/TI2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/TI2V_ode/latents \
    bash HY15/scripts/ode_sampling/ode.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V_ode/latents \
    -o ./dataset/HY15/TI2V_ode/train_index.json
```

最终布局（Option B 落点）：

```
./dataset/
├── HY15/
│   ├── TI2V/                            # SFT latents（来自 §4.2.1）
│   │   ├── latents/
│   │   └── train_index.json
│   └── TI2V_ode/                        # ODE latents（本节 Option B 产出）
│       ├── latents/
│       └── train_index.json
└── others/HY/TI2V/                      # neg prompts（复用 §4.2.1）
```

**(3) 训练脚本**

ODE regression：在 §4.2.2 Stage 2(a) (2) 准备好的 ODE latents 上，让模型在关键 timestep 上直接回归 ODE 求解器输出：

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_ode.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/TI2V/causal_ode/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/TI2V/causal_ode/`，与 §4.2.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/TI2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_ode/
```

**(4) 结果验证**

4-step DMD 推理脚本（`run_infer_causal.sh`）被 Stage 2(a) / 2(b) / 3 共用，默认 `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd`（最终产物，与 §3.2 quickstart 一致）。本阶段验证用 env 切到 causal_ode：

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

##### Stage 2(b): Causal Consistency Distillation

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

同 Stage 2(a).

**(2) 数据准备**

同 §4.2.1 (2)。复用同一份编码后的数据。

**(3) 训练脚本**

Consistency distillation：`trainer: consistency_distillation`，使用冻结教师 + EMA 目标网络，在相邻 timestep (t, t_next) 上强制学生输出一致：

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_cd.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/TI2V/causal_cd/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/TI2V/causal_cd/`，与 §4.2.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/TI2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_cd/
```

**(4) 结果验证**

同 §4.2.2 Stage 2(a) (4)，env 切到本阶段产物 causal_cd：

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

##### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt :

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/causal_ode/**" # or causal_cd
```

**(2) 数据准备**

同 §4.2.1 (2)。复用同一份 §4.2.1 编码后的 SFT latents: `./dataset/HY15/TI2V/{latents, train_index.json}` 与 `./dataset/others/HY/TI2V/`。Stage 3 走 self rollout，DMD 训练只消耗 `train_index.json` 里的条件部分（首帧 image + caption），不再监督真实视频 latents。

**(3) 训练脚本**

Asymmetric DMD with self rollout。score distillation，仅消耗 `train_index.json` 里的条件部分（首帧 image + caption），不再监督真实视频 latents：

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_dmd.sh
```

脚本默认把 ckpt 落在 `./ckpts/HY15/TI2V/dmd/checkpoint-XXXX/transformer/`。训练完成后挑最佳 step，把整份 `transformer/` 内容提到 `./ckpts/HY15/TI2V/dmd/`，与 §3.2 quickstart 的推理路径对齐：

```bash
BEST_STEP=XXXX  # 按 validation 选
mv ./ckpts/HY15/TI2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/dmd/
```

**(4) 结果验证**

直接 §3.2 quickstart 即可（脚本默认 `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd`，正是本阶段产物）：

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

### 4.3 Wan Action2V

#### 4.3.1 Phase 1: Bidirectional SFT

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir ./ckpts/Wan2.1-T2V-1.3B \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" "diffusion_pytorch_model.safetensors" "config.json"
```

**(2) 数据准备**

视频来源（Option A 下载 / Option B 自备）与磁盘布局**同 §4.1.1 (2)**，复用同一份 `./dataset/preencode_input.json` + `./dataset/videos/`。差异只在编码：

- HY 编码后落 `<OUTPUT_DIR>/latents/`（per-sample `.pt`），
- Wan 编码后落 `<OUTPUT_DIR>/data/`（合并后的 LMDB），下游训练入口直接吃 LMDB。

**编码（两种 Option 共用）**

VAE 走 Wan2.1 基础模型（已按 §2.2 软链到 `Wan21/wan_models/Wan2.1-T2V-1.3B/`）。脚本默认值：

```
VAE_PATH=Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
INPUT_JSON=./dataset/preencode_input.json
VIDEO_DIR=./dataset/videos
OUTPUT_DIR=./dataset/Wan21/Action2V
```

直接跑：

```bash
bash Wan21/scripts/data_preprocessing/run_build_worldplaygen_lmdb.sh
```

合并后的 LMDB 落在 `./dataset/Wan21/Action2V/data/`。

> Wan 的 CFG negative prompt 写在 config (`Wan21/configs/causal_forcing_dmd_camera.yaml:38`)，不需预编码 `.pt`，故无 HY 那种 `others/HY/Action2V/` 的下载步骤。

最终布局：

```
./dataset/
├── preencode_input.json                 # 同 §4.1.1
├── videos/                              # 同 §4.1.1
└── Wan21/Action2V/
    └── data/                            # 编码后的 LMDB
```

**(3) 训练脚本**

Bidirectional + camera (PRoPE) SFT。`trainer: bidirectional_diffusion`，使用 `WanDiffusionWrapper(use_camera=True)` 与 `CameraLatentLMDBDataset`（携带 viewmats/Ks），flow matching loss：

```bash
bash Wan21/scripts/training/run_stage0_bidirectional_camera.sh
```

脚本默认把 ckpt 落在 `logs/bidirectional_camera/checkpoint_model_<step>/model.pt`。训练完成后挑最佳 step，把 `model.pt` 提到 `./ckpts/Wan21/Action2V/bidirectional/`，与 §4.3.2 Stage 1 (1) 的预下载格式对齐：

```bash
BEST_STEP=00x000  # 按 validation 选（6 位补零）
mkdir -p ./ckpts/Wan21/Action2V/bidirectional
mv logs/bidirectional_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/bidirectional/model.pt
```

**(4) 结果验证**

50-step bidirectional + camera 模式。脚本默认 `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/bidirectional/model.pt`（即本阶段产物，也是 §4.3.2 Stage 1 的预下载路径），用环境变量覆盖即可指向其他 ckpt：

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/bidirectional/model.pt \
OUTPUT_FOLDER=./outputs/eval_bidir_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_bidirectional_camera.sh
```

</details>

#### 4.3.2 Phase 2: Causal Forcing

##### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/bidirectional/**"
```

**(2) 数据准备**

同 §4.3.1 (2)。复用同一份 `./dataset/Wan21/Action2V/data/`（LMDB）。

**(3) 训练脚本**

把 Phase 1 的双向模型转为因果 + teacher forcing AR。模型切到 `CausalWanModel`，启用 `use_camera: true` 保留 PRoPE 投影参数，flow matching loss：

```bash
bash Wan21/scripts/training/run_stage1_ar_camera.sh
```

脚本默认把 ckpt 落在 `logs/ar_camera_tf/checkpoint_model_<step>/model.pt`。训练完成后挑最佳 step，把 `model.pt` 提到 `./ckpts/Wan21/Action2V/ar_diffusion_tf/`，与 §4.3.2 Stage 2(a) (1) 的预下载格式对齐：

```bash
BEST_STEP=00x000  # 按 validation 选（6 位补零）
mkdir -p ./ckpts/Wan21/Action2V/ar_diffusion_tf
mv logs/ar_camera_tf/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

**(4) 结果验证**

50-step AR + camera 模式。脚本默认 `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`（即本阶段产物，也是 Stage 2(a) 的预下载路径）：

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
OUTPUT_FOLDER=./outputs/eval_ar_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_ar_camera.sh
```


</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**


若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/**"
```

**(2) 数据准备**

二选一。所有数据落到 `./dataset/Wan21/Action2V/ode_lmdb/`，与 §4.3.1 的 SFT LMDB (`./dataset/Wan21/Action2V/data/`) 平级、不互相覆盖。

**Option A：下载 minWM-dataset 预生成 ODE latents 并本地合并 LMDB**

HF 上发布的是**未合并的 `.pt` latents**（`get_causal_ode_data_prope.py` 的产物），下载后还要本地跑一次合并：

```bash
# 1) 下载 .pt latents（HF 仓库结构落到 ODE_data/）
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/Wan21/Action2V/**"

# 2) 合并为训练用 LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/ODE_data/Wan21/Action2V \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

**Option B：从 Stage 1 ckpt 自己采样**

需要 §4.3.2 Stage 1 训练完成的 ckpt（落点 `./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`），再用 §4.3.1 编码出的 SFT LMDB (`./dataset/Wan21/Action2V/data/`) 跑 48 步 CFG 采样，最后合并为 LMDB：

```bash
# 1) 用 Stage 1 ckpt 跑 48 步 CFG 采样，得到 .pt latents
torchrun --nproc_per_node=8 Wan21/get_causal_ode_data_prope.py \
    --generator_ckpt ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
    --rawdata_path ./dataset/Wan21/Action2V/data \
    --output_folder ./dataset/Wan21/Action2V/ode_latents

# 2) 合并为训练用 LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/Wan21/Action2V/ode_latents \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

> 与 §4.1.2 Stage 2(a)（HY Action2V）的差异：HY 的 ODE 产物是 per-sample `.pt`，HF 下载后只需 `mv` + 重生 `train_index.json` 即可；Wan 因训练入口直接吃 LMDB，**两种 Option 都需要额外跑一次 `build_ode_prope_lmdb.py` 合并**。

最终布局：

```
./dataset/
├── preencode_input.json                # 同 §4.3.1
├── videos/                             # 同 §4.3.1
└── Wan21/Action2V/
    ├── data/                           # SFT LMDB（来自 §4.3.1）
    └── ode_lmdb/                       # ODE LMDB（本节产出，训练入口直接读这里）
```

**(3) 训练脚本**

ODE regression。`trainer: ode`，在 §4.3.2 Stage 2(a) (2) 准备好的 ODE LMDB (`./dataset/Wan21/Action2V/ode_lmdb/`) 上训练，由 `CameraODERegressionLMDBDataset` 读取并在关键 timestep [1000, 750, 500, 250] 上回归 ODE 求解器输出：

```bash
bash Wan21/scripts/training/run_stage2_causal_ode_camera.sh
```

脚本默认把 ckpt 落在 `logs/causal_ode_camera/checkpoint_model_<step>/model.pt`。训练完成后挑最佳 step，把 `model.pt` 提到 `./ckpts/Wan21/Action2V/causal_ode/`，与 §4.3.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=00x000  # 按 validation 选（6 位补零）
mkdir -p ./ckpts/Wan21/Action2V/causal_ode
mv logs/causal_ode_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_ode/model.pt
```

**(4) 结果验证**

`run_infer_causal_camera.sh` 默认是 4-step DMD config + dmd ckpt（与 §3.3 quickstart 一致）。本阶段验证需 env 同时切 `CONFIG_PATH` 与 `CHECKPOINT_PATH` 到 ODE 版本：

```bash 
CONFIG_PATH=Wan21/configs/causal_ode_camera.yaml \
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_ode/model.pt \
OUTPUT_FOLDER=./outputs/eval_causal_ode_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>

##### Stage 2(b): Causal Consistency Distillation

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

同 Stage 2.

**(2) 数据准备**

同 §4.3.1 (2)。复用同一份编码后的数据。

**(3) 训练脚本**

Consistency distillation。`trainer: consistency_distillation`，使用冻结教师 + EMA 目标网络，在相邻 timestep 对 (t, t_next) 上强制学生输出一致：

```bash
bash Wan21/scripts/training/run_stage2_causal_cd_camera.sh
```

脚本默认把 ckpt 落在 `logs/causal_cd_camera/checkpoint_model_<step>/model.pt`。训练完成后挑最佳 step，把 `model.pt` 提到 `./ckpts/Wan21/Action2V/causal_cd/`，与 §4.3.2 Stage 3 (1) 的预下载格式对齐：

```bash
BEST_STEP=00x000  # 按 validation 选（6 位补零）
mkdir -p ./ckpts/Wan21/Action2V/causal_cd
mv logs/causal_cd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_cd/model.pt
```

**(4) 结果验证**

同 §4.3.2 Stage 2(a) (4)，env 切到 CD config + CD ckpt：

```bash 
CONFIG_PATH=Wan21/configs/causal_cd_camera.yaml \
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_cd/model.pt \
OUTPUT_FOLDER=./outputs/eval_causal_cd_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>

##### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>展开详细步骤</b>（模型预下载 / 数据准备 / 训练脚本 / 结果验证）</summary>

**(1) 模型预下载**

若省略上一训练直接开始此阶段，需下载提供的上一阶段 ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/causal_ode/**"  # Or causal_cd
```

**(2) 数据准备**

同 §4.3.1 (2)。复用同一份 §4.3.1 编码后的 SFT LMDB (`./dataset/Wan21/Action2V/data/`)。Stage 3 走 self rollout，不再消耗 Stage 2(a) 的 ODE LMDB。

**(3) 训练脚本**

Asymmetric DMD with self rollout。`trainer: score_distillation`，仅消耗条件无需真实视频。学生网络在自身 rollout 上对齐 teacher 的 score field：

```bash
bash Wan21/scripts/training/run_stage3_causal_dmd_camera.sh # 建议 100~200 步
```

脚本默认把 ckpt 落在 `logs/causal_dmd_camera/checkpoint_model_<step>/model.pt`。训练完成后挑最佳 step，把 `model.pt` 提到 `./ckpts/Wan21/Action2V/dmd/`，与 §3.3 quickstart / §4.3.2 Stage 3 (4) 的推理路径对齐：

```bash
BEST_STEP=000x00  # 按 validation 选（6 位补零）
mkdir -p ./ckpts/Wan21/Action2V/dmd
mv logs/causal_dmd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/dmd/model.pt
```

**(4) 结果验证**

直接 §3.3 quickstart 即可（脚本默认 `CONFIG_PATH=causal_forcing_dmd_camera.yaml`、`CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt`，正是本阶段产物）：

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=./outputs/eval_dmd_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>


### 4.4 Wan T2V

本仓库暂未集成，具体训练推理和数据模型均见 [Causal-Forcing repo](https://github.com/thu-ml/Causal-Forcing).