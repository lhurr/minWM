# Quick Start [中文版](readme_cn.md)

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

Download only what you plan to run. All weights land under `./ckpts/`.

### 2.1 HY1.5 base (required by both HY Action2V and HY TI2V)

```bash
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*"  "scheduler/*" "transformer/480p_i2v/*"
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

# Code hardcodes the load path; create a symlink.
mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B
```

### 2.3 Stage ckpts (pick one per inference task)

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

### 3.1 HY Action2V (4-step DMD, camera control)

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_action2v \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

> Camera trajectories are specified per sample in `assets/example.json` via the `"trajectory"` field. Format: `w/s/a/d` keys with `*N` repeats; comma-separated segments. Example: `"a*4,w*8,s*7"`. The script reads each sample's trajectory automatically.

### 3.2 HY TI2V (4-step DMD)

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

### 3.3 Wan Action2V (4-step DMD, camera control)

```bash 
OUTPUT_FOLDER=./outputs/quickstart_wan_action2v \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```
---

## 4. Training

> Three model lines: HY Action2V, HY TI2V, Wan Action2V.
> Each splits into two phases: **Phase 1 Bidirectional SFT** (bidirectional multi-step base) and **Phase 2 Causal Forcing** (distillation to causal few-step).
> Phase 2 has 4 stages: Stage 1 Teacher Forcing AR Diffusion, Stage 2(a) Causal ODE Distillation Initialization, Stage 2(b) Causal Consistency Distillation, Stage 3 Asymmetric DMD with Self Rollout.
> Every subsection follows the same structure: **(1) Model download**, **(2) Data preparation**, **(3) Training script**, **(4) Validation**.

### 4.1 HY Action2V

#### 4.1.1 Phase 1: Bidirectional SFT

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**


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

**(2) Data preparation**

Pick one. Everything lands under `./dataset/`.

**Option A: download minWM-dataset**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "preencode_input.json" "videos/**"
    
```

Resulting layout:

```
./dataset/
├── preencode_input.json
└── videos/
    ├── 000000_right8a11/gen.mp4
    ├── 000001_w10d9/gen.mp4
    └── ...
```

**Option B: bring your own videos and trajectories**

Match Option A's layout: provide your own `preencode_input.json` plus `videos/`. `preencode_input.json` is a list; each entry must contain at least `image_path` / `caption` / `pose_str`:

```json
[
    {
        "image_path": "/abs/path/to/image1.png",
        "caption": "A scenic mountain view",
        "pose_str": "right-8, a-11"
    }
]
```

Video directory naming rule: `{i:06d}_{slug(pose_str)}/gen.mp4`, where `i` is the JSON index and `slug` is `pose_str` lowercased with non-alphanumeric characters stripped.

**Final: encoding (shared by both options)**

Script defaults: `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`, `INPUT_DIR=./dataset`, `OUTPUT_DIR=./dataset/HY15/Action2V`:

```bash
bash HY15/scripts/data_preprocessing/run_preencode_downloaded_camera_video.sh
```

CFG negative prompt embeddings are downloaded separately:

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"
```

Final layout:

```
./dataset/
├── HY15/Action2V/                       # encoded outputs
│   ├── latents/
│   └── train_index.json
└── others/HY/Action2V/                  # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) Training script**

Bidirectional + camera (ProPE) training. Model class `HunyuanTransformer3DARActionProPEModel`, camera dataloader carrying viewmats/Ks, flow matching MSE loss:

```bash
bash HY15/scripts/training/hyvideo15/run_bi_camera_multinode.sh
```

By default the script writes ckpts to `./ckpts/HY15/Action2V/bidirectional/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents (`diffusion_pytorch_model.safetensors` + `config.json` etc.) to `./ckpts/HY15/Action2V/bidirectional/`, matching the predownload layout used in §2.3 / §4.1.2 Stage 1 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/Action2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/bidirectional/
```

**(4) Validation**

Reuses the §3.1 inference script in 50-step bidirectional mode. The script defaults to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional` (this stage's output, also the §4.1.2 Stage 1 predownload path); override via env to point at another ckpt:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_camera \
    bash HY15/scripts/inference/run_infer_bidirectional_camera.sh
```

</details>

#### 4.1.2 Phase 2: Causal Forcing

##### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/bidirectional/**"
```


**(2) Data preparation**

Same as §4.1.1 (2). Reuses the same `./dataset/HY15/Action2V/{latents, train_index.json}` and `./dataset/others/HY/Action2V/`.

**(3) Training script**

Convert the bidirectional model from Phase 1 into causal + teacher-forcing AR. Same ProPE model class and camera dataloader; loss is still flow matching MSE:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/ar_diffusion_tf/`, matching the predownload layout used in §4.1.2 Stage 2(a) (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/ar_diffusion_tf/
```

**(4) Validation**

50-step AR rollout mode. Script defaults `TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf` (this stage's output, also the Stage 2(a) predownload path); other variables likewise overridable via env:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_camera \
    bash HY15/scripts/inference/run_infer_ar_diffusion_camera.sh
```

</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Everything lands at `./dataset/HY15/Action2V_ode/`, parallel to and not overlapping the §4.1.1 SFT latents (`./dataset/HY15/Action2V/`). Negative prompts reuse the `./dataset/others/HY/Action2V/` already downloaded in §4.1.1.

**Option A: download minWM-dataset's pre-generated ODE latents**

```bash
# 1) Download from HF (the repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/Action2V/**"

# 2) Move to the unified layout ./dataset/HY15/Action2V_ode/
mkdir -p ./dataset/HY15/Action2V_ode
mv ./dataset/ODE_data/HY15/Action2V/latents ./dataset/HY15/Action2V_ode/latents


# 3) Regenerate the absolute-path index in the new location
#    (writes to ./dataset/HY15/Action2V_ode/train_index.json)
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §4.1.2 Stage 1 ckpt (at `./ckpts/HY15/Action2V/ar_diffusion_tf/`); run 48-step CFG sampling on the SFT-encoded data (§4.1.1):

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/Action2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/Action2V_ode/latents \
    bash HY15/scripts/training/hy15_camera/ode_sampling.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

Final layout:

```
./dataset/
├── HY15/
│   ├── Action2V/                        # SFT latents (from §4.1.1)
│   │   ├── latents/
│   │   └── train_index.json
│   └── Action2V_ode/                    # ODE latents (this section)
│       ├── latents/
│       └── train_index.json
└── others/HY/Action2V/                  # neg prompts (reused from §4.1.1)
```

**(3) Training script**

ODE regression: on the ODE latents prepared in §4.1.2 Stage 2(a) (2) (`./dataset/HY15/Action2V_ode/`), have the model directly regress the ODE solver outputs at key timesteps [0, 12, 24, 36, -2, -1] (corresponding to [1000, 750, 500, 250] etc.):

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_ode.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/causal_ode/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/causal_ode/`, matching the predownload layout used in §4.1.2 Stage 3 (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_ode/
```

**(4) Validation**

The 4-step DMD inference script (`run_infer_causal_camera.sh`) is shared by Stage 2(a) / 2(b) / 3, defaulting to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd` (the final output, matching the §3.1 quickstart). For this stage, override via env to causal_ode:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```


</details>

##### Stage 2(b): Causal Consistency Distillation

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as Stage 2(a).

**(2) Data preparation**

Same as §4.1.1 (2). Reuses the same encoded data.

**(3) Training script**

Consistency distillation: a frozen teacher and an EMA target network force the student to output a consistent prediction across adjacent timestep pairs (t, t_next). `trainer: consistency_distillation`:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_cd.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/causal_cd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/causal_cd/`, matching the predownload layout used in §4.1.2 Stage 3 (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_cd/
```

**(4) Validation**

Same as §4.1.2 Stage 2(a) (4); env-switch to this stage's output causal_cd:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

##### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/causal_ode/**" # Or causal_cd
```

**(2) Data preparation**

Same as §4.1.1 (2). Reuses the SFT latents encoded in §4.1.1 (`./dataset/HY15/Action2V/{latents, train_index.json}` and `./dataset/others/HY/Action2V/`). Stage 3 runs self rollout: DMD training only consumes the conditioning portion of `train_index.json` (first-frame image + caption + pose); it neither supervises against real video latents nor consumes the §4.1.2 Stage 2(a) ODE latents.

**(3) Training script**

Asymmetric DMD with self rollout: score distillation, conditioning only, no real-video supervision. The student aligns to the teacher's score field on its own rollouts:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_dmd.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/dmd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/dmd/`, matching the inference path used by §3.1 quickstart / §4.1.2 Stage 3 (4):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/dmd/
```

**(4) Validation**

Run the §3.1 quickstart directly (the script defaults to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd`, this stage's output):

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

### 4.2 HY TI2V

#### 4.2.1 Phase 1: Bidirectional SFT

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as 4.1.1.

**(2) Data preparation**

Pick one. Everything lands under `./dataset/`.

**Option A: download minWM-dataset's pre-encoded latents**

> The HF remote directory is named `ODE_data/HY15/TI2V/` (toy data, ~5K, shared with the Causal ODE stage and named accordingly in the repo); after downloading, mv it to the local SFT layout `./dataset/HY15/TI2V/`.

```bash
# 1) Download from HF (the repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/TI2V/**"

# 2) Move to the unified layout ./dataset/HY15/TI2V/
mkdir -p ./dataset/HY15/TI2V
mv ./dataset/ODE_data/HY15/TI2V/latents ./dataset/HY15/TI2V/latents

# 3) Regenerate the absolute-path index in the new location
#    (writes to ./dataset/HY15/TI2V/train_index.json)
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V/latents \
    -o ./dataset/HY15/TI2V/train_index.json
```

**Option B: bring your own videos + text (encode locally from raw video)**

Input `./dataset/videos.json` (replace paths with your own):

```json
[
    {"video_path": "/abs/path/to/video1.mp4", "caption": "A cat playing"},
    {"video_path": "/abs/path/to/video2.mp4", "caption": "A sunset"}
]
```

Script defaults: `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`, `INPUT_JSON=./dataset/videos.json`, `OUTPUT_DIR=./dataset/HY15/TI2V`:

```bash
bash HY15/scripts/data_preprocessing/run_preencode_video.sh
```

**Final: CFG negative prompt embeddings (shared by both options)**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/TI2V/**"
```

Final layout (unified at `./dataset/HY15/TI2V/`, identical for Option A/B):

```
./dataset/
├── HY15/TI2V/                           # encoded latents (A: moved from HF download; B: locally encoded)
│   ├── latents/
│   └── train_index.json
└── others/HY/TI2V/                      # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) Training script**

Bidirectional TI2V SFT. Model class `HunyuanTransformer3DARActionModel`, global (non-causal) attention, flow matching MSE loss:

```bash
bash HY15/scripts/training/hyvideo15/run_bi_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/bidirectional/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents (`diffusion_pytorch_model.safetensors` + `config.json` etc.) to `./ckpts/HY15/TI2V/bidirectional/`, matching the predownload layout used in §4.2.2 Stage 1 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/bidirectional/
```

**(4) Validation**

Reuses the §3.2 inference script in 50-step bidirectional mode. The script defaults to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional` (this stage's output, also the §4.2.2 Stage 1 predownload path); override via env to point at another ckpt:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_ti2v \
    bash HY15/scripts/inference/run_infer_bidirectional.sh
```

</details>

#### 4.2.2 Phase 2: Causal Forcing

##### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt (TI2V can also be initialized directly from the official bidirectional model; we still provide a fine-tuned version of ours for completeness):

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/bidirectional/**"
```

**(2) Data preparation**

Same as §4.2.1 (2). Reuses the same `./dataset/HY15/TI2V/{latents, train_index.json}` and `./dataset/others/HY/TI2V/`.

**(3) Training script**

Convert the bidirectional TI2V model from Phase 1 into causal + teacher-forcing AR. Loss remains flow matching MSE:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/ar_diffusion_tf/`, matching the predownload layout used in §4.2.2 Stage 2(a) (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/ar_diffusion_tf/
```

**(4) Validation**

50-step AR rollout mode. Script defaults `TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf` (this stage's output, also the Stage 2(a) predownload path); other variables likewise overridable via env:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_ti2v \
    bash HY15/scripts/inference/run_infer_ar_diffusion.sh
```

</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Option A reuses the toy ODE data already downloaded in §4.2.1 (2); Option B samples ODE trajectories from the Stage 1 ckpt yourself (lands in a separate directory `./dataset/HY15/TI2V_ode/` to avoid mixing with §4.2.1). Negative prompts reuse the `./dataset/others/HY/TI2V/` already downloaded in §4.2.1.

**Option A: reuse the ODE data already downloaded in §4.2.1**

The `ODE_data/HY15/TI2V/` downloaded under §4.2.1 Option A (the HF repo names it that way; it is the toy ODE data shared with this stage) was moved to `./dataset/HY15/TI2V/{latents, train_index.json}`. This stage reuses it directly; no extra action needed.

> Difference vs. §4.1.2 Stage 2(a) (HY Action2V): in Action2V the SFT latents and the ODE latents are two independent datasets (in `./dataset/HY15/Action2V/` and `./dataset/HY15/Action2V_ode/` respectively); in TI2V the toy dataset is itself ODE data — §4.2.1 Option A already uses it as the SFT latents, so this stage simply shares that single copy.

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §4.2.2 Stage 1 ckpt (at `./ckpts/HY15/TI2V/ar_diffusion_tf/`); run 48-step CFG sampling on the SFT-encoded data (§4.2.1):

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/TI2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/TI2V_ode/latents \
    bash HY15/scripts/ode_sampling/ode.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V_ode/latents \
    -o ./dataset/HY15/TI2V_ode/train_index.json
```

Final layout (Option B drop point):

```
./dataset/
├── HY15/
│   ├── TI2V/                            # SFT latents (from §4.2.1)
│   │   ├── latents/
│   │   └── train_index.json
│   └── TI2V_ode/                        # ODE latents (Option B output here)
│       ├── latents/
│       └── train_index.json
└── others/HY/TI2V/                      # neg prompts (reused from §4.2.1)
```

**(3) Training script**

ODE regression: on the ODE latents prepared in §4.2.2 Stage 2(a) (2), have the model directly regress the ODE solver outputs at key timesteps:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_ode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/causal_ode/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/causal_ode/`, matching the predownload layout used in §4.2.2 Stage 3 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_ode/
```

**(4) Validation**

The 4-step DMD inference script (`run_infer_causal.sh`) is shared by Stage 2(a) / 2(b) / 3, defaulting to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd` (the final output, matching the §3.2 quickstart). For this stage, override via env to causal_ode:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

##### Stage 2(b): Causal Consistency Distillation

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as Stage 2(a).

**(2) Data preparation**

Same as §4.2.1 (2). Reuses the same encoded data.

**(3) Training script**

Consistency distillation. `trainer: consistency_distillation`; with a frozen teacher and an EMA target network, the student is forced to output a consistent prediction across adjacent timestep pairs (t, t_next):

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_cd.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/causal_cd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/causal_cd/`, matching the predownload layout used in §4.2.2 Stage 3 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_cd/
```

**(4) Validation**

Same as §4.2.2 Stage 2(a) (4); env-switch to this stage's output causal_cd:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

##### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/causal_ode/**" # or causal_cd
```

**(2) Data preparation**

Same as §4.2.1 (2). Reuses the SFT latents encoded in §4.2.1: `./dataset/HY15/TI2V/{latents, train_index.json}` and `./dataset/others/HY/TI2V/`. Stage 3 runs self rollout: DMD training only consumes the conditioning portion of `train_index.json` (first-frame image + caption); it does not supervise against real video latents.

**(3) Training script**

Asymmetric DMD with self rollout. Score distillation, consuming only the conditioning portion of `train_index.json` (first-frame image + caption); no real-video supervision:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_dmd.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/dmd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/dmd/`, matching the inference path used by the §3.2 quickstart:

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/dmd/
```

**(4) Validation**

Run the §3.2 quickstart directly (the script defaults to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd`, this stage's output):

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

### 4.3 Wan Action2V

#### 4.3.1 Phase 1: Bidirectional SFT

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir ./ckpts/Wan2.1-T2V-1.3B \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" "diffusion_pytorch_model.safetensors" "config.json"
```

**(2) Data preparation**

Video source (Option A download / Option B bring your own) and disk layout are **the same as §4.1.1 (2)**, reusing the same `./dataset/preencode_input.json` + `./dataset/videos/`. The difference is only in encoding:

- HY encoding lands at `<OUTPUT_DIR>/latents/` (per-sample `.pt`),
- Wan encoding lands at `<OUTPUT_DIR>/data/` (a merged LMDB); the downstream training entry reads LMDB directly.

**Encoding (shared by both options)**

The VAE uses the Wan2.1 base model (already symlinked per §2.2 to `Wan21/wan_models/Wan2.1-T2V-1.3B/`). Script defaults:

```
VAE_PATH=Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
INPUT_JSON=./dataset/preencode_input.json
VIDEO_DIR=./dataset/videos
OUTPUT_DIR=./dataset/Wan21/Action2V
```

Run directly:

```bash
bash Wan21/scripts/data_preprocessing/run_build_worldplaygen_lmdb.sh
```

The merged LMDB lands at `./dataset/Wan21/Action2V/data/`.

> Wan's CFG negative prompt is written into the config (`Wan21/configs/causal_forcing_dmd_camera.yaml:38`), so no `.pt` preencoding is needed; hence no equivalent of HY's `others/HY/Action2V/` download step.

Final layout:

```
./dataset/
├── preencode_input.json                 # same as §4.1.1
├── videos/                              # same as §4.1.1
└── Wan21/Action2V/
    └── data/                            # encoded LMDB
```

**(3) Training script**

Bidirectional + camera (PRoPE) SFT. `trainer: bidirectional_diffusion`, using `WanDiffusionWrapper(use_camera=True)` and `CameraLatentLMDBDataset` (carrying viewmats/Ks); flow matching loss:

```bash
bash Wan21/scripts/training/run_stage0_bidirectional_camera.sh
```

By default ckpts land at `logs/bidirectional_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/bidirectional/`, matching the predownload layout used in §4.3.2 Stage 1 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/bidirectional
mv logs/bidirectional_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/bidirectional/model.pt
```

**(4) Validation**

50-step bidirectional + camera mode. Script defaults `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/bidirectional/model.pt` (this stage's output, also the §4.3.2 Stage 1 predownload path); override via env to point at another ckpt:

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
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/bidirectional/**"
```

**(2) Data preparation**

Same as §4.3.1 (2). Reuses the same `./dataset/Wan21/Action2V/data/` (LMDB).

**(3) Training script**

Convert the bidirectional model from Phase 1 into causal + teacher-forcing AR. Switch the model to `CausalWanModel`, keep `use_camera: true` to retain PRoPE projection parameters; flow matching loss:

```bash
bash Wan21/scripts/training/run_stage1_ar_camera.sh
```

By default ckpts land at `logs/ar_camera_tf/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/ar_diffusion_tf/`, matching the predownload layout used in §4.3.2 Stage 2(a) (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/ar_diffusion_tf
mv logs/ar_camera_tf/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

**(4) Validation**

50-step AR + camera mode. Script defaults `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt` (this stage's output, also the Stage 2(a) predownload path):

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
OUTPUT_FOLDER=./outputs/eval_ar_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_ar_camera.sh
```

</details>

##### Stage 2(a): Causal ODE Distillation Initialization

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**


If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Everything lands at `./dataset/Wan21/Action2V/ode_lmdb/`, parallel to and not overlapping the §4.3.1 SFT LMDB (`./dataset/Wan21/Action2V/data/`).

**Option A: download minWM-dataset's pre-generated ODE latents and merge LMDB locally**

What HF publishes is the **unmerged `.pt` latents** (the output of `get_causal_ode_data_prope.py`); after downloading, run a local merge:

```bash
# 1) Download the .pt latents (HF repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/Wan21/Action2V/**"

# 2) Merge into the training-ready LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/ODE_data/Wan21/Action2V \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §4.3.2 Stage 1 ckpt (at `./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`); run 48-step CFG sampling on the §4.3.1-encoded SFT LMDB (`./dataset/Wan21/Action2V/data/`), then merge into LMDB:

```bash
# 1) Run 48-step CFG sampling with the Stage 1 ckpt to get .pt latents
torchrun --nproc_per_node=8 Wan21/get_causal_ode_data_prope.py \
    --generator_ckpt ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
    --rawdata_path ./dataset/Wan21/Action2V/data \
    --output_folder ./dataset/Wan21/Action2V/ode_latents

# 2) Merge into the training-ready LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/Wan21/Action2V/ode_latents \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

> Difference vs. §4.1.2 Stage 2(a) (HY Action2V): HY's ODE outputs are per-sample `.pt`, so after HF download only `mv` + regenerating `train_index.json` is needed; for Wan, since the training entry consumes LMDB directly, **both options additionally require running `build_ode_prope_lmdb.py` to merge**.

Final layout:

```
./dataset/
├── preencode_input.json                # same as §4.3.1
├── videos/                             # same as §4.3.1
└── Wan21/Action2V/
    ├── data/                           # SFT LMDB (from §4.3.1)
    └── ode_lmdb/                       # ODE LMDB (output of this section, read directly by the training entry)
```

**(3) Training script**

ODE regression. `trainer: ode`; trains on the ODE LMDB prepared in §4.3.2 Stage 2(a) (2) (`./dataset/Wan21/Action2V/ode_lmdb/`), read by `CameraODERegressionLMDBDataset` and regressing the ODE solver outputs at key timesteps [1000, 750, 500, 250]:

```bash
bash Wan21/scripts/training/run_stage2_causal_ode_camera.sh
```

By default ckpts land at `logs/causal_ode_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/causal_ode/`, matching the predownload layout used in §4.3.2 Stage 3 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/causal_ode
mv logs/causal_ode_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_ode/model.pt
```

**(4) Validation**

`run_infer_causal_camera.sh` defaults to the 4-step DMD config + dmd ckpt (matching the §3.3 quickstart). For this stage, env-switch both `CONFIG_PATH` and `CHECKPOINT_PATH` to the ODE version:

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
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as Stage 2.

**(2) Data preparation**

Same as §4.3.1 (2). Reuses the same encoded data.

**(3) Training script**

Consistency distillation. `trainer: consistency_distillation`; with a frozen teacher and an EMA target network, the student is forced to output a consistent prediction across adjacent timestep pairs (t, t_next):

```bash
bash Wan21/scripts/training/run_stage2_causal_cd_camera.sh
```

By default ckpts land at `logs/causal_cd_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/causal_cd/`, matching the predownload layout used in §4.3.2 Stage 3 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/causal_cd
mv logs/causal_cd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_cd/model.pt
```

**(4) Validation**

Same as §4.3.2 Stage 2(a) (4); env-switch to the CD config + CD ckpt:

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
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/causal_ode/**"  # Or causal_cd
```

**(2) Data preparation**

Same as §4.3.1 (2). Reuses the SFT LMDB encoded in §4.3.1 (`./dataset/Wan21/Action2V/data/`). Stage 3 runs self rollout and no longer consumes the Stage 2(a) ODE LMDB.

**(3) Training script**

Asymmetric DMD with self rollout. `trainer: score_distillation`; conditioning only, no real-video supervision. The student aligns to the teacher's score field on its own rollouts:

```bash
bash Wan21/scripts/training/run_stage3_causal_dmd_camera.sh # 100~200 steps recommended
```

By default ckpts land at `logs/causal_dmd_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/dmd/`, matching the inference path used by §3.3 quickstart / §4.3.2 Stage 3 (4):

```bash
BEST_STEP=000x00  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/dmd
mv logs/causal_dmd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/dmd/model.pt
```

**(4) Validation**

Run the §3.3 quickstart directly (the script defaults to `CONFIG_PATH=causal_forcing_dmd_camera.yaml` and `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt`, this stage's output):


```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=./outputs/eval_dmd_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>


### 4.4 Wan T2V

Not yet integrated in this repo. See the [Causal-Forcing repo](https://github.com/thu-ml/Causal-Forcing) for training, inference, data, and models.
