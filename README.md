# LaVIN-Video: Video Causal Reasoning via Multimodal Adapters

## 环境配置

```bash
conda create -n VLM-MOLAE python=3.10 -y
conda activate VLM-MOLAE

# PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install bitsandbytes==0.43.0 fairscale==0.4.13 transformers==4.36.0 sentencepiece
pip install fire einops timm tensorboard ftfy opencv-python-headless decord omegaconf iopath
pip install ninja packaging psutil

# 评估指标
pip install pycocoevalcap
```

> **注意**: numpy 必须 < 2.0，否则 PyTorch 2.1 会报错。

## 数据准备

需要下载以下三项：

| 参数 | 说明 | 下载地址 |
|------|------|----------|
| `--llama_model_path` / `--ckpt_dir` | LLaMA base 模型权重 | https://huggingface.co/nyanko7/LLaMA-7B/tree/main |
| `--data_root` | 数据集 JSON 文件（含 `vlm_molae_json/` 目录下的 `eff_training.json`、`int_training.json`、`att_training.json`、`all_test.json`、`all_test_cap.json`） | <!-- TODO: 填写下载地址 --> |
| `--video_folder` | CLIP 提取的预处理好 video 特征（pickle 文件） | <!-- TODO: 填写下载地址 --> |

下载后目录结构示例：

```
data/
├── vlm_molae_json/
│   ├── eff_training.json
│   ├── int_training.json
│   ├── att_training.json
│   ├── all_test.json
│   └── all_test_cap.json
└── captions.json

video_features/
├── video7010.pkl
├── video7011.pkl
└── ...

weights/
├── tokenizer.model
├── consolidated.00.pth
├── params.json
└── ...
```

## 微调

使用 `scripts/finetune.sh` 进行微调：

```bash
bash scripts/finetune.sh
```

该脚本包含训练阶段，关键参数说明：

| 参数 | 说明 |
|------|------|
| `--cms_type` | CMS 类型，可选 `eff`（effect）、`int`（intention）、`att`（attribute） |
| `--llm_model` | 模型规模，如 `7B`、`13B` |
| `--adapter_type` | Adapter 类型，如 `attn` |
| `--bits` | 量化方式，如 `4bit`、`8bit`、`16bits` |
| `--lr_qformer` | Q-Former 学习率 |
| `--lr_lora` | LoRA 学习率 |
| `--n_prompt` | 视觉特征 token 数量 |

## 测试

使用 `scripts/test.sh` 进行测试，包含两个评估脚本：

```bash
bash scripts/test.sh
```

1. **`eval.py`** — 评估视频描述 + CMS 预测（生成 caption + effect/intention/attribute），计算 CIDEr、METEOR、ROUGE、BLEU 指标
2. **`eval_cap2cms.py`** — 仅评估 CMS 预测（给定 caption，预测 effect/intention/attribute），计算相同指标

两个脚本均通过 `--cms_type` 参数指定评估的 CMS 类型：

```bash
# 评估 effect
torchrun ... eval.py ... --cms_type eff
torchrun ... eval_cap2cms.py ... --cms_type eff

# 评估 intention
torchrun ... eval.py ... --cms_type int
torchrun ... eval_cap2cms.py ... --cms_type int

# 评估 attribute
torchrun ... eval.py ... --cms_type att
torchrun ... eval_cap2cms.py ... --cms_type att
```

结果保存在 adapter 权重同目录下，如 `preds-epoch4_eff.json`、`preds-epoch4_capeff_noperiod.json`。
