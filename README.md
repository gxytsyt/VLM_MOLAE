# LaVIN-Video: Video Causal Reasoning via Multimodal Adapters

## Environment Setup

```bash
conda create -n VLM-MOLAE python=3.10 -y
conda activate VLM-MOLAE

# PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install bitsandbytes==0.43.0 fairscale==0.4.13 transformers==4.36.0 sentencepiece
pip install fire einops timm tensorboard ftfy opencv-python-headless decord omegaconf iopath
pip install ninja packaging psutil

# evaluation
pip install pycocoevalcap
```

> **note**: numpy must be < 2.0, otherwise PyTorch 2.1 will raise errors.

## Data Preparation

The following three resources are required:

| Argument | Description | Download |
|------|------|----------|
| `--llama_model_path` / `--ckpt_dir` | LLaMA base model weights | https://huggingface.co/nyanko7/LLaMA-7B/tree/main |
| `--data_root` | Dataset JSON files (containing `vlm_molae_json/` directory with `eff_training.json`、`int_training.json`、`att_training.json`、`all_test.json`、`all_test_cap.json`） | https://drive.google.com/file/d/1xveTVORplEX87yIFBeRm0p9Ba7sz3Yjl/view?usp=drive_link |
| `--video_folder` | CLIP-extracted preprocessed video features (pickle files) | https://drive.google.com/file/d/1hU_pUYBPboJQDhDD-kWP8iilqnyUCmga/view?usp=drive_link |

Expected directory structure after downloading:

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

## Fine-tuning

Run fine-tuning using `scripts/finetune.sh`:

```bash
bash scripts/finetune.sh
```

Key arguments:

| Argument | Description |
|------|------|
| `--cms_type` | CMS type, `eff` (effect),`int` (intention) or `att` (attribute) |
| `--llm_model` | Model size, `7B` or `13B` |
| `--adapter_type` | Adapter type, e.g., `attn` |
| `--bits` | Quantization precision: `4bit`, `8bit`, or `16bits` |
| `--lr_qformer` | Q-Former learning rate |
| `--lr_lora` | low-rank expert learning rate |

## Evaluation

Run evaluation using `scripts/test.sh`, which invokes two evaluation scripts:

```bash
bash scripts/test.sh
```

1. **`eval.py`** — Evaluates video captioning + CMS prediction (generates caption + effect/intention/attribute) and computes CIDEr, METEOR, ROUGE, and BLEU metrics.
2. **`eval_cap2cms.py`** — Evaluates CMS prediction only (given a caption, predicts effect/intention/attribute) and computes the same metrics.

Both scripts accept  `--cms_type` to specify the target CMS dimension:

```bash
# Evaluate effect
torchrun ... eval.py ... --cms_type eff
torchrun ... eval_cap2cms.py ... --cms_type eff

# Evaluate intention
torchrun ... eval.py ... --cms_type int
torchrun ... eval_cap2cms.py ... --cms_type int

# Evaluate attribute
torchrun ... eval.py ... --cms_type att
torchrun ... eval_cap2cms.py ... --cms_type att
```

Results are saved to the same directory as the adapter weights, e.g., `preds-epoch4_eff.json`, `preds-epoch4_capeff_noperiod.json`.
