# Data

请将下载好的数据放置在本目录下，最终结构如下：

```
data/
├── weights/                    # LLaMA base 模型权重
│   ├── tokenizer.model
│   ├── consolidated.00.pth
│   └── params.json
├── vlm_molae_json/             # 数据集 JSON 文件
│   ├── eff_training.json
│   ├── int_training.json
│   ├── att_training.json
│   ├── all_test.json
│   └── all_test_cap.json
└── video_features/             # CLIP 提取的 video 特征（pickle 文件）
    ├── video7010.pkl
    ├── video7011.pkl
    └── ...
```

下载地址请参考项目根目录 README.md。
