

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 21515 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data/ \
    --video_folder ./data/video_features/ \
    --adapter_path ./MOLAE/llama7/eff/checkpoint-4.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 32 \
    --max_seq_len 512 \
    --split test \
    --n_prompt 40 \
    --temperature 10.\
    --visual_adapter_type router\
    --bits 4bit \
    --cms_type eff


CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 21515 eval_cap2cms.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data/ \
    --video_folder ./data/video_features/ \
    --adapter_path ./MOLAE/llama7/eff/checkpoint-4.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 32 \
    --max_seq_len 512 \
    --split test \
    --n_prompt 40 \
    --temperature 10.\
    --visual_adapter_type router\
    --bits 4bit \
    --cms_type eff
