
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port 21518 train.py \
    --llm_model 7B \
    --llama_model_path ./data/weights/ \
    --data_root ./data/ \
    --video_folder ./data/video_features/ \
    --max_seq_len 128 \
    --batch_size 32 \
    --accum_iter 1 \
    --epochs 5 \
    --warmup_epochs 0 \
    --lr_qformer 1e-4 \
    --lr_lora 1e-3 \
    --weight_decay 0.02 \
    --output_dir ./MOLAE/llama7/eff/ \
    --log_dir ./output_dir/llama7/eff/ \
    --adapter_type attn\
    --adapter_dim 8 \
    --adapter_scale 1 \
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type no_clip_adpt \
    --bits 4bit \
    --seed 42 \
    --cms_type eff
#    --gradient_checkpointing \
#    --cpu_load \
