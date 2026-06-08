
import torch

import json
from lavin import ModelArgs, Tokenizer, Transformer
from lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter

from pathlib import Path
from util.apply_delta import apply_model_delta_online



def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params


    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)


    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict


    return checkpoint, tokenizer, params


def load_and_change_llava_statedict(llama_model_path, model_name):
    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))

    llava_path = '/data/yiping/LaVIN-video-all/LLaVA-7B-Lightening-v1-1'
    resolved_archive_file = [llava_path + '/pytorch_model-00001-of-00002.bin',
                             llava_path + '/pytorch_model-00002-of-00002.bin']
    state_dict_all = {}
    for shard_file in resolved_archive_file:
        state_dict = torch.load(shard_file)
        # state_dict_all += state_dict
        state_dict_all.update(state_dict)
        del state_dict

    state_dict_new = {}
    for k in state_dict_all.keys():
        if 'model.' in k:
            new_key = k.replace('model.', '')
            if 'embed_tokens' in new_key:
                new_key = new_key.replace('embed_tokens', 'tok_embeddings')
            if 'self_attn' in new_key:
                new_key = new_key.replace('self_attn', 'attention')
                new_key = new_key.replace('q_proj', 'wq').replace('k_proj', 'wk').replace('v_proj', 'wv').replace('o_proj', 'wo')
            if 'mlp' in new_key:
                new_key = new_key.replace('mlp', 'feed_forward')
                new_key = new_key.replace('gate_proj', 'w1').replace('up_proj', 'w3').replace('down_proj', 'w2')
            if 'input_layernorm' in new_key:
                new_key = new_key.replace('input_layernorm', 'attention_norm')
            if 'post_attention_layernorm' in new_key:
                new_key = new_key.replace('post_attention_layernorm', 'ffn_norm')
            if 'mm_projector' in new_key:
                new_key = new_key.replace('mm_projector', 'mm_adapter_proj')

        if 'lm_head' in k:
            new_key = k.replace('lm_head', 'output')

        state_dict_new[new_key] = state_dict_all[k]

    return state_dict_new, tokenizer, params



def LaVIN(args):

    llama_model_path =args.llama_model_path
    model_name = args.llm_model

    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)
    # state_dict_llava, tokenizer, params = load_and_change_llava_statedict(llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=32,hidden_proj=args.hidden_proj,drop_path=args.drop_path, **params
    )

    model_args.vocab_size = tokenizer.n_words

    if args.cpu_load:
        #cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    print('define llama')
    llama = Transformer(model_args)

    # delete language encoder
    # del llama.backbone.transformer

    print('set default tensor type')
    torch.set_default_tensor_type(torch.FloatTensor)

    print('quant_model_bnb')
    if args.bits in ['4bit', '8bit']:
        from util.quantization import quant_model_bnb
        llama.layers=quant_model_bnb(llama.layers,quant_bit=args.bits)

    print('load_state_dict')
    llama.load_state_dict(checkpoint, strict=False)

    ################### 改成load llava的state dict
    # llama.load_state_dict(state_dict_llava, strict=False)

    if args.use_vicuna:
        apply_model_delta_online(llama,'../data/weights/vicuna_'+args.llm_model)


    print('define adapter', str(args.adapter_type), str(args.visual_adapter_type))
    if args.adapter_type == 'block' or args.adapter_type == 'attn':
        set_MMAdapter(llama,args.adapter_type,dim=args.adapter_dim,s=args.adapter_scale,t=args.temperature,gradient_checkpointing=args.gradient_checkpointing)
        # set_Clip_Adapter(llama.backbone.visual,args.visual_adapter_type,dim=args.adapter_dim,s=args.adapter_scale,t=args.temperature)


    learnable_keys=['adapter']
    total=0.
    total_params = 0.
    trainable_names=[]
    for name, param in llama.named_parameters():
        total_params += param.numel()
        for key in learnable_keys:

            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total += param.nelement()
                trainable_names.append((name, param.shape))
            else:
                param.requires_grad = False
    print(trainable_names)
    print('  + Number of trainable params: %.2fM' % (total / 1e6))
    print(f'*******************************Total params: {total_params / 1e6:.2f}M')
    return llama

