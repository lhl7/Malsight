import json
import os
from shlex import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Code Summarization Model

model_path = '/home/lhl/projects/models/hf/codet5p-770m'
# ft_model_path = '/home/lhl/projects/models/hf/BinT5-Demi'
ft1_model_path = '/home/lhl/projects/checkpoints/ft1_codet5p-770m/encoder_only/longer_label/checkpoint-625'
# ft_model_path = '/home/lhl/projects/checkpoints/ft1_codet5p-770m/encoder_only/6w/checkpoint-625'
ft0_model_path = '/home/lhl/projects/checkpoints/ft0_codet5p-770m/checkpoint-1666'
ph2_only = '/home/lhl/projects/checkpoints/ph2_only_codet5p-770m/checkpoint-625'
test = '/home/lhl/projects/checkpoints/diff_ratio/1:2/p2/checkpoint-972'

csum_tokenizer = AutoTokenizer.from_pretrained(model_path)
csum_model = AutoModelForSeq2SeqLM.from_pretrained(ft1_model_path).cuda()

csum_model.config.decoder_start_token_id = csum_tokenizer.bos_token_id
csum_model.config.pad_token_id = csum_tokenizer.pad_token_id
csum_model.eval()

def summarize(code_tokens: list[str], annotation: list=[]):
    
    slices = []
    summary = []
    
    if len(code_tokens) > 320:
        r = np.arange(320, len(code_tokens), 320)
        slice_num = len(r) + 1
        slices = np.array_split(code_tokens, r)
    else:
        slices.append(code_tokens)
        slice_num = 1
        
    for cts in slices:
        code = ' '.join(cts)
        anno = ' '.join(annotation) if len(annotation) != 0 else 'None'
        
        inputs = csum_tokenizer(
            code,
            text_pair=anno,
            max_length=400, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.inference_mode():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = csum_model.generate(inputs['input_ids'], max_length=int(120/slice_num))
            summary.append(csum_tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return ' '.join(summary)
    # return summary