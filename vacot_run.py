
import os
import argparse
import random
import torch
import json
import copy
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, set_seed, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from qwen3_vl.vacot_qwen_model import *
from qwen3_vl.vacot_qwen_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import numpy as np

IMG_FOLDER = './data/m3cot/data/images/'
EVAL_FILE = './data/m3cot/data/test.jsonl'
DATA_NAME = 'm3cot'

dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
dataset = [x for x in dataset if x['image'] is not None ]

model_path = './model/Qwen3-VL-8B-Instruct'
processor = Qwen3VLProcessor.from_pretrained(model_path)
model = Qwen3VLForVaCoT.from_pretrained(model_path, attn_implementation="eager").to(device='cuda', dtype=torch.float16)
        
def calculate_generated_text(messages, item_id="sample"):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                    )
    inputs = inputs.to(device='cuda', dtype=torch.float16)
    inputs['output_attentions'] = True
    out = model.generate(**inputs,  **generation_config)
    model.t_mas_history.clear()
    
    out = out[0][inputs['input_ids'].shape[1]: ]
    
    generated_text = processor.decode(out, skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    out_file_path = './results/qwen3-vl/qwen_vacot_zero.json'
    mcot_one_fh = open('./results/qwen3-vl/qwen_vacot_one.json', 'a')
    mcot_zero_fh = open('./results/qwen3-vl/qwen_vacot_zero.json', 'a')
    processed_ids = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_data = json.loads(line)
                        processed_ids.add(processed_data['id'])
                    except json.JSONDecodeError:
                        continue
    
    original_len = len(dataset)
    dataset = [data for data in dataset if data['id'] not in processed_ids]
    print(f"Found {len(processed_ids)} completed samples. Resuming: {len(dataset)} / {original_len} remaining.")
    
    for data in tqdm(dataset):
        mcot_input_str = zero_shot_prompt_template.format(data['question'])
        for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
            mcot_input_str += '{}. {}\n'.format(i, c)
        mcot_input_str += '''Let's think step by step.\n Limit your reasoning to maximum 4-5 short sentences\n'''
        vision_x = [os.path.join('./data/m3cot/data/images', TRAING_CASE_1['id']+'.png'),
                    os.path.join(IMG_FOLDER, data['id']+'.png' if DATA_NAME == 'm3cot' else data['image'])]
    

        zero_shot_messages_mcot = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": vision_x[-1],
                        "max_pixels": 50176,
                    },
                    {"type": "text", "text": mcot_input_str},
                ],
            }
        ]

        
        one_shot_messages_mcot = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": vision_x[0]},
                    {"type": "text", "text": example_user_text},
        
                    {"type": "text", "text": mcot_induct_1},
                    
                    {"type": "text", "text": mcot_induct_2},

                    {"type": "text", "text": mcot_induct_3},
        
                    {"type": "text", "text": mcot_induct_4},
        
                    {"type": "image", "image": vision_x[-1], },
                    {"type": "text", "text": "Now answer the next question. Use the same reasoning style, but base your answer only on the new image.\n\n" + mcot_input_str + "\nAfter your reasoning, the final line must be exactly:\nAnswer: (Option letter)\nFor example:\nAnswer: (B)"},
                ],
            }
        ]
        
        zero_shot_mcot_ans = calculate_generated_text(zero_shot_messages_mcot, item_id=data['id'])
        one_shot_mcot_ans = calculate_generated_text(one_shot_messages_mcot)

        zero_shot_output = copy.deepcopy(data)
        zero_shot_output['pred'] = zero_shot_mcot_ans
        
        
        one_shot_output = copy.deepcopy(data)
        one_shot_output['pred'] = one_shot_mcot_ans
        
        mcot_zero_fh.write(json.dumps(zero_shot_output) + '\n')
        mcot_one_fh.write(json.dumps(one_shot_output) + '\n')
