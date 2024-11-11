import json
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from transformers import BitsAndBytesConfig
import requests
from flask import Flask, render_template, request, jsonify
from instruction_generation import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = 'cuda'

model_name = "Qwen/Qwen-VL-Chat"
qwenvl_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
qwenvl_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, fp16=True).eval()
qwenvl_model.eval()
qwenvl_model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

print("model params: ", count_parameters(qwenvl_model))

def qwenvl_inference(instruction, image_path):
    query = qwenvl_tokenizer.from_list_format([
        {'image': image_path},
        {'text': instruction},
    ])
    # query = qwenvl_tokenizer.from_list_format([
    #     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    #     {'text': 'Generate the caption in English with grounding:'},
    # ])

    # inputs = qwenvl_tokenizer(query, return_tensors='pt')
    # inputs = inputs.to(qwenvl_model.device)
    # pred = qwenvl_model.generate(**inputs)
    # response = qwenvl_tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    response, history = qwenvl_model.chat(
        qwenvl_tokenizer, 
        query=query, 
        history=None,
        do_sample=True,
        temperature=1,
        )

    return response


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=False)
    parser.add_argument('--image_folder', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    image_folder = args.image_folder
    

    print("write path: ", write_path)
    data = json.load(open(read_path))
    results = []



    for sample in data:
        print("[sample]: ", sample)
        # instructions = sample["instructions"] 

        # mcq
        instructions = formulate_instruction(sample, task)

        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[input]: ", instruction)
            pred = qwenvl_inference(instruction, image_path)
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", pred)
        sample["result"] = cur_preds
        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
    