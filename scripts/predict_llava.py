import json
import re
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig
import requests
from flask import Flask, render_template, request, jsonify
from instruction_generation import *


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = 'cuda'

llava_7b_path = "llava-hf/llava-1.5-7b-hf"
# llava_7b_path = "llava-hf/llava-1.5-13b-hf"
print(llava_7b_path)
llava_model = LlavaForConditionalGeneration.from_pretrained(llava_7b_path, torch_dtype=torch.float16, device_map="auto")
llava_processor = AutoProcessor.from_pretrained(llava_7b_path)
llava_model.eval()

def llava_inference(instruction, image_path):
    prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
    # image_path = "data/images/sample_redlight.png"
    image = Image.open(image_path)

    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)
    # Generate
    generate_ids = llava_model.generate(
        **inputs, 
        max_new_tokens=512,
        # do_sample=True,
        # temperature=1,
        )
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


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

    
    data = json.load(open(read_path))
    results = []

    for sample in data:
        # instructions = sample["instructions"] 

        # mcq
        instructions = formulate_instruction(sample, task)

        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[sample]: ", instruction)
            pred = llava_inference(instruction, image_path)
            if "[INST]" in pred and "[/INST]" in pred:
                pred = pred.split("[/INST]")[1].strip()
            pattern1 = r'USER:(.*?)\nASSISTANT:' 
            pattern2 = r'USER:(.*?)ASSISTANT:'
            pred = re.sub(pattern1, '', pred).strip()
            pred = re.sub(pattern2, '', pred).strip()
            if "ASSISTANT: " in pred:
                pred = pred.split("ASSISTANT: ")[1].strip()
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", pred)
        sample["result"] = cur_preds
        results.append(sample)
    
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":

    main()
    
