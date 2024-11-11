import json
import re
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from instruction_generation import *



device = 'cuda'


model_id = "Salesforce/instructblip-vicuna-13b"
# model_id = "Salesforce/instructblip-vicuna-7b"
print(model_id)
insturct_blip_model = InstructBlipForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

insturct_blip_model = insturct_blip_model.half()
insturct_blip_processor = InstructBlipProcessor.from_pretrained(model_id)
insturct_blip_model.eval()


def instructblip_inference(instruction, image_path):
    image = Image.open(image_path)
    print(instruction, image_path)
    inputs = insturct_blip_processor(images=image, text=instruction, return_tensors="pt").to(device)
    
    outputs = insturct_blip_model.generate(
        **inputs,
        do_sample=False,
        temperature=1,
        num_beams=1,
        max_length=512,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
    )
    generated_text = insturct_blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(insturct_blip_processor.batch_decode(outputs, skip_special_tokens=False)[0].strip())
    print(generated_text)
    
    return generated_text


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
        print("[sample]: ", sample)
        # instructions = sample["instructions"] 
        
        # mcq
        instructions = formulate_instruction(sample, task)


        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[sample]: ", instruction)
            pred = instructblip_inference(instruction, image_path)
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
    