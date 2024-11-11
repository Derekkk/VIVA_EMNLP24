import json
import re
from PIL import Image
from instruction_generation import *
import argparse
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

device = 'cuda'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mplog_owl2_inference(instruction, image_path, tokenizer, model, image_processor):

    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(image_path).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + instruction
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return outputs


def main():
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


    model_path = 'MAGAer13/mplug-owl2-llama2-7b'

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
    model.half()

    model.eval()
    print("model params: ", count_parameters(model))


    data = json.load(open(read_path))
    results = []

  
    for sample in data:
        print("[sample]: ", sample)
        
        instructions = formulate_instruction(sample, task)
        
        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[input]: ", instruction)
            pred = mplog_owl2_inference(instruction, image_path, tokenizer, model, image_processor)
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", pred)
        sample["result"] = cur_preds
        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
   
    main()
    
