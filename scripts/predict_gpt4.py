import json
import os
from urllib import response
import openai
import requests
import re
# from openai import OpenAI
import openai
import time
from nltk.tokenize import word_tokenize
import base64
from mimetypes import guess_type
import argparse
from instruction_generation import *
   

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# gpt-4-turbo-2024-04-09
# gpt-4o-2024-05-13
# gpt-4-vision-preview
# gpt-4-1106-vision-preview
def gpt4_vision_generation(input_prompt, image_path, model="gpt-4-1106-vision-preview", temperature=1):
    

    API_KEY = "" # your key
    print(model)

    data_url = local_image_to_data_url(image_path)

    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        { 
                            "type": "text", 
                            "text": input_prompt 
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url":data_url
                            }
                        }
                        ]}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                api_key=API_KEY,
            )
            if response is not None:
                break
        except Exception as e:
            print(["[OPENAI ERROR]: ", [e]])
            response = None
            time.sleep(5)
    if response != None:
    # print(response)
        response = response.choices[0].message.content
    return response


def generate_captions():
    all_files = os.listdir("data/images_empathy")
    data = {}
    for fil in all_files:
        cur_path = "data/images_empathy/" + fil
        prompt = "Generate a brief caption of the image. You do not need to include too many details, but focus on the situation description:"
        caption = gpt4_vision_generation(prompt, cur_path)
        data[fil] = caption
    
    with open("images_empathy_captions.json", "w") as f_w:
        json.dump(data, f_w, indent=2)


def main():
    parser = argparse.ArgumentParser()
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


        # mcq
        instructions = formulate_instruction(sample, task)
    
        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[input: ]", [instruction, image_path])
            pred = gpt4_vision_generation(instruction, image_path)
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", [pred])
        sample["result"] = cur_preds
        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

    