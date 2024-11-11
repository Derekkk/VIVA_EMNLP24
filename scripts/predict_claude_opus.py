
import json
import os
import anthropic

import time
#from nltk.tokenize import word_tokenize
import httpx
import base64
from mimetypes import guess_type
import argparse
import openai
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


def claude3_vision_generation(input_prompt, image_path, model="claude-3-opus-20240229", temperature=1):
    
    API_KEY = "" # your key

    data_url = local_image_to_data_url(image_path)
    image_media_type="image/jpeg"
    image_data = data_url.split(",")[1] 


    for _ in range(10):
        try:
            response = anthropic.Anthropic(api_key=API_KEY).messages.create(model=model,
                                            max_tokens=1024,
                                            temperature=temperature,
                                            messages=[
                                                {
                                                    "role": "user",
                                                    "content": [
                                                                {
                                                                    "type": "image",
                                                                    "source": {
                                                                        "type": "base64",
                                                                        "media_type": image_media_type,
                                                                        "data": image_data,
                                                                    },
                                                                },
                                                                {
                                                                    "type": "text",
                                                                    "text": input_prompt
                                                                }
                                                                ],
                                                }
                                            ],
                                            )

            
            if response is not None:
                break
        except Exception as e:
            #print(["[CLAUDE3 ERROR]: ", [e]])
            response = None
            time.sleep(5)
    if response != None:
        #print(response.content[0].text)
        response = response.content[0].text
    return response



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
        print("[sample]: ", sample)
        
        instructions = formulate_instruction(sample, task)
    
        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[input: ]", [instruction, image_path])
            pred = claude3_vision_generation(instruction, image_path)
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", [pred])
        sample["result"] = cur_preds
        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

   