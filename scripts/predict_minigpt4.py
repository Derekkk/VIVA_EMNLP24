"""
Refer to https://github.com/Vision-CAIR/MiniGPT-4 to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO


from minigpt4_utils.common.config import Config
from minigpt4_utils.common.dist_utils import get_rank
from minigpt4_utils.common.registry import registry
from minigpt4_utils.conversation.conversation import Chat, CONV_VISION

from minigpt4_utils.datasets.builders import *
from minigpt4_utils.models import *
from minigpt4_utils.processors import *
from minigpt4_utils.runners import *
from minigpt4_utils.tasks import *
import json
from instruction_generation import *


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class VisITBaseModel(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, instruction, images):
        return self.generate(instruction, images)
    
    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError

class VisITMiniGPT4(VisITBaseModel):
    def __init__(self,):
        super().__init__()

        cfg = Config(cfg_path='./minigpt4_utils/minigpt4_eval.yaml')
        # minigpt4_llama2_eval.yaml, minigpt4_eval.yaml
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, self.vis_processor, device=device)

    def generate(self, instruction, image_path):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        # if len(images) == 0:
        #     raise ValueError('No image is provided.')
        # if len(images) > 1:
        #     return '[Skipped]: Currently only support single image.'
        
        # Init chat state
        chat_state = CONV_VISION.copy()
        img_list = []

        # download image image
        # response = requests.get(images[0])
        img = Image.open(image_path).convert("RGB")
        
        # upload Image
        self.chat.upload_img(img, chat_state, img_list)

        # ask
        self.chat.ask(instruction, chat_state)

        # answer
        out = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000
        )[0]

        return out


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

    model = VisITMiniGPT4()


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
            
            pred = model.generate(
                instruction=instruction,
                image_path=image_path,
            )

            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", pred)
        sample["result"] = cur_preds
        results.append(sample)
    
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()

