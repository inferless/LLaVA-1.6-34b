import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image

import requests
from io import BytesIO

import time
import subprocess
from threading import Thread
import os



class InferlessPythonModel:
    def initialize(self):
    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model("liuhaotian/llava-v1.6-34b", model_name="llava-v1.6-34b", model_base=None, load_8bit=False, load_4bit=False)

    def infer(self, inputs):
        """Run a single prediction on the model"""

        image = inputs["image"]
        prompt: str = inputs["prompt"] 
        top_p = 1.0
        temperature = 0.2
        max_tokens = 1024
        
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        with torch.inference_mode():
        # Directly generate output without using a separate thread and streaming.
            output = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return {"output": decoded_output }


def finalize(self):
    self.model = None
    self.image_processor = None
    self.context_len = None
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

