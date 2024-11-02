"""utils"""
import os
import gc

import torch
from PIL import Image
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText


class MultimodalSystem:
    """MultimodalSystem"""
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str) -> None:
        """__init__"""
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model = None
        self.processor = None
        self.model_name_or_path = None
        self.dataset_name_or_path = None
        self.load(model_name_or_path, dataset_name_or_path)

    def load(self, model_name_or_path: str, dataset_name_or_path: str) -> None:
        """load"""
        if model_name_or_path != self.model_name_or_path:
            del self.model, self.processor
            torch.cuda.empty_cache()
            gc.collect()

            self.model_name_or_path = model_name_or_path
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path
            self.dataset = load_dataset(self.dataset_name_or_path)

        return str(self.dataset), str(self.model)

    def generate(self, text: str, image: Image) -> None:
        """generate"""
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]}
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)[0]
        outputs = generated_ids[len(inputs.input_ids[0]) :]
        return  self.processor.decode(outputs, skip_special_tokens=True)
