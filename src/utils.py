"""utils"""
import os
import re
import gc

import torch
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


class MultimodalSystem:
    """MultimodalSystem"""
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str) -> None:
        """__init__"""
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model_name_or_path = None
        self.dataset_name_or_path = None
        self.load(model_name_or_path, dataset_name_or_path)

    def load(self, model_name_or_path: str, dataset_name_or_path: str) -> dict:
        """load"""
        if model_name_or_path != self.model_name_or_path:
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()
            self.model_name_or_path = model_name_or_path
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quantization_config,
            )

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path
            self.dataset = load_dataset(self.dataset_name_or_path)

        return {
            "dataset": self.dataset_name_or_path,
            "model": self.model_name_or_path,
        }

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

    def evaluate(self) -> float:
        """evaluate"""
        accuracy = 0

        if self.dataset_name_or_path == "m-a-p/CII-Bench":
            dataset = self.dataset["dev"]
            correct_count = 0
            option_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

            for example in tqdm(dataset):
                options = "\n".join([
                    f"({option_labels[i]}){example[f'option{i + 1}']}"
                    for i in range(6) if example.get(f"option{i + 1}")
                ])
                full_question = f"Question: {example['question']}\nOptions:\n{options}\nAnswer:"

                prediction = self.generate(full_question, example["image"])
                match = re.search(r'\b[a-hA-H]\b', prediction)

                if match:
                    predicted_answer = match.group(0).upper()
                    if predicted_answer == example["answer"]:
                        correct_count += 1

            accuracy = correct_count / len(dataset)

        gc.collect()
        torch.cuda.empty_cache()

        return accuracy
