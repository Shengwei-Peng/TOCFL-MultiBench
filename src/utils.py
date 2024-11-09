"""utils"""
import os
import re
import gc
import json

import torch
import pandas as pd
from PIL.Image import Image
from tqdm.auto import tqdm
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText


class MultimodalSystem:
    """MultimodalSystem"""
    def __init__(
        self, model_name_or_path: str,
        dataset_name_or_path: str,
    ) -> None:
        """__init__"""
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model_name_or_path = None
        self.dataset_name_or_path = None
        self.model_settings = {
            "llava-hf/llava-1.5-7b-hf": {"attn_implementation": "flash_attention_2"},
            "llava-hf/llava-v1.6-mistral-7b-hf": {"attn_implementation": "flash_attention_2"},
            "Qwen/Qwen2-VL-7B-Instruct": {"attn_implementation": "flash_attention_2"},
        }
        self.processor_settings = {
            "Qwen/Qwen2-VL-7B-Instruct": {"min_pixels": 256*28*28, "max_pixels": 1280*28*28},
        }

        print(self.load(model_name_or_path, dataset_name_or_path).to_string(index=False))

    def load(self, model_name_or_path: str, dataset_name_or_path: str) -> dict:
        """load"""
        if model_name_or_path != self.model_name_or_path:
            self._clear_resources()
            self.model_name_or_path = model_name_or_path

            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True,
                **self.processor_settings.get(self.model_name_or_path, {})
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                **self.model_settings.get(self.model_name_or_path, {})
            )
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path
            self.dataset = load_dataset(self.dataset_name_or_path, split="test")

        tensor_type = next(self.model.parameters()).dtype
        total_params = sum(p.numel() for p in self.model.parameters())
        return pd.DataFrame(
            {
                "Model name": [self.model_name_or_path],
                "Dataset name": [self.dataset_name_or_path],
                "Tensor type": [str(tensor_type)],
                "Model size": [f"{total_params / 1e9:.2f}B"]
            }
        )

    def generate(self, texts: list | str, images: list | Image) -> list | str:
        """generate"""
        is_batch = isinstance(texts, list)
        if not is_batch:
            texts = [texts]
            images = [images]

        conversations = [
            [
                {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]}
            ]
            for text in texts
        ]
        prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]

        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        decoded_outputs = [
        self.processor.decode(
                generated_ids[i][len(inputs.input_ids[i]):], skip_special_tokens=True)
            for i in range(len(texts))
        ]

        return decoded_outputs if is_batch else decoded_outputs[0]

    def evaluate(self, batch_size: int = 1) -> float:
        """evaluate"""
        dataset = []

        if self.dataset_name_or_path == "m-a-p/CII-Bench":
            def format_question(example):
                option_labels = ["A", "B", "C", "D", "E", "F"]
                options = "\n".join([
                    f"({option_labels[i]}) {example[f'option{i + 1}']}"
                    for i in range(6) if example.get(f"option{i + 1}")
                ])
                example["input"] = (
                    "请根据提供的图片尝试回答下面的单选题。直接回答正确选项，不要包含额外的解释。"
                    "请使用以下格式：“答案：$LETTER”，其中$LETTER是你认为正确答案的字母。\n"
                   f"{example['question']}\n"
                   f"{options}\n"
                    "答案：\n"
                )
                return example
            dataset = self.dataset.map(format_question)

        correct = 0
        results = []
        total_examples = len(dataset)
        for start_idx in tqdm(range(0, total_examples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_examples)
            batch = dataset.select(range(start_idx, end_idx))
            predictions = self.generate(batch["input"], batch["image"])
            correct += self._count_correct(predictions, batch["answer"])
            results.extend([
                {"input": inp, "prediction": pred, "answer": ans}
                for inp, pred, ans in zip(batch["input"], predictions, batch["answer"])
            ])

        with open(f"{self.model_name_or_path.split("/")[-1]}.json", "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        return correct / total_examples

    def _clear_resources(self) -> None:
        if hasattr(self, "model"):
            delattr(self, "model")
        if hasattr(self, "processor"):
            delattr(self, "processor")
        torch.cuda.empty_cache()
        gc.collect()

    def _count_correct(self, predictions: list, answers: list) -> float:
        return sum(
            1
            for prediction, answer in zip(predictions, answers)
            if (match := re.search(r"\(?([A-F])\)?", prediction))
            and match.group(1).upper() == answer.upper()
        )
