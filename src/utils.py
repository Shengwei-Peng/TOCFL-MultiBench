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
from datasets import load_dataset, Dataset
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
        self.model_config = {
            "llava-hf/llava-1.5-7b-hf": {"attn_implementation": "flash_attention_2"},
            "llava-hf/llava-v1.6-mistral-7b-hf": {"attn_implementation": "flash_attention_2"},
            "Qwen/Qwen2-VL-7B-Instruct": {"attn_implementation": "flash_attention_2"},
        }
        self.processor_config = {
            "Qwen/Qwen2-VL-7B-Instruct": {"min_pixels": 256*28*28, "max_pixels": 1280*28*28},
        }
        self.dataset_config = {
            "m-a-p/CII-Bench": "test",
            "Lin-Chen/MMStar": "val"
        }
        print(self.load(model_name_or_path, dataset_name_or_path).to_string(index=False))

    def load(self, model_name_or_path: str, dataset_name_or_path: str) -> dict:
        """load"""
        if model_name_or_path != self.model_name_or_path:
            self._clear_resources()
            self.model_name_or_path = model_name_or_path

            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True,
                **self.processor_config.get(self.model_name_or_path, {})
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                **self.model_config.get(self.model_name_or_path, {})
            )
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path
            self.dataset = load_dataset(
                self.dataset_name_or_path,
                split=self.dataset_config[self.dataset_name_or_path]
            )

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
        correct = 0
        results = []
        total_examples = len(self.dataset)
        dataset = self.dataset.map(self._format_question)

        for start_idx in tqdm(range(0, total_examples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_examples)
            batch = dataset.select(range(start_idx, end_idx))

            generations = self.generate(batch["question"], batch["image"])
            correct += self._count_correct(generations, batch["answer"])
            results.extend([
                {"question": q, "generation": g, "answer": a}
                for q, g, a in zip(batch["question"], generations, batch["answer"])
            ])

        model_name = self.model_name_or_path.split("/")[-1]
        dataset_name = self.dataset_name_or_path.split("/")[-1]
        with open( f"{dataset_name}_{model_name}.json", "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        return correct / len(self.dataset)

    def _format_question(self, example: Dataset) -> Dataset:
        if self.dataset_name_or_path == "m-a-p/CII-Bench":
            option_labels = ["A", "B", "C", "D", "E", "F"]
            options = "\n".join(
                f"({option_labels[i]}) {example[f'option{i + 1}']}"
                for i in range(6)
            )
            example["question"] = (
                "请根据提供的图片尝试回答下面的单选题。直接回答正确选项，不要包含额外的解释。\n"
                "请使用以下格式：“答案：$LETTER”，其中$LETTER是你认为正确答案的字母。\n"
                f"{example['question']}\n{options}\n答案：\n"
            )
        return example

    def _clear_resources(self) -> None:
        if hasattr(self, "model"):
            delattr(self, "model")
        if hasattr(self, "processor"):
            delattr(self, "processor")
        torch.cuda.empty_cache()
        gc.collect()

    def _count_correct(self, generations: list, answers: list) -> float:
        return sum(
            1
            for generation, answer in zip(generations, answers)
            if (match := re.search(r"\(?([A-F])\)?", generation))
            and match.group(1).upper() == answer.upper()
        )
