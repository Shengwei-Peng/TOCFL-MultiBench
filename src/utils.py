"""utils"""
import os
import re
import gc
import json
import warnings

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from tabulate import tabulate
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,
    BitsAndBytesConfig,
    pipeline,
    logging
)

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

class MultimodalSystem:
    """MultimodalSystem"""
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str,
        tensor_type: str,
        asr_model_name_or_path: str | None = None,
    ) -> None:
        """__init__"""
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.tensor_type = ""
        self.model_name_or_path = ""
        self.asr_model_name_or_path = ""
        self.dataset_name_or_path = ""
        self.asr_model = None
        self.processor_config = {
            "Qwen/Qwen2-VL-7B-Instruct": {"min_pixels": 256*28*28, "max_pixels": 1280*28*28},
        }
        self.dataset_config = {
            "m-a-p/CII-Bench": "test",
            "Lin-Chen/MMStar": "val"
        }
        
        df = self.load(tensor_type, model_name_or_path, asr_model_name_or_path, dataset_name_or_path)
        print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

    def load(
        self,
        tensor_type: str,
        model_name_or_path: str,
        asr_model_name_or_path: str,
        dataset_name_or_path: str
    ) -> pd.DataFrame:
        """load"""
        if model_name_or_path != self.model_name_or_path or tensor_type != self.tensor_type:
            self._clear_resources(model_name_or_path)
            self.tensor_type = tensor_type
            self.model_name_or_path = model_name_or_path
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True,
                **self.processor_config.get(self.model_name_or_path, {})
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                **self._get_model_config(self.model_name_or_path, tensor_type)
            )
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        if (
            (asr_model_name_or_path != self.asr_model_name_or_path or tensor_type != self.tensor_type)
            and asr_model_name_or_path is not None
            ):
            self._clear_resources(asr_model_name_or_path)
            self.tensor_type = tensor_type
            self.asr_model_name_or_path = asr_model_name_or_path
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.asr_model_name_or_path,
                device_map="auto",
                **self._get_model_config(self.asr_model_name_or_path, tensor_type)
            )
            asr_processor = AutoProcessor.from_pretrained(self.asr_model_name_or_path)
            self.asr_model = pipeline(
                task="automatic-speech-recognition",
                model=asr_model,
                tokenizer=asr_processor.tokenizer,
                feature_extractor=asr_processor.feature_extractor
            )

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path
            if self.dataset_name_or_path not in self.dataset_config:
                dataset = load_dataset(
                    "json",
                    data_files=self.dataset_name_or_path,
                    split="train",
                )
            else:
                dataset = load_dataset(
                    self.dataset_name_or_path,
                    split=self.dataset_config[self.dataset_name_or_path],
                )
            self.dataset = Dataset.from_list([
                self._preprocess(example)
                for example in tqdm(dataset, desc="Processing dataset", unit="example")
            ])

        return pd.DataFrame(
            {
                "Model name": [self.model_name_or_path],
                "ASR Model name": [self.asr_model_name_or_path],
                "Dataset name": [self.dataset_name_or_path],
                "Tensor type": [self.tensor_type],
                "Model size": [
                    f"{sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B"
                ],
                "ASR Model size": [
                    f"{sum(p.numel() for p in self.asr_model.model.parameters()) / 1e9:.2f}B"
                    if self.asr_model is not None else "0B"
                ]
            }
        )

    def generate(
        self,
        texts: list | str,
        images: list | str = None,
        audios: list | str = None
    ) -> list | str:
        """generate"""
        is_batch = isinstance(texts, list)
        if not is_batch:
            texts = [texts]
            images = [images] if images is not None else [None] * len(texts)
            audios = [audios] if audios is not None else [None] * len(texts)

        if audios and self.asr_model is not None:
            audio_texts = [
                "".join([
                    segment['text'] for segment in self.asr_model(
                        audio, return_timestamps=True)["chunks"]
                ]) if audio else "" for audio in audios
            ]
            texts = [f"{audio_text}{text}" for audio_text, text in zip(audio_texts, texts)]

        if images:
            images = [Image.open(image) if isinstance(image, str) else image for image in images]
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}] +
            ([{"type": "image"}] if image else [])}
            for text, image in zip(texts, images)
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_prompt,
            images=images if any(images) else None,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text if is_batch else output_text[0]

    def evaluate(self, batch_size: int = 1) -> float:
        """evaluate"""
        correct = 0
        results = []
        total_examples = len(self.dataset)

        for start_idx in tqdm(range(0, total_examples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_examples)
            batch = self.dataset.select(range(start_idx, end_idx))

            generations = self.generate(batch["question"], batch["image"], batch["audio"])

            correct += self._count_correct(generations, batch["answer"])
            results.extend([
                {"id": i, "question": q, "generation": g, "answer": a}
                for i, q, g, a in zip(batch["id"], batch["question"], generations, batch["answer"])
            ])

        model_name = self.model_name_or_path.split("/")[-1]
        dataset_name = self.dataset_name_or_path.split("/")[-1]
        with open( f"{dataset_name}_{model_name}.json", "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        return correct / len(self.dataset)

    def _preprocess(self, example: dict) -> dict:
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
            example["audio"] = None

        elif self.dataset_name_or_path == "Lin-Chen/MMStar":
            example["audio"] = None

        else:
            options = "\n".join(f"{example[f'option{i + 1}']}" for i in range(4))
            example["question"] = f"{example['instruction']}\n{example['question']}\n{options}\n"

        return example

    def _clear_resources(self, name: str) -> None:
        if hasattr(self, name):
            delattr(self, name)
        torch.cuda.empty_cache()
        gc.collect()

    def _count_correct(self, generations: list, answers: list) -> float:
        return sum(
            1
            for generation, answer in zip(generations, answers)
            if (match := re.search(r"\(?([A-F])\)?", generation))
            and match.group(1).upper() == answer.upper()
        )

    def _get_model_config(self, model_name_or_path: str, tensor_type: str) -> dict:
        model_config = {
            "llava-hf/llava-1.5-7b-hf": {"attn_implementation": "flash_attention_2"},
            "llava-hf/llava-v1.6-mistral-7b-hf": {"attn_implementation": "flash_attention_2"},
            "Qwen/Qwen2-VL-7B-Instruct": {"attn_implementation": "flash_attention_2"},
            "Qwen/Qwen2-VL-2B-Instruct": {"attn_implementation": "flash_attention_2"},
            "openai/whisper-large-v3-turbo": {"attn_implementation": "flash_attention_2"},
        }.get(model_name_or_path, {})

        dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
        model_config["torch_dtype"] = dtype_mapping.get(tensor_type, "auto")

        if tensor_type in {"int8", "fp4", "nf4"}:
            model_config["quantization_config"] = BitsAndBytesConfig(**{
                "load_in_8bit": tensor_type == "int8",
                "load_in_4bit": tensor_type in {"fp4", "nf4"},
                "bnb_4bit_quant_type": "nf4" if tensor_type == "nf4" else "fp4",
                "bnb_4bit_compute_dtype": torch.float16,
            })

        return model_config
