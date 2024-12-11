"""utils"""
import os
import re
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import torch
import librosa
import pandas as pd
import lmdeploy
from lmdeploy.vl import load_image
from PIL import Image
from tqdm.auto import tqdm
from tabulate import tabulate
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    logging,
    pipeline,
    set_seed
)

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

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
        set_seed(11207330)
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.prompt = "僅輸出正確答案的字母，格式必須為 'A', 'B', 'C', 'D'，輸出限制為單個字母，無需解釋。\n"
        self.tensor_type = ""
        self.model_name_or_path = ""
        self.asr_model_name_or_path = ""
        self.dataset_name_or_path = ""
        self.asr_model = None

        df = self.load(
            tensor_type, model_name_or_path, asr_model_name_or_path, dataset_name_or_path
        )
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
            self.is_audio_language_model = "Audio" in self.model_name_or_path
            self.is_lmdeploy_model = "lmdeploy:" in self.model_name_or_path
            self.is_seq2seq =  "lmdeploy:" not in self.model_name_or_path

            if self.is_lmdeploy_model:
                self.model_name_or_path = self.model_name_or_path.replace("lmdeploy:", "")
                self.model = lmdeploy.pipeline(
                    self.model_name_or_path,
                    backend_config=lmdeploy.TurbomindEngineConfig(
                        session_len=8192, cache_max_entry_count=0.2
                    ),
                )

            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True,
                    **(
                        {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}
                        if "Qwen" in self.model_name_or_path else {}
                    )
                )

                if self.is_audio_language_model:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name_or_path,
                        trust_remote_code=True,
                        device_map="cuda",
                        **self._get_model_config(tensor_type)
                    ).eval()
                else:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name_or_path,
                        trust_remote_code=True,
                        device_map="cuda",
                        **self._get_model_config(tensor_type)
                    ).eval()

                self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        if asr_model_name_or_path is not None and (
            asr_model_name_or_path != self.asr_model_name_or_path or tensor_type != self.tensor_type
        ):
            self._clear_resources(asr_model_name_or_path)
            self.tensor_type = tensor_type
            self.asr_model_name_or_path = asr_model_name_or_path
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.asr_model_name_or_path,
                device_map="auto",
                **self._get_model_config(tensor_type)
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
            dataset = load_dataset(
                "json",
                data_files=self.dataset_name_or_path,
                split="train",
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
                ] if not self.is_lmdeploy_model else [None],
                "ASR Model size": [
                    f"{sum(p.numel() for p in self.asr_model.model.parameters()) / 1e9:.2f}B"
                    if self.asr_model is not None else "0B"
                ],
            }
        )

    def evaluate(
        self,
        batch_size: int = 1,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy"
    ) -> float:
        """evaluate"""
        correct = 0
        results = []
        total_examples = len(self.dataset)

        start_time = time.time()
        for start_idx in tqdm(range(0, total_examples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_examples)
            batch = self.dataset.select(range(start_idx, end_idx))

            generations = self.generate(
                batch["question"], batch["image"], batch["audio"],
                max_new_tokens=max_new_tokens, decoding_strategy=decoding_strategy
            )
            correct += self._count_correct(generations, batch["answer"])
            results.extend([
                {
                    "id": batch["id"][i],
                    "question": batch["question"][i],
                    "image": batch["image"][i],
                    "audio": batch["audio"][i],
                    "generation": generations[i],
                    "answer": batch["answer"][i],
                }
                for i in range(len(batch["id"]))
            ])

        end_time = time.time()
        accuracy = correct / total_examples

        output_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "tensor_type": self.tensor_type,
            "model_name_or_path": self.model_name_or_path,
            "asr_model_name_or_path": self.asr_model_name_or_path,
            "dataset_name_or_path": self.dataset_name_or_path,
            "max_new_tokens": max_new_tokens,
            "decoding_strategy": decoding_strategy,
            "total_examples": total_examples,
            "correct": correct,
            "accuracy": accuracy,
            "runtime": end_time - start_time,
        }

        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8"
        )

        (output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=4), encoding="utf-8"
        )

        return accuracy

    def generate(
        self,
        texts: list | str,
        images: list | str = None,
        audios: list | str = None,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy",
    ) -> list | str:
        """generate"""
        is_batch = isinstance(texts, list)

        if not is_batch:
            texts = [texts]
            images = [images] if images is not None else [None] * len(texts)
            audios = [audios] if audios is not None else [None] * len(texts)

        if audios and self.asr_model is not None and not self.is_audio_language_model:
            audio_texts = [
                "".join([
                    segment['text'] for segment in self.asr_model(
                        audio, return_timestamps=True)["chunks"]
                ]) if audio else "" for audio in audios
            ]
            texts = [f"{audio_text}\n{text}" for audio_text, text in zip(audio_texts, texts)]

        if self.is_lmdeploy_model:
            if images:
                image = load_image(images[0])
                response = self.model((texts[0], image))
            else:
                response = self.model((texts[0]))
            return [response.text.strip()]

        if images:
            requires_image = any(
                keyword in self.model_name_or_path
                for keyword in ["llava", "paligemma"]
            )
            for i, image in enumerate(images):
                if isinstance(image, str):
                    images[i] = Image.open(image).convert("RGB")
                elif image is None and requires_image:
                    images[i] = Image.new("RGB", (224, 224), (255, 255, 255))

        conversation = []
        for text, image, audio in zip(texts, images, audios):
            content = []

            if audio and self.is_audio_language_model:
                content.append({"type": "audio", "audio": audio})

            if image and not self.is_audio_language_model:
                content.append({"type": "image"})

            if text:
                content.append({"type": "text", "text": text})

            conversation.append({"role": "user", "content": content})

        if (
            getattr(self.processor, "apply_chat_template", None)
            and getattr(self.processor, "chat_template", None)
        ):
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            ) if self.processor.chat_template else conversation
        else:
            text_prompt = conversation

        if self.is_audio_language_model:
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(librosa.load(
                                ele["audio"], sr=self.processor.feature_extractor.sampling_rate
                                )[0]
                            )

            inputs = self.processor(
                text=text_prompt,
                audios=audios,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

        elif "paligemma" in self.model_name_or_path:
            inputs = self.processor(
                text=texts,
                images=images if any(images) else None,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

        else:
            inputs = self.processor(
                text=text_prompt,
                images=images if any(images) else None,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

        output_text = self._generate_text(
            inputs,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy
        )
        return output_text if is_batch else output_text[0]

    def _generate_text(
        self,
        inputs: dict,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy"
    ) -> str | list:
        strategy_params = {
            "greedy": {},
            "contrastive": {"penalty_alpha": 0.6, "top_k": 4},
            "sampling": {"do_sample": True, "num_beams": 1},
            "beam_search": {"num_beams": 5},
            "beam_search_sampling": {"do_sample": True, "num_beams": 5},
            "diverse_beam_search": {
                "do_sample": False, "num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0
            },
            "self_speculative": {"do_sample": False, "assistant_early_exit": 4},
            "dola_high": {"do_sample": False, "dola_layers": "high"},
            "dola_low": {"do_sample": False, "dola_layers": "low"}
        }

        params = strategy_params.get(decoding_strategy, {})
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **params)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _preprocess(self, example: dict) -> dict:
        example["question"] = (
            f"{example['instruction']}\n"
            + f"{example['question']}\n"
            + "".join(example[f"option{i + 1}"] for i in range(4)) + "\n"
            + self.prompt
        )
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
            if re.fullmatch(r"[A-F]", generation)
            and generation.upper() == answer.upper()
        )

    def _get_model_config(self, tensor_type: str) -> dict:
        model_config = {}
        if tensor_type != "fp16" and not self.is_audio_language_model:
            model_config["attn_implementation"] = "flash_attention_2"

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
