"""utils"""
import gc
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import lmdeploy
import pandas as pd
import torch
from PIL import Image
from tabulate import tabulate
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import login
from lmdeploy.vl import load_image
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    logging,
    pipeline,
    set_seed,
)

from .stcm import STCM
from .utils import calculate_metrics

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class MultimodalSystem:
    """MultimodalSystem"""
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str,
        asr_model_name_or_path: str | None = None,
        prompt_template_path: str | None = None,
        tensor_type: str = "auto",
    ) -> None:
        """__init__"""
        set_seed(11207330)
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model_name_or_path: str = ""
        self.dataset_name_or_path: str = ""
        self.asr_model_name_or_path: str = ""
        self.prompt_template_path: str = ""
        self.tensor_type: str = ""

        self.load(
            model_name_or_path = model_name_or_path,
            dataset_name_or_path = dataset_name_or_path,
            asr_model_name_or_path = asr_model_name_or_path,
            prompt_template_path = prompt_template_path,
            tensor_type = tensor_type,
        )

    def load(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str,
        asr_model_name_or_path: str | None = None,
        prompt_template_path: str | None = None,
        tensor_type: str = "auto",
    ) -> pd.DataFrame:
        """load"""

        if prompt_template_path != self.prompt_template_path and prompt_template_path is not None:
            self.prompt_template_path = prompt_template_path
            try:
                with Path(prompt_template_path).open("r", encoding="utf-8") as file:
                    self.prompt_template = file.read()
            except (FileNotFoundError, IOError) as e:
                raise RuntimeError(f"Failed to load the prompt template: {e}") from e
        elif prompt_template_path is None:
            self.prompt_template = "{question}"

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
        elif asr_model_name_or_path is None:
            self.asr_model = None

        if dataset_name_or_path != self.dataset_name_or_path:
            self.dataset_name_or_path = dataset_name_or_path

            if self.dataset_name_or_path == "m-a-p/CII-Bench":
                dataset = load_dataset(self.dataset_name_or_path, split="test")
                self.all_choices = ["A", "B", "C", "D", "E", "F"]

            elif self.dataset_name_or_path == "Lin-Chen/MMStar":
                dataset = load_dataset(self.dataset_name_or_path, split="val")
                self.all_choices = ["A", "B", "C", "D"]

            elif self.dataset_name_or_path == "lmms-lab/MMBench":
                dataset = load_dataset(self.dataset_name_or_path, "cc", split="test")
                self.all_choices = ["A", "B", "C", "D"]

            else:
                dataset = load_dataset(
                    "json",
                    data_files=self.dataset_name_or_path,
                    split="train",
                )
                self.all_choices = ["A", "B", "C", "D"]

            self.dataset = Dataset.from_list([
                self._preprocess(example)
                for example in tqdm(dataset, desc="Processing dataset", unit="example")
            ])

        self.stcm = STCM(allowed_tokens=self.all_choices, tokenizer=self.processor.tokenizer)
        df = pd.DataFrame(
            {
                "Model": [self.model_name_or_path],
                "ASR Model": [self.asr_model_name_or_path],
                "Dataset": [self.dataset_name_or_path],
                "Prompt": [self.prompt_template_path],
                "Tensor type": [self.tensor_type],
                "Dataset size": [len(self.dataset)],
                "Model size": [
                    f"{sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B"
                ] if not self.is_lmdeploy_model else [None],
                "ASR Model size": [
                    f"{sum(p.numel() for p in self.asr_model.model.parameters()) / 1e9:.2f}B"
                    if self.asr_model is not None else "0B"
                ],
            }
        )
        print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
        return df

    def evaluate(
        self,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy",
        use_stcm: bool = False,
    ) -> pd.DataFrame:
        """evaluate"""
        total_examples = len(self.dataset)
        batch_size = 1
        results = []
        all_response = []
        all_answers = []
        all_index2ans = []

        start_time = time.time()
        for start_idx in tqdm(range(0, total_examples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_examples)
            batch = self.dataset.select(range(start_idx, end_idx))

            generations = self.generate(
                batch["question"], batch["image"], batch["audio"],
                max_new_tokens=max_new_tokens,
                decoding_strategy=decoding_strategy,
                use_stcm = use_stcm,
            )

            batch_index2ans = (
                batch["index2ans"] if "index2ans" in batch.column_names
                else [None] * len(batch["id"])
            )

            all_response.extend(generations)
            all_answers.extend(batch["answer"])
            all_index2ans.extend(batch_index2ans)

            results.extend([
                {
                    "id": batch["id"][i],
                    "question": batch["question"][i],
                    "generation": generations[i],
                    "answer": batch["answer"][i],
                    "index2ans": batch_index2ans[i],
                }
                for i in range(len(batch["id"]))
            ])

        end_time = time.time()

        metrics = calculate_metrics(
            all_choices=self.all_choices,
            all_answers=all_answers,
            all_response=all_response,
            all_index2ans=all_index2ans
        )

        output_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "tensor_type": self.tensor_type,
            "model_name_or_path": self.model_name_or_path,
            "asr_model_name_or_path": self.asr_model_name_or_path,
            "dataset_name_or_path": self.dataset_name_or_path,
            "max_new_tokens": max_new_tokens,
            "decoding_strategy": decoding_strategy,
            "use_stcm": use_stcm,
            "total_examples": total_examples,
            "runtime": end_time - start_time,
            "prompt_template": self.prompt_template,
            **metrics
        }

        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8"
        )

        (output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=4), encoding="utf-8"
        )

        metrics_df = pd.DataFrame([{key: round(value * 100, 2) for key, value in metrics.items()}])
        print(tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=False))
        return metrics_df

    def generate(
        self,
        texts: list | str,
        images: list | str = None,
        audios: list | str = None,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy",
        use_stcm: bool = False,
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
                elif isinstance(image, Image.Image):
                    images[i] = image.convert("RGB")
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
                audios=audios if audios else None,
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
            decoding_strategy=decoding_strategy,
            use_stcm=use_stcm
        )
        return output_text if is_batch else output_text[0]

    def _generate_text(
        self,
        inputs: dict,
        max_new_tokens: int = 1,
        decoding_strategy: str = "greedy",
        use_stcm: bool = False,
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
        params["max_new_tokens"] = max_new_tokens
        if use_stcm:
            params["logits_processor"] = [self.stcm]

        generated_ids = self.model.generate(**inputs, **params)

        if use_stcm:
            return self.stcm.generate()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _preprocess(self, example: dict) -> dict:

        if "audio" not in example:
            example["audio"] = None

        if "TOCFL-MultiBench" in self.dataset_name_or_path:
            example["question"] = (
                f"{example['instruction']}\n"
                + f"{example['question']}\n"
                + "".join(example[f"option{i + 1}"] for i in range(len(self.all_choices)))
            )

        elif self.dataset_name_or_path == "m-a-p/CII-Bench":
            options = "\n".join(
                f"({self.all_choices[i]}) {example[f'option{i + 1}']}"
                for i in range(len(self.all_choices))
            )
            example["question"] = (
                "请根据提供的图片尝试回答下面的单选题。直接回答正确选项，不要包含额外的解释。"
                + "请使用以下格式：“答案：$LETTER”，其中$LETTER是你认为正确答案的字母。\n\n"
                + f"{example['question']}\n"
                + f"{options}\n\n"
                + "答案："
            )
            example["index2ans"] = {
                label: example[f"option{idx + 1}"]
                for idx, label in enumerate(self.all_choices)
            }

        elif self.dataset_name_or_path == "Lin-Chen/MMStar":
            example["id"] = example["index"]

        elif self.dataset_name_or_path == "lmms-lab/MMBench":
            example["id"] = example["index"]
            options = "\n".join(f"({i}) {example[i]}" for i in self.all_choices)
            example["question"] = f"{example['question']}\n{options}\n"
            example["index2ans"] = {
                key: example[key] for key in self.all_choices if key in example
            }

        example["question"] = self.prompt_template.format(question=example["question"])

        return example

    def _clear_resources(self, name: str) -> None:
        if hasattr(self, name):
            delattr(self, name)
        torch.cuda.empty_cache()
        gc.collect()

    def _get_model_config(self, tensor_type: str) -> dict:
        model_config = {}
        if (
            tensor_type in ["fp16", "bf16"] and not self.is_audio_language_model
            and "meta-llama" not in self.model_name_or_path
        ):
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
