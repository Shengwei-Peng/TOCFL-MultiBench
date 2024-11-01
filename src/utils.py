"""utils"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class MultimodalModel:
    """MultimodalModel"""
    def __init__(self, initial_model_name: str) -> None:
        """__init__"""
        self.model_name = initial_model_name
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )

    def update_model(self, model_name: str) -> None:
        """update_model"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )

    def generate(self, image: Image, text: str, model_name: str) -> None:
        """generate"""
        self.update_model(model_name)
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
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
