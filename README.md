# Chinese-Multimodal-Hallucination-Mitigation

# 🧪 Experiment

| Model            | Model name                               | Dataset name    | Tensor type | Model size | Accuracy |
| ---------------- | ---------------------------------------- | --------------- | ----------- | ----------:| --------:|
| LLaVA-v1.5       | llava-hf/llava-1.5-7b-hf                 | m-a-p/CII-Bench |    FP16     |      7.06B |   20.78% |
| LLaVA-NeXT       | llava-hf/llava-v1.6-mistral-7b-hf        | m-a-p/CII-Bench |    FP16     |      7.57B |   27.97% |
| Llama 3.2-Vision | meta-llama/Llama-3.2-11B-Vision-Instruct | m-a-p/CII-Bench |    BF16     |     10.67B |   30.72% |
| Qwen2-VL         | Qwen/Qwen2-VL-7B-Instruct                | m-a-p/CII-Bench |    BF16     |      8.29B |   41.83% |
| LLaVA-v1.5       | llava-hf/llava-1.5-7b-hf                 | Lin-Chen/MMStar |    FP16     |      7.06B |   31.67% |
| LLaVA-NeXT       | llava-hf/llava-v1.6-mistral-7b-hf        | Lin-Chen/MMStar |    FP16     |      7.57B |   26.40% |
| Llama 3.2-Vision | meta-llama/Llama-3.2-11B-Vision-Instruct | Lin-Chen/MMStar |    BF16     |     10.67B |    4.07% |
| Qwen2-VL         | Qwen/Qwen2-VL-7B-Instruct                | Lin-Chen/MMStar |    BF16     |      8.29B |   26.13% |