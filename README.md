# TOCFL-MultiBench: A Multimodal Benchmark for Evaluating Chinese Language Proficiency

## 🌟 Overview
TOCFL-MultiBench is a comprehensive multimodal benchmark designed to evaluate Chinese language proficiency across diverse dimensions. It is inspired by the Test of Chinese as a Foreign Language (TOCFL) and integrates deep learning techniques to assess proficiency through multimodal data, including text, audio, and visual inputs.

## 💻 Installation

### Prerequisites

- Python 3.12.7
- CUDA 12.4

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Shengwei-Peng/TOCFL-MultiBench.git
   ```

2. Navigate to the project directory:
    ```sh
    cd TOCFL-MultiBench
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🛠️ Usage

### 🗂️ Data Collection with `collector.py`
The `collector.py` script is a manual data collection tool designed to organize collected data into a structured directory format.
```
python collector.py --dataset_dir TOCFL-MultiBench
```

#### Arguments
- **`--dataset_dir`** *(optional)*:  
  Path to the root directory where the collected data will be stored. Defaults to `"TOCFL-MultiBench"`.  

#### Output Structure
```
TOCFL-MultiBench/
├── audios/                  # Audio files
├── images/                  # Image files
└── TOCFL-MultiBench.json    # Metadata and labels
```

### 🧪 Experiment with `experiment.py`
The `experiment.py` script is designed to evaluate a multimodal system using the provided model, dataset, and optional configurations.
```
python experiment.py \
    --dataset_name_or_path "TOCFL-MultiBench/TOCFL-MultiBench.json" \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --asr_model_name_or_path "openai/whisper-large-v3-turbo" \
    --batch_size 1 \
    --max_new_tokens 1 \
    --tensor_type auto \
    --decoding_strategy greedy \
    --use_stcm
```

#### Arguments
- **`--model_name_or_path`** *(required)*:  
  Path to the model or model name to be used for evaluation.

- **`--dataset_name_or_path`** *(required)*:  
  Path to the dataset or dataset name to be evaluated.

- **`--asr_model_name_or_path`** *(optional)*:  
  Path to the Automatic Speech Recognition (ASR) model or its name. Defaults to `None`.

- **`--batch_size`** *(optional)*:  
  Number of samples per batch. Defaults to `1`.

- **`--max_new_tokens`** *(optional)*:  
  Maximum number of tokens to generate for each prediction. Defaults to `1`.

- **`--tensor_type`** *(optional)*:  
  Specifies the tensor type (e.g., `auto`, `bf16`, etc.). Defaults to `"auto"`.

- **`--decoding_strategy`** *(optional)*:  
  Strategy for decoding model outputs (e.g., `greedy`, `beam_search`). Defaults to `"greedy"`.

- **`--use_stcm`** *(optional)*:  
  Flag to enable the use of Selective Token Constraint Mechanism (STCM). Set this flag to activate.

#### Output Structure
```
20241218_235900/    # Timestamped directory
├── config.json     # Experiment configuration
└── results.json    # Evaluation results
```

## 📊 Results

#### Performance Comparison Across Different Methods
|    Method    |                     Model                      | Model Size | Accuracy | F1 Score | Precision | Recall |
|:------------:|:----------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
|     VLM      |              Qwen2-VL-7B-Instruct              |   8.29B    |  43.89%  |  51.76%  |  67.86%   | 43.89% |
|     ALM      |            Qwen2-Audio-7B-Instruct             |   8.40B    |  30.22%  |  28.13%  |  42.83%   | 30.22% |
| VLM + Random |              Qwen2-VL-7B-Instruct              |   8.29B    |  51.89%  |  51.54%  |  51.40%   | 51.89% |
| ALM + Random |            Qwen2-Audio-7B-Instruct             |   8.40B    |  35.22%  |  31.03%  |  35.59%   | 35.22% |
|  VLM + ALM   | Qwen2-VL-7B-Instruct + Qwen2-Audio-7B-Instruct |   16.69B   |  52.33%  |  57.84%  |  66.95%   | 52.33% |
|  VLM + ASR   | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.22%  |  79.27%  |  80.15%   | 79.22% |

#### Performance Comparison Across Different Models
|                      Model                      | Model Size | Accuracy | F1 Score | Precision | Recall |
|:-----------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
| Phi-3.5-Vision-Instruc + Whisper-Large-V3-Turbo |   4.96B    |  49.78%  |  49.27%  |  55.88%   | 49.78% |
|     LLaVA-v1.5-7B + Whisper-Large-V3-Turbo      |   7.87B    |  31.89%  |  21.08%  |  62.19%   | 31.89% |
|  LLaVA-NeXT-Vicuna-7B + Whisper-Large-V3-Turbo  |   8.38B    |  33.56%  |  25.69%  |  53.81%   | 33.56% |
| LLaVA-NeXT-Mistral-7B + Whisper-Large-V3-Turbo  |   8.38B    |  41.00%  |  38.25%  |  55.63%   | 41.00% |
|      InternVL2-8B + Whisper-Large-V3-Turbo      |   8.89B    |  78.00%  |  77.90%  |  78.24%   | 78.00% |
|      MiniCPM-v2.6 + Whisper-Large-V3-Turbo      |   8.91B    |  75.78%  |  75.76%  |  75.88%   | 75.78% |
|  Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.22%  |  79.27%  |  80.15%   | 79.22% |
|      Idefics2-8B + Whisper-Large-V3-Turbo       |   9.21B    |  41.22%  |  41.08%  |  42.05%   | 41.22% |
|   MiniCPM-Llama3-2.5 + Whisper-Large-V3-Turbo   |   9.35B    |  63.89%  |  63.85%  |  64.40%   | 63.89% |
|     Paligemma2-10B + Whisper-Large-V3-Turbo     |   10.47B   |  24.33%  |  24.72%  |  25.63%   | 24.33% |
|  Llama-3.2-11B-Vision-Instruct + Whisper-Small  |   10.94B   |  54.33%  |  53.78%  |  61.12%   | 54.33% |

#### Performance Comparison Across Decoding Strategies
|                     Model                     |        Decoding Strategy         | Model Size |  Accuracy  |  F1 Score  | Precision  |   Recall   |
|:---------------------------------------------:|:--------------------------------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |          Greedy Search           |   9.10B    |   79.22%   |   79.27%   |   80.15%   |   79.22%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |        Contrastive Search        |   9.10B    |   79.22%   |   79.27%   |   80.15%   |   79.22%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |       Multinomial Sampling       |   9.10B    |   79.22%   |   79.27%   |   80.15%   |   79.22%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |           Beam Search            |   9.10B    |   61.67%   |   61.59%   |   64.35%   |   61.67%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo | Beam Search Multinomial Sampling |   9.10B    |   61.67%   |   61.59%   |   64.35%   |   61.67%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |       Diverse Beam Search        |   9.10B    |   61.67%   |   61.59%   |   64.35%   |   61.67%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |         Self-Speculative         |   9.10B    |   79.67%   |   79.69%   |   80.48%   |   79.67%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |           DoLa (High)            |   9.10B    |   58.11%   |   58.18%   |   68.29%   |   58.11%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            DoLa (Low)            |   9.10B    |   79.89%   |   79.85%   |   80.91%   |   79.89%   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |        DoLa (Low) + STCM         |   9.10B    | **80.22%** | **80.18%** | **81.24%** | **80.22%** |

### ✨ Selective Token Constraint Mechanism (STCM)

#### Performance Comparison with and without STCM
|      Method      |                     Model                      | Model Size | Accuracy | F1 Score | Precision | Recall |
|:----------------:|:----------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
|       VLM        |              Qwen2-VL-7B-Instruct              |   8.29B    |  43.89%  |  51.76%  |  67.86%   | 43.89% |
|       ALM        |            Qwen2-Audio-7B-Instruct             |   8.40B    |  30.22%  |  28.13%  |  42.83%   | 30.22% |
|   VLM + Random   |              Qwen2-VL-7B-Instruct              |   8.29B    |  51.89%  |  51.54%  |  51.40%   | 51.89% |
|   ALM + Random   |            Qwen2-Audio-7B-Instruct             |   8.40B    |  35.22%  |  31.03%  |  35.59%   | 35.22% |
|    VLM + STCM    |              Qwen2-VL-7B-Instruct              |   8.29B    |  53.78%  |  52.01%  |  59.80%   | 53.78% |
|    ALM + STCM    |            Qwen2-Audio-7B-Instruct             |   8.40B    |  36.67%  |  32.32%  |  37.77%   | 36.67% |
|    VLM + ALM     | Qwen2-VL-7B-Instruct + Qwen2-Audio-7B-Instruct |   16.69B   |  52.33%  |  57.84%  |  66.95%   | 52.33% |
|    VLM + ASR     | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.22%  |  79.27%  |  80.15%   | 79.22% |
| VLM + ASR + STCM | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.33%  |  79.34%  |  80.13%   | 79.33% |

#### Performance Comparison with and without STCM Across Models
|      Method      |                     Model                     | Model Size | Accuracy | F1 Score | Precision | Recall |
|:----------------:|:---------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
|    VLM + ASR     |     Idefics2-8B + Whisper-Large-V3-Turbo      |   9.21B    |  41.22%  |  41.08%  |  42.05%   | 41.22% |
| VLM + ASR + STCM |     Idefics2-8B + Whisper-Large-V3-Turbo      |   9.21B    |  42.33%  |  42.50%  |  43.26%   | 42.33% |
|    VLM + ASR     | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |   9.10B    |  79.22%  |  79.27%  |  80.15%   | 79.22% |
| VLM + ASR + STCM | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |   9.10B    |  79.33%  |  79.34%  |  80.13%   | 79.33% |
|    VLM + ASR     |    Paligemma2-10B + Whisper-Large-V3-Turbo    |   10.47B   |  24.33%  |  24.72%  |  25.63%   | 24.33% |
| VLM + ASR + STCM |    Paligemma2-10B + Whisper-Large-V3-Turbo    |   10.47B   |  27.89%  |  12.51%  |  25.08%   | 27.89% |
|    VLM + ASR     | Llama-3.2-11B-Vision-Instruct + Whisper-Small |   10.94B   |  54.33%  |  53.78%  |  61.12%   | 54.33% |
| VLM + ASR + STCM | Llama-3.2-11B-Vision-Instruct + Whisper-Small |   10.94B   |  55.44%  |  55.08%  |  64.79%   | 55.44% |

## 🙏 Acknowledgements

We would like to express our gratitude to the following organizations for their support:

- **Test of Chinese as a Foreign Language (TOCFL)**: For providing valuable datasets that contributed to this project.
- **NTU Miulab**: For offering technical guidance and expertise throughout the development of this project.

## 🤝 Contributing

We welcome contributions to the project! Please follow the guidelines below:

1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Commit your changes.
4. Submit a pull request.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## 📧 Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw