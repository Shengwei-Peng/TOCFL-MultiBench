# TOCFL-MultiBench: A Multimodal Benchmark for Evaluating Chinese Language Proficiency

## 🌟 Overview
TOCFL-MultiBench is a comprehensive multimodal benchmark designed to evaluate Chinese language proficiency through multiple-choice questions. Inspired by the Test of Chinese as a Foreign Language (TOCFL), it incorporates deep learning techniques to assess proficiency using multimodal data such as text, audio, and visual inputs. The benchmark introduces a novel method called Selective Token Constraint Mechanism (STCM), which enhances decoding stability and performance on multiple-choice questions without additional computational cost or fine-tuning.

## 💻 Installation

### Prerequisites

- Python 3.12.7
- CUDA 12.4
- GPU with 24GB+ VRAM recommended

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

4. Download the TOCFL-MultiBench and place it in the appropriate directory. You can either:
    - Download it manually from [this link](https://drive.google.com/file/d/1hspOJO9e5-m_0GFU84RKmzx9Np8RRZrA/view?usp=drive_link)  
    - Or use `gdown`:

      ```bash
      pip install gdown
      gdown https://drive.google.com/uc?id=1hspOJO9e5-m_0GFU84RKmzx9Np8RRZrA
      unzip TOCFL-MultiBench.zip
      ```

5. Create a `.env` file and add your `HUGGINGFACE_TOKEN`:
    ```bash
    echo "HUGGINGFACE_TOKEN=your_huggingface_token_here" > .env
    ```
    Replace `your_huggingface_token_here` with your actual token from Hugging Face. If you don’t have one, you can create an account on [Hugging Face](https://huggingface.co/) and generate a token in your user settings.

## 🛠️ Usage

### 🧪 Experiment with `experiment.py`
The `experiment.py` script is designed to evaluate a multimodal system using the provided model, dataset, and optional configurations.
```bash
python experiment.py \
  --dataset_name_or_path "TOCFL-MultiBench/TOCFL-MultiBench.json" \
  --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct"
```

#### 🏆 Reproducing Best Results
To reproduce the best results as shown in the `best/` directory of the repository, execute the following command using the specific configurations:
```bash
python experiment.py \
  --dataset_name_or_path "TOCFL-MultiBench/TOCFL-MultiBench.json" \
  --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
  --asr_model_name_or_path "openai/whisper-large-v3-turbo" \
  --prompt_template_path "prompt/base.txt" \
  --max_new_tokens 1 \
  --tensor_type "bf16" \
  --decoding_strategy "dola_low" \
  --use_stcm
```

#### Arguments
- **`--model_name_or_path`** *(required)*:  
  Path to the model or model name to be used for evaluation.

- **`--dataset_name_or_path`** *(required)*:  
  Path to the dataset or dataset name to be evaluated.

- **`--asr_model_name_or_path`** *(optional)*:  
  Path to the Automatic Speech Recognition (ASR) model or its name. Defaults to `None`.

- **`--prompt_template_path`** *(optional)*:  
  Path to a prompt template file. Defaults to `None`.

- **`--max_new_tokens`** *(optional)*:  
  Maximum number of tokens to generate for each prediction. Defaults to `1`.

- **`--tensor_type`** *(optional)*:  
  Specifies the tensor type for computations.  
  **Options**:  
  `auto` | `fp16` | `bf16` | `int8` | `fp4` | `nf4`  
  *(Default: `auto`)*

- **`--decoding_strategy`** *(optional)*:  
  Strategy for decoding model outputs.  
  **Options**:  
  `greedy` | `contrastive` | `sampling` | `beam_search` | `beam_search_sampling`  
  `diverse_beam_search` | `self_speculative` | `dola_high` | `dola_low`  
  *(Default: `greedy`)*

- **`--use_stcm`** *(optional)*:  
  Flag to enable the use of Selective Token Constraint Mechanism (STCM). Set this flag to activate.

#### Output Structure
```
20241218_235900/    # Timestamped directory
├── config.json     # Experiment configuration
└── results.json    # Evaluation results
```

### 🌐 Web Interface with `interface.py`
The `interface.py` script provides a graphical user interface (GUI) for interacting with the `TOCFL-MultiBench` system, allowing you to select models, datasets, and configurations, and to visualize results interactively.
```bash
python interface.py \
  --dataset_name_or_path "TOCFL-MultiBench/TOCFL-MultiBench.json" \
  --prompt_dir "prompt"
```
#### Arguments
- **`--dataset_name_or_path`** *(optional)*:  
  Path to the dataset or dataset name to be evaluated. Defaults to `"TOCFL-MultiBench/TOCFL-MultiBench.json"`.  
- **`--prompt_dir`** *(optional)*:  
  Directory containing prompt templates for evaluation. Defaults to `"prompt"`.

Unlike `experiment.py`, which requires rerunning for each configuration, `interface.py` allows dynamic adjustments via the **Load** button. Each **Evaluate** button click automatically creates a timestamped folder with the configuration and results.
#### Output Structure
```
20241218_235900/    # Timestamped directory
├── config.json     # Experiment configuration
└── results.json    # Evaluation results
```

### 🗂️ Data Collection with `collector.py`
The `collector.py` script is a manual data collection tool designed to organize collected data into a structured directory format.
```bash
python collector.py --dataset_dir "TOCFL-MultiBench"
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

## 📊 Results

#### Performance Comparison Across Different Methods
<details>
  <summary>Show/Hide Results Table</summary>

|    Method    |                     Model                      | Model Size | Accuracy | F1 Score | Precision | Recall |
|:------------:|:----------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
|     VLM      |              Qwen2-VL-7B-Instruct              |   8.29B    |  43.89   |  51.76   |   67.86   | 43.89  |
|     ALM      |            Qwen2-Audio-7B-Instruct             |   8.40B    |  30.22   |  28.13   |   42.83   | 30.22  |
| VLM + Random |              Qwen2-VL-7B-Instruct              |   8.29B    |  51.89   |  51.54   |   51.40   | 51.89  |
| ALM + Random |            Qwen2-Audio-7B-Instruct             |   8.40B    |  35.22   |  31.03   |   35.59   | 35.22  |
|  VLM + ALM   | Qwen2-VL-7B-Instruct + Qwen2-Audio-7B-Instruct |   16.69B   |  52.33   |  57.84   |   66.95   | 52.33  |
|  VLM + ASR   | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.11   |  79.16   |   79.97   | 79.11  |
</details>

#### Performance Comparison Across Different Models
<details>
  <summary>Show/Hide Results Table</summary>

|                      Model                      | Model Size | Accuracy | F1 Score | Precision | Recall |
|:-----------------------------------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
| Phi-3.5-Vision-Instruc + Whisper-Large-V3-Turbo |   4.96B    |  49.78   |  49.27   |   55.88   | 49.78  |
|     LLaVA-v1.5-7B + Whisper-Large-V3-Turbo      |   7.87B    |  31.89   |  21.08   |   62.19   | 31.89  |
|  LLaVA-NeXT-Vicuna-7B + Whisper-Large-V3-Turbo  |   8.38B    |  33.56   |  25.69   |   53.81   | 33.56  |
| LLaVA-NeXT-Mistral-7B + Whisper-Large-V3-Turbo  |   8.38B    |  41.00   |  38.25   |   55.63   | 41.00  |
|      InternVL2-8B + Whisper-Large-V3-Turbo      |   8.89B    |  78.00   |  77.90   |   78.24   | 78.00  |
|      MiniCPM-v2.6 + Whisper-Large-V3-Turbo      |   8.91B    |  75.78   |  75.76   |   75.88   | 75.78  |
|  Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |   9.10B    |  79.11   |  79.16   |   79.97   | 79.11  |
|      Idefics2-8B + Whisper-Large-V3-Turbo       |   9.21B    |  41.22   |  41.08   |   42.05   | 41.22  |
|   MiniCPM-Llama3-2.5 + Whisper-Large-V3-Turbo   |   9.35B    |  63.89   |  63.85   |   64.40   | 63.89  |
|     Paligemma2-10B + Whisper-Large-V3-Turbo     |   10.47B   |  24.33   |  24.72   |   25.63   | 24.33  |
|  Llama-3.2-11B-Vision-Instruct + Whisper-Small  |   10.94B   |  54.33   |  53.78   |   61.12   | 54.33  |
</details>

#### Performance Comparison Across Decoding Strategies
<details>
  <summary>Show/Hide Results Table</summary>

|                     Model                     |        Decoding Strategy         | Model Size | Accuracy  | F1 Score  | Precision |  Recall   |
|:---------------------------------------------:|:--------------------------------:|:----------:|:---------:|:---------:|:---------:|:---------:|
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |          Greedy Search           |   9.10B    |   79.11   |   79.16   |   79.97   |   79.11   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |        Greedy Search + STCM      |   9.10B    |   79.33   |   79.34   |   80.13   |   79.33   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |        Contrastive Search        |   9.10B    |   79.11   |   79.16   |   79.97   |   79.11   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |       Multinomial Sampling       |   9.10B    |   79.11   |   79.16   |   79.97   |   79.11   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |           Beam Search            |   9.10B    |   61.67   |   61.59   |   64.35   |   61.67   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo | Beam Search Multinomial Sampling |   9.10B    |   61.67   |   61.59   |   64.35   |   61.67   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |       Diverse Beam Search        |   9.10B    |   61.67   |   61.59   |   64.35   |   61.67   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |         Self-Speculative         |   9.10B    |   79.67   |   79.69   |   80.48   |   79.67   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |           DoLa (High)            |   9.10B    |   58.11   |   58.18   |   68.29   |   58.11   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            DoLa (Low)            |   9.10B    |   79.89   |   79.85   |   80.91   |   79.89   |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |        DoLa (Low) + STCM         |   9.10B    | **80.33** | **80.31** | **81.24** | **80.33** |
</details>

### ✨ Selective Token Constraint Mechanism (STCM)

#### Performance Comparison with and without STCM
<details>
  <summary>Show/Hide Results Table</summary>

|    Method    |                     Model                      | STCM | Model Size | Accuracy | F1 Score | Precision | Recall |
|:------------:|:----------------------------------------------:|:----:|:----------:|:--------:|:--------:|:---------:|:------:|
|     VLM      |              Qwen2-VL-7B-Instruct              |      |   8.29B    |  43.89   |  51.76   |   67.86   | 43.89  |
|     ALM      |            Qwen2-Audio-7B-Instruct             |      |   8.40B    |  30.22   |  28.13   |   42.83   | 30.22  |
| VLM + Random |              Qwen2-VL-7B-Instruct              |      |   8.29B    |  49.11   |  46.47   |   56.41   | 49.11  |
| ALM + Random |            Qwen2-Audio-7B-Instruct             |      |   8.40B    |  35.22   |  31.03   |   35.59   | 35.22  |
|     VLM      |              Qwen2-VL-7B-Instruct              |  ✅  |   8.29B    |  53.78   |  52.01   |   59.80   | 53.78  |
|     ALM      |            Qwen2-Audio-7B-Instruct             |  ✅  |   8.40B    |  36.67   |  32.32   |   37.77   | 36.67  |
|  VLM + ALM   | Qwen2-VL-7B-Instruct + Qwen2-Audio-7B-Instruct |      |   16.69B   |  52.33   |  57.84   |   66.95   | 52.33  |
|  VLM + ASR   | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |      |   9.10B    |  79.11   |  79.16   |   79.97   | 79.11  |
|  VLM + ASR   | Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo  |  ✅  |   9.10B    |  79.33   |  79.34   |   80.13   | 79.33  |

</details>

#### Performance Comparison with and without STCM Across Models
<details>
  <summary>Show/Hide Results Table</summary>

|                     Model                     | STCM | Model Size | Accuracy | F1 Score | Precision | Recall |
|:---------------------------------------------:|:----:|:----------:|:--------:|:--------:|:---------:|:------:|
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |      |   10.47B   |  24.33   |  24.72   |   25.63   | 24.33  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |  ✅  |   10.47B   |  27.89   |  12.51   |   25.08   | 27.89  |
|     Idefics2-8B + Whisper-Large-V3-Turbo      |      |   9.21B    |  41.22   |  41.08   |   42.05   | 41.22  |
|     Idefics2-8B + Whisper-Large-V3-Turbo      |  ✅  |   9.21B    |  42.33   |  42.50   |   43.26   | 42.33  |
| Llama-3.2-11B-Vision-Instruct + Whisper-Small |      |   10.94B   |  54.33   |  53.78   |   61.12   | 54.33  |
| Llama-3.2-11B-Vision-Instruct + Whisper-Small |  ✅  |   10.94B   |  55.44   |  55.08   |   64.79   | 55.44  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |      |   9.10B    |  79.11   |  79.16   |   79.97   | 79.11  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |  ✅  |   9.10B    |  79.33   |  79.34   |   80.13   | 79.33  |

</details>

#### Performance Comparison Across Probability Stacking Counts
<details>
  <summary>Show/Hide Results Table</summary>

|                     Model                     | Probability Stack Count | Model Size | Accuracy | F1 Score | Precision | Recall |
|:---------------------------------------------:|:-----------------------:|:----------:|:--------:|:--------:|:---------:|:------:|
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            1            |   9.10B    |  79.33   |  79.34   |   80.13   | 79.33  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            2            |   9.10B    |  62.78   |  62.85   |   71.18   | 62.78  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            3            |   9.10B    |  55.44   |  52.98   |   63.18   | 55.44  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            4            |   9.10B    |  51.22   |  47.61   |   56.00   | 51.22  |
| Qwen2-VL-7B-Instruct + Whisper-Large-V3-Turbo |            5            |   9.10B    |  49.33   |  45.24   |   59.47   | 49.33  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |            1            |   10.47B   |  27.89   |  12.51   |   25.08   | 27.89  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |            2            |   10.47B   |  29.22   |  16.26   |   20.00   | 29.22  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |            3            |   10.47B   |  29.44   |  15.50   |   35.81   | 29.44  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |            4            |   10.47B   |  28.56   |  15.49   |   45.31   | 28.56  |
|    Paligemma2-10B + Whisper-Large-V3-Turbo    |            5            |   10.47B   |  27.44   |  20.39   |   46.16   | 27.44  |

</details>

#### Performance Comparison with and without STCM Across Benchmarks
<details>
  <summary>Show/Hide Results Table</summary>

|        Model         | Benchmark | STCM | Model Size | Accuracy | F1 Score | Precision | Recall |
|:--------------------:|:---------:|:----:|:----------:|:--------:|:--------:|:---------:|:------:|
| Qwen2-VL-7B-Instruct | CII-Bench |      |   8.29B    |  48.89   |  48.19   |   53.77   | 48.89  |
| Qwen2-VL-7B-Instruct | CII-Bench |  ✅  |   8.29B    |  50.59   |  50.68   |   53.74   | 50.59  |
| Qwen2-VL-7B-Instruct |  MMStar   |      |   8.29B    |  28.20   |  28.24   |   28.61   | 28.20  |
| Qwen2-VL-7B-Instruct |  MMStar   |  ✅  |   8.29B    |  54.87   |  55.05   |   56.11   | 54.87  |
| Qwen2-VL-7B-Instruct |  MMBench  |      |   8.29B    |  54.51   |  54.30   |   54.90   | 54.51  |
| Qwen2-VL-7B-Instruct |  MMBench  |  ✅  |   8.29B    |  66.72   |  66.58   |   68.08   | 66.72  |
|    Paligemma2-10B    | CII-Bench |      |   10.7B    |  19.35   |  13.14   |   17.27   | 19.35  |
|    Paligemma2-10B    | CII-Bench |  ✅  |   10.7B    |  19.35   |  16.19   |   24.75   | 19.35  |
|    Paligemma2-10B    |  MMStar   |      |   10.7B    |  26.67   |  26.84   |   27.16   | 26.67  |
|    Paligemma2-10B    |  MMStar   |  ✅  |   10.7B    |  32.27   |  29.34   |   31.65   | 32.27  |
|    Paligemma2-10B    |  MMBench  |      |   10.7B    |   27.3   |  27.04   |   27.48   |  27.3  |
|    Paligemma2-10B    |  MMBench  |  ✅  |   10.7B    |  28.24   |  20.81   |   38.91   | 28.24  |

</details>

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