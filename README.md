# Chinese-Multimodal-Hallucination-Mitigation

## 📑 Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Results](#Results)
- [Acknowledgements](#acknowledgements)
- [Contributing](#Contributing)
- [Contact](#contact)

## 💻 Installation

### Prerequisites

- Python 3.9+
- CUDA (for GPU acceleration)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Shengwei-Peng/Classical-Chinese-Translation.git
   ```

2. Navigate to the project directory:
    ```sh
    cd Chinese-Multimodal-Hallucination-Mitigation
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🛠️ Usage

### 1. Data Collection with `collector.py` 🗂️

The `collector.py` script provides a manual data collection interface that automatically organizes the collected data into a structured directory format. It saves images in the `dataset/images` directory, audio files in the `dataset/audio` directory, and metadata in a `dataset.json` file.

#### Output Structure
```plaintext
dataset/
│
├── audio/           # Audio files
├── images/          # Image files
└── dataset.json     # Metadata and labels for collected data
```

#### Running `collector.py`

```bash
python collector.py
```
### 2. Evaluation and Testing with `main.py` 🧪

The `main.py` script serves as the primary interface for evaluating and testing the collected datasets and pre-trained models.

```bash
python main.py
```

## 📊 Results

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

## 📧 Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw