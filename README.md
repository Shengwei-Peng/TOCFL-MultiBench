# TOCFL-MultiBench: A Multimodal Benchmark for Evaluating Chinese Language Proficiency

## 💻 Installation

### Prerequisites

- Python 3.9+
- CUDA (for GPU acceleration)

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

## 📊 Results

| Model                                  | Dataset          | Tensor type | Model size | Accuracy |
| ---------------------------------------| ---------------- | ----------- | ----------:| --------:|
| LLaVA-v1.5-7B                          | CII-Bench        | FP16        |      7.06B |   20.78% |
| LLaVA-NeXT-7B                          | CII-Bench        | FP16        |      7.57B |   27.97% |
| Qwen2-VL-7B                            | CII-Bench        | BF16        |      8.29B |   41.83% |
| Qwen2-VL-2B                            | CII-Bench        | BF16        |      2.21B |   29.54% |
| Llama 3.2-Vision-11B                   | CII-Bench        | BF16        |     10.67B |   30.72% |
| LLaVA-v1.5-7B + Whisper-Large-V3-Turbo | TOCFL-MultiBench | FP16        |      7.87B |   22.11% |
| LLaVA-NeXT-7B + Whisper-Large-V3-Turbo | TOCFL-MultiBench | FP16        |      8.38B |   33.56% |
| Qwen2-VL-7B + Whisper-Large-V3-Turbo   | TOCFL-MultiBench | BF16        |      9.10B |   70.67% |
| Qwen2-VL-2B + Whisper-Large-V3-Turbo   | TOCFL-MultiBench | BF16        |      3.02B |   40.67% |
| Llama 3.2-Vision-11B + Whisper-Small   | TOCFL-MultiBench | BF16        |     10.91B |   40.78% |

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