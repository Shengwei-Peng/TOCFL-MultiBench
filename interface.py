"""interface"""
import argparse
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from src import MultimodalSystem


def parse_args() -> argparse.Namespace:
    """parse_args"""
    parser = argparse.ArgumentParser(description="Interface")
    parser.add_argument(
        "--dataset_name_or_path", 
        type=str,
        default="TOCFL-MultiBench/TOCFL-MultiBench.json",
        help=(
            "Path to the dataset or dataset name to be evaluated. "
            "Defaults to 'TOCFL-MultiBench/TOCFL-MultiBench.json'."
        )
    )
    parser.add_argument(
        "--prompt_dir", 
        type=str,
        default="prompt",
        help="Directory containing prompt templates for evaluation. Defaults to 'prompt'."
    )
    return parser.parse_args()

def main() -> None:
    """main"""
    load_dotenv()

    args = parse_args()

    model_options = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "HuggingFaceM4/idefics2-8b",
        "google/paligemma2-10b-pt-224",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ]
    asr_model_options = [
        None,
        "openai/whisper-large-v3-turbo",
        "openai/whisper-small",
        "openai/whisper-large-v3",
    ]
    dataset_options = [
        "m-a-p/CII-Bench",
        "Lin-Chen/MMStar",
        "lmms-lab/MMBench",
    ]
    dataset_options.append(args.dataset_name_or_path)

    decoding_options = [
        "greedy", "contrastive", "sampling", "beam_search", "beam_search_sampling", 
        "diverse_beam_search", "self_speculative", "dola_high", "dola_low",
    ]
    tensor_type_options = ["auto", "fp16", "bf16", "int8", "fp4", "nf4"]

    prompt_options = [None]
    if args.prompt_dir:
        prompt_path = Path(args.prompt_dir)
        if prompt_path.is_dir():
            prompt_options.extend([str(path) for path in prompt_path.glob("*.txt")])

    system = MultimodalSystem(
        model_name_or_path=model_options[0],
        dataset_name_or_path=dataset_options[-1],
        asr_model_name_or_path=asr_model_options[0],
        prompt_template_path=prompt_options[0],
        tensor_type=tensor_type_options[0]
    )

    interface = gr.Blocks()
    with interface:
        gr.Markdown(
            "# TOCFL-MultiBench: A Multimodal Benchmark for Evaluating Chinese Language Proficiency"
        )

        model_dropdown = gr.Dropdown(
            choices=model_options,
            label="Select Model",
            value=model_options[0]
        )
        dataset_dropdown = gr.Dropdown(
            choices=dataset_options,
            label="Select Dataset",
            value=dataset_options[-1],
        )
        prompt_dropdown = gr.Dropdown(
            choices=prompt_options,
            label="Select Prompt",
            value=prompt_options[0],
        )
        asr_model_dropdown = gr.Dropdown(
            choices=asr_model_options,
            label="Select ASR Model",
            value=asr_model_options[0]
        )
        tensor_type_dropdown = gr.Dropdown(
            choices=tensor_type_options,
            label="Select Tensor Type",
            value=tensor_type_options[0]
        )

        load_button = gr.Button("Load")
        load_output = gr.Dataframe(label="Load")
        load_button.click(
            fn=system.load,
            inputs=[
                model_dropdown,
                dataset_dropdown,
                asr_model_dropdown,
                prompt_dropdown,
                tensor_type_dropdown
            ],
            outputs=load_output
        )

        with gr.Row():
            max_new_tokens_slider = gr.Slider(
                label="Max New Tokens", minimum=1, maximum=512, value=1, step=1
            )
            decoding_strategy_dropdown = gr.Dropdown(
                choices=decoding_options,
                label="Decoding Strategy",
                value="greedy",
            )
            use_stcm_checkbox = gr.Checkbox(
                label="Use STCM", value=False
            )

        evaluate_button = gr.Button("Evaluate")
        evaluate_output = gr.Dataframe(label="Evaluate")
        evaluate_button.click(
            fn=system.evaluate,
            inputs=[
                max_new_tokens_slider,
                decoding_strategy_dropdown,
                use_stcm_checkbox
            ],
            outputs=evaluate_output,
        )

    interface.launch(share=True)

if __name__ == "__main__":
    main()
