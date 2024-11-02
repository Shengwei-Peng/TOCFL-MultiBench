"""main"""
import gradio as gr
from dotenv import load_dotenv

from src import MultimodalSystem


def main() -> None:
    """main"""
    model_options = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2-VL-7B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ]
    dataset_options = [
        "m-a-p/CII-Bench",
        "BAAI/CMMU",
    ]
    system = MultimodalSystem(model_options[0], dataset_options[0])

    interface = gr.Blocks()
    with interface:
        gr.Markdown("# 🧠 Chinese Multimodal Hallucination Mitigation")
        with gr.Row():
            with gr.Column():
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_options,
                    label="Select Dataset",
                    value=dataset_options[0],
                )
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="Select Model",
                    value=model_options[0]
                )
                load_button = gr.Button("Load")
                model_structure = gr.Textbox(label="Model Structure", interactive=False)
                dataset_structure = gr.Textbox(label="Dataset Structure", interactive=False)

                load_button.click(
                    fn=system.load,
                    inputs=[model_dropdown, dataset_dropdown],
                    outputs=[dataset_structure, model_structure],
                )

            with gr.Column():
                image = gr.Image(label="Image")
                text = gr.Textbox(label="Text Input", lines=2)
                generate_button = gr.Button("Generate")
                output_text = gr.Textbox(label="Generated Response")

                generate_button.click(
                    fn=system.generate,
                    inputs=[text, image],
                    outputs=output_text,
                )
    interface.launch(share=True)

if __name__ == "__main__":
    load_dotenv()
    main()
