"""main"""
import gradio as gr
from dotenv import load_dotenv

from src import MultimodalSystem


def main() -> None:
    """main"""
    model_options = [
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2-VL-7B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ]
    dataset_options = [
        "m-a-p/CII-Bench",
        "Lin-Chen/MMStar",
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
                load_output = gr.Dataframe(label="Load")
                load_button.click(
                    fn=system.load,
                    inputs=[model_dropdown, dataset_dropdown],
                    outputs=load_output
                )
                evaluate_button = gr.Button("Evaluate")
                evaluate_output = gr.Label(label="Accuracy")
                evaluate_button.click(
                    fn=system.evaluate,
                    inputs=[],
                    outputs=evaluate_output,
                )

            with gr.Column():
                audio = gr.Audio(label="Audio", type="filepath")
                image = gr.Image(label="Image")
                text = gr.Textbox(label="Text Input", lines=2)
                generate_button = gr.Button("Generate")
                generation = gr.Textbox(label="Generation")
                generate_button.click(
                    fn=system.generate,
                    inputs=[text, image, audio],
                    outputs=generation,
                )

    interface.launch(share=True)

if __name__ == "__main__":
    load_dotenv()
    main()
