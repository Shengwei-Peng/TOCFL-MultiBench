"""main"""
import gradio as gr

from src import MultimodalModel


def main() -> None:
    """main"""
    model_options = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]
    model = MultimodalModel(model_options[0])

    interface = gr.Interface(
        fn=model.generate,
        inputs=[
            gr.Image(),
            gr.Textbox(),
            gr.Dropdown(
                choices=model_options,
                label="Select Model",
                value=model_options[0]
            )
        ],
        outputs="text",
        title="Chinese Multimodal Hallucination Mitigation",
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()
