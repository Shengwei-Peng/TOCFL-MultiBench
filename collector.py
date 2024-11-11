"""collector"""
import json
from pathlib import Path

import gradio as gr

base_dir = Path("dataset")
json_file_path = base_dir / "dataset.json"
images_dir = base_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

def collect(
    image: str, question: str, answer: str,
    option1: str, option2: str, option3: str, option4: str
) -> list:
    """collect"""
    if not image:
        return []

    if not json_file_path.exists():
        json_file_path.write_text(json.dumps([]))

    records = json.loads(json_file_path.read_text())

    image_id = len(records) + 1
    image_path = images_dir / f"{image_id}.png"
    image_path.write_bytes(Path(image).read_bytes())

    data = {
        "id": image_id,
        "image_path": str(image_path),
        "question": question,
        "answer": answer,
        "option1": option1,
        "option2": option2,
        "option3": option3,
        "option4": option4
    }

    records.append(data)

    json_file_path.write_text(json.dumps(records, indent=4, ensure_ascii=False))

    return convert_to_list_format(records)

def convert_to_list_format(records: list) -> list:
    """convert_to_list_format"""
    return [
        [
            rec["id"], rec["image_path"], rec["question"], rec["answer"],
            rec["option1"], rec["option2"], rec["option3"], rec["option4"]
        ]
        for rec in records
    ]

def clear_inputs()-> None:
    """clear_inputs"""
    return None, "", "", "", "", "", ""

def delete_last_entry()-> None:
    """delete_last_entry"""
    records = json.loads(json_file_path.read_text())
    if records:
        last_record = records.pop()
        image_path = Path(last_record["image_path"])
        if image_path.exists():
            image_path.unlink()
        json_file_path.write_text(json.dumps(records, indent=4, ensure_ascii=False))
    return convert_to_list_format(records)

def load_existing_data():
    """load_existing_data"""
    if json_file_path.exists():
        with open(json_file_path, "r", encoding="utf-8") as file:
            records = json.load(file)
        return convert_to_list_format(records)
    else:
        return []

def main() -> None:
    """main"""
    existing_data = load_existing_data()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="filepath")
                question_input = gr.Textbox(label="Enter Question")
                answer_input = gr.Textbox(label="Enter Answer")
                option1_input = gr.Textbox(label="Option 1")
                option2_input = gr.Textbox(label="Option 2")
                option3_input = gr.Textbox(label="Option 3")
                option4_input = gr.Textbox(label="Option 4")
                submit_button = gr.Button("Submit")
                clear_button = gr.Button("Clear")
                delete_last_button = gr.Button("Delete Last Entry")

            outputs = gr.Dataframe(
                headers=[
                    "ID", "Image Path", "Question", "Answer",
                    "Option 1", "Option 2", "Option 3", "Option 4"
                ],
                value=existing_data
            )

        submit_button.click(
            fn=collect,
            inputs=[
                image_input, question_input, answer_input,
                option1_input, option2_input, option3_input, option4_input
            ],
            outputs=outputs
        )
        clear_button.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                image_input, question_input, answer_input,
                option1_input, option2_input, option3_input, option4_input
            ]
        )
        delete_last_button.click(
            fn=delete_last_entry,
            inputs=[],
            outputs=outputs
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
