"""collector"""
import json
from pathlib import Path

import gradio as gr

base_dir = Path("dataset")
json_file_path = base_dir / "dataset.json"
images_dir = base_dir / "images"
audio_dir = base_dir / "audio"
images_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)

def collect(
    image: str, audio: str, question: str, answer: str,
    option1: str, option2: str, option3: str, option4: str,
    edition: str, test_type: str, level: str, part: str, sequence: int
) -> list:
    """collect"""
    if not json_file_path.exists():
        json_file_path.write_text(json.dumps([]))

    records = json.loads(json_file_path.read_text())

    if sequence is None or sequence <= 0:
        sequence = len(records) + 1

    record_id = generate_id(edition, test_type, level, part, sequence)

    image_path = None
    if image:
        image_path = images_dir / f"{record_id}.png"
        image_path.write_bytes(Path(image).read_bytes())

    audio_path = None
    if audio:
        audio_path = audio_dir / f"{record_id}.mp3"
        audio_path.write_bytes(Path(audio).read_bytes())

    data = {
        "id": record_id,
        "image_path": str(image_path) if image_path else None,
        "audio_path": str(audio_path) if audio_path else None,
        "question": question.strip(),
        "answer": answer.strip().upper(),
        "option1": option1.strip(),
        "option2": option2.strip(),
        "option3": option3.strip(),
        "option4": option4.strip()
    }

    records.append(data)

    json_file_path.write_text(json.dumps(records, indent=4, ensure_ascii=False))

    return convert_to_list_format(records)

def convert_to_list_format(records: list) -> list:
    """convert_to_list_format"""
    return [
        [
            rec["id"], rec.get("image_path"), rec.get("audio_path"),
            rec["question"], rec["answer"],
            rec["option1"], rec["option2"], rec["option3"], rec["option4"]
        ]
        for rec in records
    ]

def clear_inputs()-> None:
    """clear_inputs"""
    return None, None, "", "", "", "", "", "", None, None, None, None, None

def delete_last_entry()-> None:
    """delete_last_entry"""
    records = json.loads(json_file_path.read_text())
    if records:
        last_record = records.pop()

        image_path = last_record.get("image_path")
        if image_path:
            image_path_obj = Path(image_path)
            if image_path_obj.exists():
                image_path_obj.unlink()

        audio_path = last_record.get("audio_path")
        if audio_path:
            audio_path_obj = Path(audio_path)
            if audio_path_obj.exists():
                audio_path_obj.unlink()

        json_file_path.write_text(json.dumps(records, indent=4, ensure_ascii=False))

    return convert_to_list_format(records)

def load_existing_data():
    """load_existing_data"""
    if json_file_path.exists():
        with open(json_file_path, "r", encoding="utf-8") as file:
            records = json.load(file)
        return convert_to_list_format(records)
    return []

def generate_id(edition: str, test_type: str, level: str, part: str, sequence: int) -> str:
    """generate_id"""
    edition_mapping = {"第一輯": "01", "第二輯": "02", "第三輯": "03", "第四輯": "04", "第五輯": "05"}
    test_type_mapping = {"聽力測驗": "L", "閱讀測驗": "R"}
    level_mapping = {
        "準備級 Novice": "N", "入門基礎級 Band A": "A", "進階高階級 Band B": "B", "流利精通級 Band C": "C"
    }
    part_mapping = {"第一部分": "P1", "第二部分": "P2", "第三部分": "P3", "第四部分": "P4", "第五部分": "P5"}

    edition_code = edition_mapping.get(edition, "00")
    test_type_code = test_type_mapping.get(test_type, "X")
    level_code = level_mapping.get(level, "X")
    part_code = part_mapping.get(part, "P0")
    sequence_code = f"{sequence:03d}"

    return f"{edition_code}-{test_type_code}-{level_code}-{part_code}-{sequence_code}"

def main() -> None:
    """main"""
    existing_data = load_existing_data()

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="filepath")
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
            with gr.Column(scale=1):
                edition_input = gr.Dropdown(
                    choices=["第一輯", "第二輯", "第三輯", "第四輯", "第五輯"],
                    label="Select Edition"
                )
                test_type_input = gr.Dropdown(
                    choices=["聽力測驗", "閱讀測驗"],
                    label="Select Test Type"
                )
                level_input = gr.Dropdown(
                    choices=["準備級", "入門基礎級", "進階高階級", "流利精通級"],
                    label="Select Level"
                )
                part_input = gr.Dropdown(
                    choices=["第一部分", "第二部分", "第三部分", "第四部分", "第五部分"],
                    label="Select Part"
                )
                sequence_input = gr.Number(label="Enter Sequence Number", precision=0)
            with gr.Column(scale=1):
                question_input = gr.Textbox(label="Enter Question")
                answer_input = gr.Textbox(label="Enter Answer")
                option1_input = gr.Textbox(label="Option 1")
                option2_input = gr.Textbox(label="Option 2")
                option3_input = gr.Textbox(label="Option 3")
                option4_input = gr.Textbox(label="Option 4")

        with gr.Row():
            clear_button = gr.Button("Clear")
            delete_last_button = gr.Button("Delete Last Entry")
            submit_button = gr.Button("Submit")

        outputs = gr.Dataframe(
            headers=[
                "ID", "Image Path", "Audio Path", "Question", "Answer",
                "Option 1", "Option 2", "Option 3", "Option 4"
            ],
            value=existing_data
        )

        submit_button.click(
            fn=collect,
            inputs=[
                image_input, audio_input, question_input, answer_input,
                option1_input, option2_input, option3_input, option4_input,
                edition_input, test_type_input, level_input, part_input,
                sequence_input
            ],
            outputs=outputs
        )
        clear_button.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                image_input, audio_input, question_input, answer_input,
                option1_input, option2_input, option3_input, option4_input,
                edition_input, test_type_input, level_input, part_input,
                sequence_input
            ]
        )
        delete_last_button.click(
            fn=delete_last_entry,
            inputs=[],
            outputs=outputs
        )

    interface.launch(share=True)

if __name__ == "__main__":
    main()
