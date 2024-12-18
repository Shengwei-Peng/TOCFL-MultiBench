"""collector"""
import json
from pathlib import Path
from argparse import ArgumentParser

import gradio as gr
from pydub import AudioSegment


class DatasetManager:
    """DatasetManager"""
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.json_file_path = self.dataset_dir / "TOCFL-MultiBench.json"
        self.images_dir = self.dataset_dir / "images"
        self.audios_dir = self.dataset_dir / "audios"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.audios_dir.mkdir(parents=True, exist_ok=True)

        if not self.json_file_path.exists():
            self.json_file_path.write_text("[]", encoding="utf-8")

    def load_records(self) -> list:
        """load_records"""
        with self.json_file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save_records(self, records: list) -> None:
        """save_records"""
        self.json_file_path.write_text(
            json.dumps(records, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

    def get_image_path(self, record_id: str) -> Path:
        """get_image_path"""
        return self.images_dir / f"{record_id}.png"

    def get_audio_path(self, record_id: str) -> Path:
        """get_audio_path"""
        return self.audios_dir / f"{record_id}.mp3"

def collect(
    image: str, audio_file1: str, audio_file2: str, instruction: str, question: str,
    option1: str, option2: str, option3: str, option4: str, answer: str,
    edition: str, test_type: str, level: str, part: str, sequence: int,
    dataset_manager: DatasetManager
) -> list:
    """collect"""
    records = dataset_manager.load_records()

    if sequence is None or sequence <= 0:
        sequence = len(records) + 1

    record_id = generate_id(edition, test_type, level, part, sequence)

    image_path = None
    if image:
        image_path = dataset_manager.get_image_path(record_id)
        image_path.write_bytes(Path(image).read_bytes())

    audio_path = None
    if audio_file1 or audio_file2:
        audio_path = dataset_manager.get_audio_path(record_id)
        merge_audio_files(audio_file1, audio_file2, audio_path)

    data = {
        "id": record_id,
        "image": str(image_path) if image_path else None,
        "audio": str(audio_path) if audio_path else None,
        "instruction": instruction.strip().replace("(", "（").replace(")", "）"),
        "question": question.strip().replace("(", "（").replace(")", "）"),
        "option1": option1.strip().replace("(", "（").replace(")", "）"),
        "option2": option2.strip().replace("(", "（").replace(")", "）"),
        "option3": option3.strip().replace("(", "（").replace(")", "）"),
        "option4": option4.strip().replace("(", "（").replace(")", "）"),
        "answer": answer.strip().upper(),
    }

    records.append(data)
    dataset_manager.save_records(records)

    return convert_to_list_format(records)

def convert_to_list_format(records: list) -> list:
    """convert_to_list_format"""
    return [
        [
            rec["id"], rec["image"], rec["audio"], rec["instruction"], rec["question"],
            rec["option1"], rec["option2"], rec["option3"], rec["option4"], rec["answer"]
        ]
        for rec in records
    ]

def clear_inputs() -> tuple:
    """clear_inputs"""
    return None, None, None, "", "", "", "", "", "", None, None, None, None, None

def delete_last_entry(dataset_manager: DatasetManager) -> list:
    """delete_last_entry"""
    records = dataset_manager.load_records()
    if records:
        last_record = records.pop()

        image_path = last_record.get("image")
        if image_path:
            image_path_obj = Path(image_path)
            if image_path_obj.exists():
                image_path_obj.unlink()

        audio_path = last_record.get("audio")
        if audio_path:
            audio_path_obj = Path(audio_path)
            if audio_path_obj.exists():
                audio_path_obj.unlink()

        dataset_manager.save_records(records)

    return convert_to_list_format(records)

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

def merge_audio_files(audio_file1: str, audio_file2: str, output_path: Path) -> None:
    """merge_audio_files"""
    combined_audio = AudioSegment.empty()
    if audio_file1:
        combined_audio += AudioSegment.from_file(audio_file1)
    if audio_file2:
        combined_audio += AudioSegment.from_file(audio_file2)
    combined_audio.export(output_path, format="mp3")

def main() -> None:
    """main"""
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="TOCFL-MultiBench")
    args = parser.parse_args()

    dataset_manager = DatasetManager(args.dataset_dir)
    records = convert_to_list_format(dataset_manager.load_records())

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="filepath")
                audio_input1 = gr.Audio(label="Upload Audio File 1", type="filepath")
                audio_input2 = gr.Audio(label="Upload Audio File 2", type="filepath")

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
                    choices=["準備級 Novice", "入門基礎級 Band A", "進階高階級 Band B", "流利精通級 Band C"],
                    label="Select Level"
                )
                part_input = gr.Dropdown(
                    choices=["第一部分", "第二部分", "第三部分", "第四部分", "第五部分"],
                    label="Select Part"
                )
                sequence_input = gr.Number(label="Enter Question Number", precision=0)

            with gr.Column(scale=1):
                instruction_input = gr.Textbox(label="Instruction")
                question_input = gr.Textbox(label="Question")
                option1_input = gr.Textbox(label="Option 1")
                option2_input = gr.Textbox(label="Option 2")
                option3_input = gr.Textbox(label="Option 3")
                option4_input = gr.Textbox(label="Option 4")
                answer_input = gr.Textbox(label="Answer")

        with gr.Row():
            clear_button = gr.Button("Clear")
            delete_last_button = gr.Button("Delete Last Entry")
            submit_button = gr.Button("Submit")

        outputs = gr.Dataframe(
            headers=[
                "ID", "Image", "Audio", "Instruction", "Question",
                "Option 1", "Option 2", "Option 3", "Option 4", "Answer"
            ],
            value=records
        )

        submit_button.click(
            fn=lambda *args: collect(*args, dataset_manager=dataset_manager),
            inputs=[
                image_input, audio_input1, audio_input2, instruction_input, question_input,
                option1_input, option2_input, option3_input, option4_input, answer_input,
                edition_input, test_type_input, level_input, part_input,
                sequence_input
            ],
            outputs=outputs
        )
        clear_button.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                image_input, audio_input1, audio_input2, instruction_input, question_input,
                option1_input, option2_input, option3_input, option4_input, answer_input,
                edition_input, test_type_input, level_input, part_input,
                sequence_input
            ]
        )
        delete_last_button.click(
            fn=lambda: delete_last_entry(dataset_manager),
            inputs=[],
            outputs=outputs
        )

    interface.launch(share=True)

if __name__ == "__main__":
    main()
