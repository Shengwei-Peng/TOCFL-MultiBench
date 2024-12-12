"""experiment"""
import argparse

from dotenv import load_dotenv

from src.utils import MultimodalSystem

def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--asr_model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--tensor_type", type=str, default="auto")
    parser.add_argument("--decoding_strategy", type=str, default="greedy")
    parser.add_argument("--use_stcm",  action="store_true")
    args = parser.parse_args()

    system = MultimodalSystem(
        model_name_or_path=args.model_name_or_path,
        asr_model_name_or_path=args.asr_model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        tensor_type=args.tensor_type
    )
    accuracy = system.evaluate(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        decoding_strategy=args.decoding_strategy,
        use_stcm=args.use_stcm,
    )
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    load_dotenv()
    main()
