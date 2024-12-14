"""experiment"""
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.system import MultimodalSystem

def main():
    """main"""
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--asr_model_name_or_path", type=str, default=None)
    parser.add_argument("--prompt_template_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument(
        "--tensor_type", type=str, default="auto",
        choices= ["auto", "fp16", "bf16", "int8", "fp4", "nf4"]
    )
    parser.add_argument(
        "--decoding_strategy", type=str, default="greedy",
        choices= [
            "greedy", "contrastive", "sampling", "beam_search", "beam_search_sampling", 
            "diverse_beam_search", "self_speculative", "dola_high", "dola_low"
        ]
    )
    parser.add_argument("--use_stcm",  action="store_true")
    args = parser.parse_args()

    system = MultimodalSystem(
        model_name_or_path=args.model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        asr_model_name_or_path=args.asr_model_name_or_path,
        prompt_template_path=args.prompt_template_path,
        tensor_type=args.tensor_type,
    )
    system.evaluate(
        max_new_tokens=args.max_new_tokens,
        decoding_strategy=args.decoding_strategy,
        use_stcm=args.use_stcm,
    )

if __name__ == "__main__":
    load_dotenv()
    main()
