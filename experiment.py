"""experiment"""
import argparse

from dotenv import load_dotenv

from src.system import MultimodalSystem


def parse_args() -> argparse.Namespace:
    """parse_args"""
    parser = argparse.ArgumentParser(description="Experiment")

    parser.add_argument(
        "--model_name_or_path", 
        type=str,
        required=True,
        help="Path to the model or model name to be used for evaluation."
    )
    parser.add_argument(
        "--dataset_name_or_path", 
        type=str,
        required=True,
        help="Path to the dataset or dataset name to be evaluated."
    )
    parser.add_argument(
        "--asr_model_name_or_path", 
        type=str,
        default=None,
        help="Path to the Automatic Speech Recognition (ASR) model or its name. Defaults to None."
    )
    parser.add_argument(
        "--prompt_template_path", 
        type=str,
        default=None,
        help="Path to a prompt template file. Defaults to None."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int,
        default=1,
        help="Maximum number of tokens to generate for each prediction. Defaults to 1."
    )
    parser.add_argument(
        "--tensor_type", 
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "int8", "fp4", "nf4"],
        help=(
            "Specifies the tensor type for computations. "
            "Options: auto, fp16, bf16, int8, fp4, nf4. Defaults to auto."
        )
    )
    parser.add_argument(
        "--decoding_strategy", 
        type=str,
        default="greedy",
        choices=[
            "greedy", "contrastive", "sampling", "beam_search", "beam_search_sampling",
            "diverse_beam_search", "self_speculative", "dola_high", "dola_low"
        ],
        help=(
            "Strategy for decoding model outputs. "
            "Options: greedy, contrastive, sampling, beam_search, beam_search_sampling, "
            "diverse_beam_search, self_speculative, dola_high, dola_low. Defaults to greedy."
        )
    )
    parser.add_argument(
        "--use_stcm",
        action="store_true",
        help=(
            "Flag to enable the use of Selective Token Constraint Mechanism (STCM). "
            "Set this flag to activate."
        )
    )

    return parser.parse_args()

def main() -> None:
    """main"""
    load_dotenv()

    args = parse_args()

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
    main()
