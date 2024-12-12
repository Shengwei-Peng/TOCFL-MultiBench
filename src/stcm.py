"""Selective Token Constraint Mechanism (STCM)"""
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast


class STCM(LogitsProcessor):
    """Selective Token Constraint Mechanism (STCM)"""
    def __init__(
        self,
        allowed_tokens: list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        penalty: float = None,
        temperature: float = 1.0
    ) -> None:
        super().__init__()
        if not allowed_tokens:
            raise ValueError("allowed_tokens must be a non-empty list of tokens.")
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError("A valid PreTrainedTokenizer instance must be provided.")
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")

        self.penalty = penalty
        self.temperature = temperature

        self.allowed_token_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens
                if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
            ],
            dtype=torch.long
        ).unique()

        if len(self.allowed_token_ids) == 0:
            raise ValueError("None of the allowed tokens could be converted to valid token IDs.")

        self.cumulative_scores = None
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:

        allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
        allowed_mask[self.allowed_token_ids] = True

        if self.penalty is None:
            scores[:, ~allowed_mask] = -float('inf')
        else:
            scores[:, ~allowed_mask] -= self.penalty

        if self.temperature != 1.0:
            scores = scores / self.temperature

        if self.cumulative_scores is None:
            self.cumulative_scores = scores.clone()
        else:
            self.cumulative_scores += scores

        return scores

    def generate(self) -> list[str]:
        """generate"""
        if self.cumulative_scores is None:
            raise RuntimeError("No scores have been accumulated. Ensure generation has occurred.")

        top_token_ids = torch.argmax(self.cumulative_scores, dim=-1)
        self.cumulative_scores = None

        return self.tokenizer.batch_decode(
            top_token_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
