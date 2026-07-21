"""HuggingFace decoder wrapper: activation capture and behavior probing.

Builds on the reference implementation's HFLensModel (jlens.from_hf) so that
layer indexing, encoding, and unembedding are identical to the lens's own:
activation at layer l is the *output* of decoder block l, and
unembed(h) = lm_head(final_norm(h)).

Requires the [models] extra plus the jlens package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


@dataclass
class CaptureResult:
    """One forward pass with residual streams kept.

    activations[l] is the residual stream output of decoder block l,
    shape (seq, d_model) — same indexing as jlens source layers.
    """

    tokens: list[str]
    activations: dict[int, torch.Tensor]
    logits: torch.Tensor  # (seq, vocab)


class ModelWrapper:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        hf.to(device)
        hf.eval()
        self.hf = hf
        self.lens_model = jlens.from_hf(hf, self.tokenizer)
        self.n_layers = self.lens_model.n_layers
        self.d_model = self.lens_model.d_model
        self.vocab_size = hf.config.vocab_size

    def encode(self, text: str, max_tokens: int = 512) -> torch.Tensor:
        return self.lens_model.encode(text, max_length=max_tokens)

    @torch.no_grad()
    def capture(
        self, text: str, layers: list[int] | None = None, max_tokens: int = 512
    ) -> CaptureResult:
        """Run the model, recording residual streams at the given layers."""
        layers = list(range(self.n_layers)) if layers is None else layers
        ids = self.encode(text, max_tokens)
        # The recorder hooks the decoder blocks themselves, so entering
        # through the full HF model (which adds the LM head) records the
        # same activations lens_model.forward would, plus gives us logits.
        with jlens.ActivationRecorder(self.lens_model.layers, at=layers) as rec:
            out = self.hf(ids, use_cache=False)
            activations = {l: rec.activations[l][0].detach() for l in layers}
        return CaptureResult(
            tokens=self.tokenizer.convert_ids_to_tokens(ids[0]),
            activations=activations,
            logits=out.logits[0].float(),
        )

    # -- behavior probing ---------------------------------------------------

    @torch.no_grad()
    def answer_logprobs(self, prompt: str, answers: list[str]) -> dict[str, float]:
        """Log P(answer | prompt) summed over the answer's tokens.

        Answers are scored as continuations " {answer}" of the prompt.
        """
        scores: dict[str, float] = {}
        for ans in answers:
            prompt_ids = self.encode(prompt)
            full_ids = self.encode(prompt + " " + ans)
            n_prompt = prompt_ids.shape[1]
            if full_ids.shape[1] <= n_prompt:
                raise ValueError(f"answer {ans!r} adds no tokens")
            logits = self.hf(full_ids, use_cache=False).logits[0]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            total = 0.0
            for pos in range(n_prompt, full_ids.shape[1]):
                total += float(logprobs[pos - 1, full_ids[0, pos]])
            scores[ans] = total
        return scores

    def answer_distribution(self, prompt: str, answers: list[str]) -> np.ndarray:
        """P(answer | prompt) normalized over the candidate set."""
        lp = self.answer_logprobs(prompt, answers)
        arr = np.array([lp[a] for a in answers], dtype=np.float64)
        arr -= arr.max()
        p = np.exp(arr)
        return p / p.sum()
