import json
import os 
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from pytannic.client import Client
from pytannic.torch.tensors import serialize, deserialize

from llama3.llama3 import ModelArgs
from llama3.llama3_tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]


from dataclasses import dataclass

@dataclass
class Model:
    params: any


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        sequence_length_limit: int,
        batch_size_limit: int,
        seed: int = 2,
        device: Optional[str] = None,
    ) -> "Llama":
        """
        Build a Llama instance for **single GPU or CPU** (no model parallelism).
        """

        assert 1 <= sequence_length_limit <= 8192, (
            f"sequence_length_limit must be between 1 and 8192, got {sequence_length_limit}."
        )
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)
 
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}" 

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            sequence_length_limit=sequence_length_limit,
            batch_size_limit=batch_size_limit,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
  
        model = Model(model_args)  
        return Llama(model, tokenizer, device)

    def __init__(self, model, tokenizer: Tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]: 
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.batch_size_limit, (bsz, params.batch_size_limit)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.sequence_length_limit
        total_len = min(params.sequence_length_limit, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)
 
        for cur_pos in range(min_prompt_len, total_len):
            resquest = serialize(tokens[:, prev_pos:cur_pos])
            logits = None
            with Client("0.0.0.0", 8080) as client: 
                client.send(resquest)
                client.send(prev_pos.to_bytes(4, byteorder='little', signed=True))
                response = client.receive()
                logits = deserialize(response)
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )

            for i, token_id in enumerate(next_token.tolist()):
                if not eos_reached[i]:
                    word = self.tokenizer.decode([token_id])
                    print(word, end="", flush=True)

            eos_reached |= (~input_text_mask[:, cur_pos]) & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        print()  # newline after generation

        if logprobs:
            token_logprobs = token_logprobs.tolist()

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]

            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass

            out_tokens.append(toks)
            out_logprobs.append(probs)

        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.sequence_length_limit - 1

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        print(prompts[0])
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )

        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]

        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.sequence_length_limit - 1

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


from typing import List 
import fire  

#run this with:
"""
python main.py \
    --ckpt_dir data \
    --tokenizer_path data/tokenizer.model \
    --temperature 0.6 \
    --top_p 0.9 \
    --sequence_length_limit 512 
""" 

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    sequence_length_limit: int = 128,
    max_gen_len: int = 64,
    batch_size_limit: int = 1,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `sequence_length_limit` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        sequence_length_limit=sequence_length_limit,
        batch_size_limit=batch_size_limit,
        device='cpu'
    )

    prompts: List[str] = [  
        """Hola como estan? Espero que la """
    ]
    
    generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

if __name__ == "__main__":
    fire.Fire(main) 