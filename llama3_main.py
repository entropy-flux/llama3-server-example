# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List

import fire 
from llama3.llama3_generation import Llama

#run this with:
"""
python llama3_main.py \
    --ckpt_dir data \
    --tokenizer_path data/tokenizer.model \
    --temperature 0.6 \
    --top_p 0.9 \
    --sequence_length_limit 512 
""" 

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
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
    )

    prompts: List[str] = [  
        """ Saint Seiya is better than Dragon Ball because """
    ]
    
    generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

if __name__ == "__main__":
    fire.Fire(main) 