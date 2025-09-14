# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement. 
from typing import List 
from llama3.llama3_generation import Llama
from pytannic.torch.modules import write
 

if __name__ == '__main__':
    generator = Llama.build(
        ckpt_dir = "data",
        tokenizer_path = "data/tokenizer.model",
        sequence_length_limit=512,
        batch_size_limit=512,
        device='cpu'
    )
    
    write(generator.model, 'data/llama3-model')