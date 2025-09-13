Example model built with Meta LLaMA 3

⚠️🚨☢️  DANGER ☢️ 🚨⚠️  

META'S LLAMA 3 HAVE A RESTRICTIVE LICENSE ⚠️.  

IF YOU USE OR REDISTRIBUTE LLAMA 3 WITHOUT READING AND UNDERSTANDING THE LICENSE,  
YOU MAY BE IN VIOLATION OF META'S LEGAL TERMS. ☠️☠️☠️  

💀 DO NOT USE LLAMA 3 UNLESS YOU HAVE:  
  1. READ THE LICENSE IN FULL  
  2. UNDERSTOOD EVERY CONDITION  
  3. ACCEPTED THE LEGAL RISKS  

READ THE LICENSE FIRST:  
👉 https://llama.meta.com/llama-downloads

🚫 IF YOU IGNORE THIS WARNING, YOU ARE ON YOUR OWN. 🚫  

IF YOU ARE LIKE ME AND YOU ARE NOT SURE ABOUT WHAT YOU ARE DOING ,  
JUST COPY AND PASTE THE LICENSE EVERYWHERE AND NAME EVERYTHING
LLAMA3.


Before trying to run the C++ server try to run the python llama3 example of text completion.

Donwload llama3 1B from official llama3 website and put the following files in data folder:
* checklist.chk
* consolidated.00.pth
* params.json
* tokenizer.model

Make sure you have all dependencies installed, I removed fairscale so it will be easier for you to try 1B on a single GPU, however
you still will need to manually install:

* Rust compiler and Cargo
* TikToken
* Torch
* Fire

All these dependencies are for runing the original llama3 model not the C++ server.