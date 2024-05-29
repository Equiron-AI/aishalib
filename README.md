# AISHA Lib: A High-Level Abstraction for Building AI Assistants
In the evolving landscape of artificial intelligence, the development of smart assistants has become increasingly prevalent. To streamline this process, the **AISHA (AI Smart Human Assistant) Lib** offers a high-level abstraction designed for creating AI assistants. This versatile library supports various large language models (LLMs) and different LLM backends, providing developers with a powerful and flexible toolset.

## Environment
To create a Python virtual environment, use the command:
```console
conda env create -f environment.yml
``` 

## Installation
```console
pip install aishalib
``` 

## Supported Models
The following LLM models are supported:
- Phi-3-medium-128k-instruct
- c4ai-command-r-v01

## LLM backends
The following LLM backends are supported:
- Llama.cpp Server API

## Telegram bot example
```python

```

## Chainlit example
```python
import chainlit as cl
from aishalib.aishalib import Aisha

@cl.on_chat_start
async def on_chat_start():
    aisha = Aisha("http://127.0.0.1:8000/completion",
                  "CohereForAI/c4ai-command-r-v01",
                  prompt="Ты отвечаешь на русском языке.",
                  max_context=4096,
                  max_predict=512)
    cl.user_session.set("aisha", aisha)

@cl.on_message
async def on_message(input_msg: cl.Message):
    output_msg = cl.Message(content="")
    await output_msg.send()
    aisha = cl.user_session.get("aisha")
    aisha.add_user_request(input_msg.content)
    response = await cl.make_async(aisha.completion)(temp=0.5, top_p=0.5)
    output_msg.content = response
    await output_msg.update()
```

## Run Llama.CPP Server backend
```console
llama.cpp/build/bin/server -m model_q5_k_m.gguf -ngl 99 -fa -c 4096 --host 0.0.0.0 --port 8000
```

## Install CUDA toolkit for Llama.cpp compilation
Please note that the toolkit version must match the driver version. The driver version can be found using the nvidia-smi command.
Аor example, to install toolkit for CUDA 12.2 you need to run the following commands:
```console
CUDA_TOOLKIT_VERSION=12-2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-${CUDA_TOOLKIT_VERSION}
echo -e '
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
' >> ~/.bashrc
```

