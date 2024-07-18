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
- microsoft/Phi-3-medium-128k-instruct
- CohereForAI/c4ai-command-r-v01
- google/gemma-2-27b-it
- Qwen/Qwen2-72B-Instruct

## LLM backends
The following LLM backends are supported:
- Llama.cpp Server API

## Telegram bot example
```python
import os

from aishalib.aishalib import Aisha
from aishalib.llmbackend import LlamaCppBackend
from aishalib.tools import parseToolResponse
from aishalib.utils import get_time_string
from aishalib.memory import SimpleMemory
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters


BOT_NAME = os.environ['BOT_NAME']
TG_TOKEN = os.environ['TG_TOKEN']

PERSISTENCE_DIR = BOT_NAME + "/"

if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)

memory = SimpleMemory(PERSISTENCE_DIR + "memory.json")


def get_aisha(aisha_context_key, tg_context):
    if aisha_context_key not in tg_context.user_data:
        backend = LlamaCppBackend("http://127.0.0.1:8088/completion", max_predict=256)
        aisha = Aisha(backend, "google/gemma-2-27b-it", prompt_file="system_prompt_example.txt", max_context=8192)
        tg_context.user_data[aisha_context_key] = aisha
    aisha = tg_context.user_data[aisha_context_key]
    aisha.load_context(aisha_context_key)
    return aisha


async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = str(update.message.from_user.id)
    user_name = memory.get_memory_value("names:" + user_id, "")
    computed_name = user_name if user_name else f"id_{user_id}"
    message = update.message.text

    aisha = get_aisha(PERSISTENCE_DIR + str(chat_id), context)
    aisha.add_user_request(f"{computed_name}: {message}", meta_info=get_time_string())
    tools_response = aisha.completion(temp=0.7, top_p=0.9)
    aisha.save_context(PERSISTENCE_DIR + str(chat_id))

    tools = parseToolResponse(tools_response, ["directly_answer", "save_human_name", "pass"])

    if "save_human_name" in tools:
        user_name = tools["save_human_name"]
        memory.save_memory_value("names:" + user_name.split(":")[0].replace("id_", ""), user_name.split(":")[1])

    if "pass" not in tools:
        await context.bot.send_message(chat_id=chat_id,
                                       text=tools["directly_answer"],
                                       reply_to_message_id=update.message.message_id)


application = Application.builder().token(TG_TOKEN).build()
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
application.run_polling()
```

## Run Llama.CPP Server backend
```console
llama.cpp/build/bin/llama-server -m model_q5_k_m.gguf -ngl 99 -fa -c 4096 --host 0.0.0.0 --port 8000
```

## Install CUDA toolkit for Llama.cpp compilation
Please note that the toolkit version must match the driver version. The driver version can be found using the nvidia-smi command.
To install toolkit for CUDA 12.5 you need to run the following commands:
```console
CUDA_TOOLKIT_VERSION=12-5
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
