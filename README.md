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
from aishalib.aishalib import Aisha
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

TG_TOKEN = "YOUR_TG_TOKEN"

SYSTEM_PROMPT = """
Ты умный бот помощник для общения в телеграме.
Ты общаешься в групповом чате с другими пользователями.
Ты отвечаешь на русском языке.
"""

SYSTEM_INJECTION = """
Последнее сообщение написал пользователь с идентификатором {user_id}.
Используй эти идентификаторы для того, чтобы различать пользователей.
Запрещено обращаться к пользователю по его идентификатору! Можно только по имени.
Если пользователь не представился спроси как его зовут.
Если исходя из контекста и смысла беседы это сообщение адресовано тебе или это общее сообщение для всех в чате то ты обязан на него ответить.
Если это сообщение адресовано другому пользователю, то напиши специальную команду "ignoring_message" в ответе.
"""

def get_aisha(chat_id, tg_context):
    if chat_id not in tg_context.user_data:
        aisha = Aisha("http://127.0.0.1:8000/completion",
                      "CohereForAI/c4ai-command-r-v01",
                      prompt=SYSTEM_PROMPT,
                      max_context=8192,
                      max_predict=512)
        tg_context.user_data[chat_id] = aisha
    aisha = tg_context.user_data[chat_id]
    aisha.load_context(chat_id)
    return aisha

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = str(update.message.from_user.id)
    aisha = get_aisha(str(chat_id), context)
    aisha.add_user_request(update.message.text,
                           system_injection=SYSTEM_INJECTION.replace("{user_id}", user_id))
    text_response = aisha.completion(temp=0.0, top_p=0.5)
    aisha.save_context(chat_id)
    if "ignoring_message" not in text_response:
        await context.bot.send_message(chat_id=chat_id,
                                       text=text_response,
                                       reply_to_message_id=update.message.message_id)

application = Application.builder().token(TG_TOKEN).build()
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
application.run_polling()
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

## Document search example
```python
from aishalib.aishalib import Aisha

MODEL_ID = "microsoft/Phi-3-medium-128k-instruct"
COMPLETION_URL = "http://172.17.0.1:8088/completion"

with open("documents.txt") as f:
    docs = f.read()

system_prompt = f"""## Ты - поисковая система.
Ниже находятся документы по которым необходимо выполнять поиск.
## Документы:
{docs}
## Ответь на вопрос пользователя используя эти документы: """

aisha = Aisha(COMPLETION_URL, MODEL_ID, prompt=system_prompt, max_context=32768, max_predict=1024)
aisha.add_user_request("Что такое ...?")
print(aisha.completion(temp=0.0, top_p=0.0))
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

