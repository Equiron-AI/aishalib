import os
import json
import logging
from transformers import AutoTokenizer, AutoConfig


logger = logging.getLogger(__name__)


class Aisha:
    def __init__(self, llm_backend, base_model, max_context=4096, prompt="", prompt_file=""):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, add_bos_token=False)
        self.max_context = max_context
        self.max_predict = llm_backend.max_predict

        if prompt_file:
            with open(prompt_file) as f:
                prompt = f.read()

        config = AutoConfig.from_pretrained(base_model)

        match config.model_type:
            case "qwen2":
                self.generation_promp_template = "<|im_start|>assistant\n"
                self.user_req_template = "<|im_start|>user\n{user_req}<|im_end|>"
                self.system_injection_template = "<|im_start|>system\n{system_injection}<|im_end|>"
                self.tokens = [self.tokenizer.apply_chat_template([{"role": "system", "content": prompt}])]
                self.stop_token = "<|im_end|>"
            case "llama":
                self.generation_promp_template = "<|start_header_id|>assistant<|end_header_id|>\n"
                self.user_req_template = "<|start_header_id|>user<|end_header_id|>\n{user_req}<|eot_id|>"
                self.system_injection_template = "<|start_header_id|>system<|end_header_id|>\n{system_injection}<|eot_id|>"
                self.tokens = [self.tokenizer.apply_chat_template([{"role": "system", "content": prompt}])]
                self.stop_token = "<|eot_id|>"
            case "cohere":
                self.generation_promp_template = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                self.user_req_template = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_req}<|END_OF_TURN_TOKEN|>"
                self.system_injection_template = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_injection}<|END_OF_TURN_TOKEN|>"
                self.tokens = [self.tokenizer.apply_chat_template([{"role": "system", "content": prompt}])]
                self.stop_token = self.tokenizer.eos_token
            case "phi3":
                self.generation_promp_template = "<|assistant|>\n"
                self.user_req_template = "<|user|>\n{user_req}<|end|>\n"
                self.system_injection_template = "<|system|>\n{system_injection}<|end|>\n"
                self.tokens = [self.tokenizer(self.tokenizer.bos_token + f"<|system|>\n{prompt}<|end|>\n")["input_ids"]]
                self.stop_token = "<|end|>"
            case _:
                raise RuntimeError("Unknown model: " + config.model_type)

        self.llm_backend = llm_backend
        self.llm_backend.stop_token = self.stop_token
        self.llm_backend.base_model = base_model
        self.llm_backend.tokenizer = self.tokenizer

        self.generation_prompt_tokens = self.tokenizer(self.generation_promp_template)["input_ids"]
        logger.info("System prompt size: " + str(len(self.tokens[0])))

    def sanitize(self, text):
        return text.replace("#", "").replace("<|", "").replace("|>", "")

    def add_user_request(self, user_request, system_injection=""):
        text = self.user_req_template.replace("{user_req}", self.sanitize(user_request.strip()))
        if system_injection:
            text += self.system_injection_template.replace("{system_injection}", system_injection)
        tokens = self.tokenizer(text)["input_ids"]
        self.tokens.append(tokens)
        self._cut_context()

    def add_system_injection(self, system_injection):
        text = self.system_injection_template.replace("{system_injection}", system_injection)
        self.tokens.append(self.tokenizer(text)["input_ids"])
        self._cut_context()

    def completion(self, temp=0.0, top_p=0.5):
        request_tokens = sum(self.tokens, [])
        request_tokens += self.generation_prompt_tokens
        text_resp = self.llm_backend.completion(request_tokens, temp, top_p)
        response_tokens = self.tokenizer(text_resp.strip() + self.stop_token)["input_ids"]
        response_tokens = self.generation_prompt_tokens + response_tokens
        self.tokens.append(response_tokens)
        return text_resp

    async def stream_completion(self, callback, temp=0.0, top_p=0.5):
        request_tokens = sum(self.tokens, [])
        request_tokens += self.generation_prompt_tokens
        text_resp = await self.llm_backend.stream_completion(request_tokens, callback, temp, top_p)
        response_tokens = self.tokenizer(text_resp.strip() + self.stop_token)["input_ids"]
        response_tokens = self.generation_prompt_tokens + response_tokens
        self.tokens.append(response_tokens)
        return text_resp

    def load_context(self, file_name):
        if os.path.isfile(file_name):
            with open(file_name) as f:
                self.tokens = json.load(f)

    def save_context(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.tokens, f)

    def _cut_context(self):
        busy_tokens = len(sum(self.tokens, []))
        free_tokens = self.max_context - busy_tokens
        while free_tokens < self.max_predict:
            free_tokens += len(self.tokens[1])
            del self.tokens[1]

    def clear_context(self):
        del self.tokens[1:]
