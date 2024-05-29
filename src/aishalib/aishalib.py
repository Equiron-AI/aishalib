import os
import json
from transformers import AutoTokenizer, AutoConfig
from aishalib.llmbackend import LlamaCppBackend


class Aisha:
    def __init__(self, llm_backend_url, base_model, max_context=4096, max_predict=256, prompt="", prompt_file=""):
        self.llm_backend_url = llm_backend_url
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, add_bos_token=False)
        self.max_context = max_context
        self.max_predict = max_predict

        if prompt_file:
            with open(prompt_file) as f:
                prompt = f.read()

        config = AutoConfig.from_pretrained(base_model)

        if config.model_type == "cohere":
            self.generation_promp_template = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            self.user_req_template = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_req}<|END_OF_TURN_TOKEN|>"
            self.system_injection_template = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_injection}<|END_OF_TURN_TOKEN|>"
            inst = [{"role": "system", "content": prompt}]
            prompt_tokens = self.tokenizer.apply_chat_template(inst)
            stop_token = self.tokenizer.eos_token
        elif config.model_type == "phi3":
            self.generation_promp_template = "<|assistant|>"
            self.user_req_template = "<|user|>{user_req}<|end|>"
            self.system_injection_template = "<|user|>{system_injection}<|end|>"
            inst = [{"role": "user", "content": prompt}]
            prompt_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.apply_chat_template(inst)
            stop_token = "<|end|>"
        else:
            raise RuntimeError("Unknown model: " + config.model_type)

        self.llm_backend = LlamaCppBackend(llm_backend_url, stop_token, max_predict)
        self.generation_prompt_tokens = self.tokenizer(self.generation_promp_template)["input_ids"]
        self.instructions_len = len(inst)
        self.tokens = [prompt_tokens]

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
        response_tokens = self.tokenizer(text_resp.strip() + self.tokenizer.eos_token)["input_ids"]
        response_tokens = self.generation_prompt_tokens + response_tokens
        self.tokens.append(response_tokens)
        return text_resp

    def load_context(self, id, path=""):
        file_name = path + str(id) + ".context"
        if os.path.isfile(file_name):
            with open(file_name) as f:
                self.tokens = json.load(f)

    def save_context(self, id, path=""):
        file_name = path + str(id) + ".context"
        with open(file_name, "w") as f:
            json.dump(self.tokens, f)

    def _cut_context(self):
        busy_tokens = len(sum(self.tokens, []))
        free_tokens = self.max_context - busy_tokens
        while free_tokens < self.max_predict:
            free_tokens += len(self.tokens[self.instructions_len])
            del self.tokens[self.instructions_len]
