import requests
import sseclient
import json


class LlamaCppBackend:
    def __init__(self, url, max_predict=1024):
        self.url = url
        self.max_predict = max_predict

    def get_request_object(self, request_tokens, stream, temp, top_p):
        return {"prompt": request_tokens,
                "stream": stream,
                "n_predict": self.max_predict,
                "temperature": temp,
                "repeat_last_n": 0,
                "repeat_penalty": 1.0,
                "top_k": -1,
                "top_p": top_p,
                "min_p": 0,
                "tfs_z": 1,
                "typical_p": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "stop": [self.stop_token],
                "cache_prompt": True}

    def completion(self, request_tokens, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, False, temp, top_p)
        response = requests.post(self.url, json=request)
        response.raise_for_status()
        return response.json()["content"]

    async def stream_completion(self, request_tokens, callback, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, True, temp, top_p)
        response = requests.post(self.url, json=request, stream=True, headers={'Accept': 'text/event-stream'})
        response.raise_for_status()
        stream = sseclient.SSEClient(response).events()
        text_resp = ""
        for event in stream:
            parsed_event = json.loads(event.data)
            if parsed_event["stop"]:
                break
            content = parsed_event["content"]
            text_resp += content
            await callback(content)
        return text_resp


class TogetherAiBackend:
    def __init__(self, api_token, max_predict=1024):
        self.url = "https://api.together.xyz/v1/completions"
        self.max_predict = max_predict
        self.api_token = api_token

    def get_request_object(self, request_tokens, stream, temp, top_p):
        return {"model": self.base_model,
                "prompt": self.tokenizer.decode(request_tokens),
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": self.max_predict,
                "stop": [self.stop_token],
                "temperature": temp,
                "top_p": top_p,
                "top_k": -1,
                "repetition_penalty": 1,
                "stream": stream,
                "min_p": 0}

    def completion(self, request_tokens, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, False, temp, top_p)
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        response = requests.post(self.url, json=request, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    async def stream_completion(self, request_tokens, callback, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, True, temp, top_p)
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        response = requests.post(self.url, json=request, stream=True, headers=headers, verify=False)
        response.raise_for_status()
        stream = sseclient.SSEClient(response).events()
        text_resp = ""
        for event in stream:
            event_data = event.data
            parsed_event = json.loads(event_data)
            if parsed_event["choices"][0]["finish_reason"] is not None:
                break
            content = parsed_event["choices"][0]["text"]
            text_resp += content
            await callback(content)
        return text_resp


class DeepInfraBackend:
    def __init__(self, api_token, max_predict=1024):
        self.url = "https://api.deepinfra.com/v1/inference/"
        self.max_predict = max_predict
        self.api_token = api_token

    def get_request_object(self, request_tokens, stream, temp, top_p):
        if top_p == 0:
            top_p = 0.1
        return {"input": self.tokenizer.decode(request_tokens),
                "max_new_tokens": self.max_predict,
                "stop": [self.stop_token],
                "temperature": temp,
                "top_p": top_p,
                "top_k": 0,
                "repetition_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "stream": stream,
                "min_p": 0}

    def completion(self, request_tokens, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, False, temp, top_p)
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        response = requests.post(self.url + self.base_model, json=request, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()["results"][0]["generated_text"]

    async def stream_completion(self, request_tokens, callback, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, True, temp, top_p)
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_token}"
        }
        response = requests.post(self.url + self.base_model, json=request, stream=True, headers=headers, verify=False)
        response.raise_for_status()
        stream = sseclient.SSEClient(response).events()
        text_resp = ""
        for event in stream:
            event_data = event.data
            parsed_event = json.loads(event_data)
            if parsed_event["details"] != None and parsed_event["details"]["finish_reason"] != None:
                break
            content = parsed_event["token"]["text"]
            text_resp += content
            await callback(content)
        return text_resp
