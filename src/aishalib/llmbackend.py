import requests


class LlamaCppBackend:
    def __init__(self, url, stop_token, max_predict=1024):
        self.url = url
        self.stop_token = stop_token
        self.max_predict = max_predict

    def completion(self, request_tokens, temp=0.5, top_p=0.5):
        request = {"prompt": request_tokens,
                   "stream": False,
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
        return requests.post(self.url, json=request).json()["content"]
