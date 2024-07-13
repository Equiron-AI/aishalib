import os
import threading
import json

class SimpleMemory:
    def __init__(self, file_name):
        self.file_name = file_name
        self._lock = threading.Lock()

    def get_memory(self):
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}

    def get_memory_value(self, key, default = None):
        with self._lock:
            memory = self.get_memory()
            if key in memory:
                return memory[key]
            return default

    def save_memory_value(self, key, value):
        with self._lock:
            memory = self.get_memory()
            memory[key] = value
            with open(self.file_name, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False)
