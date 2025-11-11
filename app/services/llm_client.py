import requests


class LLMClient:
    def __init__(self, model: str = "phi"):
        self.model = model
        self.base_url = "http://ollama:11434/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt}
        response = requests.post(self.base_url, json=payload, stream=True)

        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    text = data.split('"response":"')[1].split('"')[0]
                    output += text
        return output.strip()
