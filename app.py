from typing import Literal
from ollama import Client
from llama_index.llms.ollama import Ollama
import os

# from langchain_experimental.llms.ollama_functions import OllamaFunctions
from fastapi import FastAPI

app = FastAPI()

BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.222.236:11434")


@app.get("/ollama")
def call_ollama(
    text: str,
    model: str = "llama2",
    request_timeout: float = 60.0,
    mode: Literal["llamaindex", "ollama"] = "ollama",
    base_url: str = BASE_URL,
) -> dict:
    if mode == "llamaindex":
        llm = Ollama(base_url=base_url, model=model, request_timeout=request_timeout)
        response = llm.complete(text)
    elif mode == "ollama":
        client = Client(host=base_url, timeout=request_timeout)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8777, reload=True)
