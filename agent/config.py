import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def build_default_model():
    load_dotenv()
    api_key = os.getenv("FUNHPC_API_KEY")
    model_name = "Qwen3-Coder-30B-A3B-Instruct"
    base_url = "https://funhpc.com/v1"
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
