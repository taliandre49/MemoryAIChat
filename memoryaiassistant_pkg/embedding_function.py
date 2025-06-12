from chromadb import Documents, EmbeddingFunction, Embeddings
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_content: Documents) -> Embeddings:
        client = OpenAI(api_key= OpenAI_API_KEY)
        embeddings = client.embeddings.create(
            model='',
            input= input_content
            ).data[0].embedding
        return embeddings