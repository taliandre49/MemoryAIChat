from chromadb import Documents, EmbeddingFunction, Embeddings
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")

import sqlite3
import os

# Ensure python uses its own sqlite3 instead of a system one
print("SQLite3 version used by Python:", sqlite3.sqlite_version)

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_content: Documents) -> Embeddings:
        client = OpenAI(api_key= OpenAI_API_KEY)
        embeddings = client.embeddings.create(
            model='',
            input= input_content
            ).data[0].embedding
        return embeddings