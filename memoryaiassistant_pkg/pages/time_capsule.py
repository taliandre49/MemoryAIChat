import openai
from openai import OpenAI
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadbx import UUIDGenerator
import json
import streamlit as st
import random
import chromadb.api
import json
from dotenv import load_dotenv, find_dotenv
from streamlit import logger
# import sqlite3
import pysqlite3 as sqlite3

import os


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
app_loger = logger.get_logger("MemoryAIAPLog")
app_loger.log(f"sqlite version: {sqlite3.sqlite_version}")

chromadb.api.client.SharedSystemClient.clear_system_cache()


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_content: Documents) -> Embeddings:
        client = OpenAI(api_key= OpenAI_API_KEY)

        embeddings = embedding_func = client.embeddings.create(
            model='text-embedding-3-large',
            input= input_content
            ).data[0].embedding
        return embeddings

client = OpenAI(api_key= OpenAI_API_KEY)



st.title('Time Capsule')
st.session_state["messages"] = [{"role": "You are a helpful assistant designed to recreate and enhance user memories by synthesizing information from various related documents. Your goal is to evoke a sense of déjà vu by combining details, feelings, and experiences found in the provided texts. Related Documents: {documents} Analyze Documents: Understand key themes, emotions, and specific details that align with the user's memories. Combine Information: Create a cohesive narrative under 100 words that blends elements from the documents while staying true to the user's actual experiences. Invoke Emotion: Make the story feel personal and keep the style informal, light-hearted and humorous.",
                                 "content": "Welcome to Time Capsule! Let me help you remember a cherished moment."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


#Set up database
def set_up_database():
    """Set up the database."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="test", metadata={"hnsw:space": "cosine"}, embedding_function=MyEmbeddingFunction())
    return collection

def set_up_seed_data(collection):
    """Set up the seed data."""
    with open("seed_data.json") as f:
        data = json.load(f)["documents"]

    for d in data:
        description = d.get("description", "")
        metadata = d.get("metadata", {})

        # Ensure description is a string
        if not isinstance(description, str):
            print(f"Warning: non-string description found: {description} (type: {type(description)})")
            description = str(description)

        # Ensure metadata is a dict
        if not isinstance(metadata, dict):
            print(f"Warning: metadata is not a dict: {metadata}")
            metadata = {}

        add_document(collection, description, metadata)
    # data = json.loads(open("seed_data.json").read())["documents"]
    # for d in data:

    #     add_document(collection, d["description"], d["metadata"])

def add_document(collection, text, metadata):
    """Add a document to the database."""
    print(metadata)
    # Validate and transform metadata to work with Chroma input
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ", ".join(map(str, value))  # Chroma only takes in string so convert list to a comma-separated string
        elif not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Invalid metadata value for key '{key}': {value} (type {type(value)})")
    
    print(metadata)
    doc_id = UUIDGenerator(1)  # likely returns a single string
    collection.add(
    documents=[text],
    ids= doc_id,
    metadatas=[metadata]
    )

    # collection.add(
    # documents=[text],
    # ids=UUIDGenerator(1),
    # metadatas=metadata
    # )

def retrieve_document(collection, document, metadata):
    #query_where = {"$and": [{"fruit": "pineapple"}, {"climate": "tropical"}]}
    #query_where_document = {"$contains": document}
    #select_ids = collection.get(where_document=query_where_document, where=query_where, include=[])
    query_where = {"$or": [{k: v} for k, v in metadata.items()]}
    print(query_where)
    result = collection.query(
    query_texts=document,
    n_results=1,
    #where=metadata,
    #where_document={"$contains":document}
    )
    st.write("Here's a memory:")
    return result['documents']

def augmentation_process(documents):
    """Aggregate and augment the documents."""
   
    # prompt = f"""Given the following list of documents, {documents}, 
    # create a single cohesive paragraph that summarizes the key information. 
    # Ensure the paragraph is clear and accessible for visually impaired individuals and those with early Alzheimer's. 
    # Focus on simplicity and clarity, connecting the main ideas smoothly without excessive detail."""
    prompt = f"""
    You are a helpful assistant designed to recreate and enhance user memories by synthesizing information from various related documents. Your goal is to evoke a sense of déjà vu by combining details, feelings, and experiences found in the provided texts.
    Related Documents: {documents}
    Instructions:
    Analyze Documents: Understand key themes, emotions, and specific details that align with the user's memories.
    Combine Information: Create a cohesive narrative under 100 words that blends elements from the documents while staying true to the user's actual experiences.
    Invoke Emotion: Make the story feel personal and keep the style informal, light-hearted and humorous.
    """
   
    result = client.chat.completions.create(
            model="gpt-4o", # model to send to the proxy
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    st.write(result.choices[0].message.content)

def tagging(user_input): 
    prompt = f'''
        Please analyze the following detailed description of an image and generate relevant metadata in the form of a json object. The dictionary should include the following keys: 'people', 'date', 'location', 'memory'(1 word), and 'activity'. 
        If specific data or people are not mentioned, leave those fields as empty string "". Here is the image description:
        {user_input}
        Make sure the metadata accurately reflects the content and context of the image, consider aspects such as subject matter, colors, emotions, and any notable elements present, and return only the JSON object without any additional text or formatting.
        '''

    response = client.chat.completions.create(
            model="gpt-4o", # model to send to the proxy
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
    metadata = response.choices[0].message.content
    return metadata

collection = set_up_database()
set_up_seed_data(collection)
image_description = "This is an image description of Risley Hall"

# retrieval
if user_input := st.chat_input(): #user input description of image

    metadata = tagging(user_input)
    print(metadata)
    documents = retrieve_document(collection, user_input, json.loads(metadata[7:-3]))
    augmentation_process(documents)
  

# Prompt Buttons
with open("seed_data.json", "r") as f:
    seed_data = json.load(f)["documents"]   
    

def generate_prompt_buttons(data, num_buttons=5):
    prompts = []
    for item in data:
        if "location" in item["metadata"]:
            prompts.append(item["metadata"]["location"])
        if "class_year" in item["metadata"]:
            prompts.append(item["metadata"]["class_year"])
        if "name" in item["metadata"]:
            prompts.append(item["metadata"]["name"])
    
    return random.sample(prompts, min(num_buttons, len(prompts)))

prompt_buttons = generate_prompt_buttons(seed_data)


st.write("Enter a question in chat or spark a memory by exploring your time capsules with topics like:")
cols = st.columns(len(prompt_buttons))
for i, prompt in enumerate(prompt_buttons):
    if cols[i].button(prompt, key=f"prompt_button_{i}"):
        metadata = tagging(prompt)
        documents = retrieve_document(collection, prompt, json.loads(metadata[7:-3]))
        augmentation_process(documents)

