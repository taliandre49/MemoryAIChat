import streamlit as st
from openai import AzureOpenAI
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import markdown
import json
# import azure
import azure.cognitiveservices.speech as speechsdk
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
from io import BytesIO
import requests
import copy
import torch
import base64
from gtts import gTTS
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
from pydub import AudioSegment
import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadbx import UUIDGenerator
import json
import chromadb.api
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
from io import BytesIO
import requests
import copy
import torch
import base64
from PIL import Image
import base64
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# import sqlite3
import pysqlite3 as sqlite3

from streamlit import logger
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
app_loger = logger.get_logger("MemoryAIAPLog")
app_loger.log(f"sqlite version: {sqlite3.sqlite_version}")




chromadb.api.client.SharedSystemClient.clear_system_cache()

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_content: Documents) -> Embeddings:
        client = OpenAI(api_key= OpenAI_API_KEY)
        embeddings = embedding_func = client.embeddings.create(
            model='text-embedding-3-large',
            input= input_content
            ).data[0].embedding
        return embeddings



# st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
speech_config = speechsdk.SpeechConfig(subscription=AZURE_OPENAI_API_KEY, region="eastus")



# Function to transcribe audio
def transcribe_audio(file_path):
    
    audio_input = speechsdk.audio.AudioConfig(filename= file_path)

    # Speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Perform speech recognition
    print("Recognizing speech...")
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech recognition canceled: {}".format(result.cancellation_details.reason))
        if result.cancellation_details.error_details:
            print(f"Error details: {result.cancellation_details.error_details}")



# Function to convert text to speech using gTTS
def speak_text(text, output_audio="outputReal.mp3"):
    print("Generating speech...")
    tts = gTTS(text,tld= "es")
    tts.save(output_audio)
    print(f"Audio saved to {output_audio}")
    # Playback skipped intentionally
    print("Audio file generated successfully. Playback is disabled in this environment.")



#Set up database
def set_up_database():
    """Set up the database."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="test", metadata={"hnsw:space": "cosine"}, embedding_function=MyEmbeddingFunction())
    return collection

def add_document(collection, text, metadata):
    """Add a document to the database."""
    collection.add(
    documents=[text],
    ids=UUIDGenerator(1),
    metadatas=metadata
    )

def retrieve_document(collection, document, metadata):
    query_where = {"$and": [{"fruit": "pineapple"}, {"climate": "tropical"}]}
    query_where_document = {"$contains": document}
    select_ids = collection.get(where_document=query_where_document, where=query_where, include=[])
    result = collection.query(
    query_texts=document,
    n_results=10,
    where=metadata,
    #where_document={"$contains":document}
    )
    return result['documents']

def augmentation_process(documents):
    """Aggregate and augment the documents."""
   
    prompt = f"""Given the following list of documents, {documents}, 
    create a single cohesive paragraph that summarizes the key information. 
    Ensure the paragraph is clear and accessible for visually impaired individuals and those with early Alzheimer's. 
    Focus on simplicity and clarity, connecting the main ideas smoothly without excessive detail."""
   
    result = client.chat.completions.create(
            model="gpt-4o", # model to send to the proxy
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    print(result.choices[0].message.content)

collection = set_up_database()


# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
speech_config = speechsdk.SpeechConfig(subscription= AZURE_OPENAI_API_KEY, region="eastus")


# Function to resample and load audio to 16,000 Hz
def resample_audio(file_path, target_sample_rate=16000):
    print("Resampling audio to 16,000 Hz...")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(target_sample_rate)  # Resample to target sample rate
    return audio

# Function to transcribe audio
def transcribe_audio(file_path):
    
    audio_input = speechsdk.audio.AudioConfig(filename= file_path)

    # Speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Perform speech recognition
    print("Recognizing speech...")
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech recognition canceled: {}".format(result.cancellation_details.reason))
        if result.cancellation_details.error_details:
            print(f"Error details: {result.cancellation_details.error_details}")


# Function to call speech to text question to API chatbot
def chat_completion_call(text):
    client2 = OpenAI(api_key= OpenAI_API_KEY)

    response = client2.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content

 

# Function to convert text to speech using gTTS
def speak_text(text, output_audio="outputReal.mp3"):
    print("Generating speech...")
    tts = gTTS(text,tld= "es")
    tts.save(output_audio)
    print(f"Audio saved to {output_audio}")
    # Playback skipped intentionally
    print("Audio file generated successfully. Playback is disabled in this environment.")


# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": """
        You are a Retrieval-Augmented Visual Memory Assistant called DejaVu designed to help memory-impaired individuals 
        enrich their memories and recall details about their experiences. 

        Your goal is to engage the user in a **natural, conversational manner**. Do not provide lists, 
        numbered responses, or multiple suggestions in one reply. Keep your answers **clear, concise, and engaging**.

        Start by analyzing the provided image description. Summarize it in one positive, friendly sentence, 
        then follow up with a single conversational question to encourage the user to share more details. 
        Focus on personal experiences, emotions, and meaningful moments.

        Conclude the conversation when you have gathered enough meaningful context by:
        - Thanking the user for sharing their memories.
        - Asking if there is anything else they would like to add.
        - Politely signaling the end of the memory-annotation process if the user has no additional input.

        Example:
        Input: "A group of people sitting on the grass in front of a large brick building."
        Output: "This looks like such a cheerful day with friends! Can you share where this photo was taken?"

        Additional Instructions:
        - Use empathetic and supportive language.
        - Avoid repeating questions or providing overly lengthy responses.
        - If the user struggles to recall details, gently encourage them without pressure.
        """
    }
]

# Define a function to initialize the conversation with an image description
def initialize_conversation(image_description):
    """
    Starts the conversation based on an image description by generating a friendly opening line.
    """
    try:
        # Add the image description as the initial user input
        conversation_history.append({"role": "user", "content": f"Image description: {image_description}"})
        client = OpenAI(api_key= OpenAI_API_KEY)
        # Make the API call to generate the assistant's initial response
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the desired model name
            messages=conversation_history,
            max_tokens=150,  # Adjust as needed
            temperature=0.5  # Lower temperature for more focused output
        )
        
        # Get the assistant's response
        assistant_response = response.choices[0].message.content.strip()
        
        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response
    except Exception as e:
        return f"An error occurred: {e}"

# Define a function to continue the chat with the assistant
def chat_with_assistant(user_input):
    """
    Sends user input to the OpenAI chat model and returns the assistant's response.
    """
    try:
        # Add the user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        client = OpenAI(api_key= OpenAI_API_KEY)
        
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the desired model name
            messages=conversation_history,
            max_tokens=300,  # Adjust as needed
            temperature=0.5  # Lower temperature for more consistent responses
        )
        
        # Get the assistant's response
        assistant_response = response.choices[0].message.content.strip()
        
        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response
    except Exception as e:
        return f"An error occurred: {e}"

# Start the chatbot loop
def run_chatbot(image_description):
    # Initialize the conversation with the image description
    initial_response = initialize_conversation(image_description)
    st.write(f"Assistant: {initial_response}")
    
    interaction_count = 0
    max_interactions = 5  # Define the threshold for interactions
    
    print("\nChatbot is ready! Type 'exit' to end the conversation.")
    while True:
        # Increment interaction count
        interaction_count += 1
        
        # Check if the conversation should conclude
        if interaction_count > max_interactions:
            st.write("Assistant: Thank you so much for sharing your memories with me. Is there anything else you'd like to add?")
            user_input = input("You: ")
            
            if user_input.strip().lower() in ["no", "nothing else", "exit"]:
                st.write("Assistant: I'm glad I could help. Take care and have a wonderful day!")
                break
            else:
                st.write("Assistant: Thank you for sharing! If you have any more details to add later, feel free to revisit.")
                break
        
        # Get user input
        user_input = input("You: ")
        
        # Exit the loop if the user types 'exit'
        if user_input.lower() == "exit":
            st.write("Assistant: Thank you for chatting with me. Goodbye!")
            break
        
        # Get the assistant's response
        assistant_response = chat_with_assistant(user_input)
        
        # Print the assistant's response
        print(f"Assistant: {assistant_response}")

    # Example usage: Run the chatbot with an image description
    example_image_description = """
    The image depicts a modern building with a striking design, set against a dramatic sky.

    **Main Subject and Setting**
    The building's main subject is its unique architecture, featuring a large glass facade and a cantilevered structure that extends over the entrance. The building is situated in a public space, with a sidewalk leading up to it and a grassy area to the right.

    **Time of Day and Weather Conditions**
    The time of day appears to be dusk, as the sky is overcast and dark clouds are visible. The lighting inside the building suggests that it is illuminated from within, possibly due to the setting sun.

    **Colors and Visual Elements**
    The building's exterior is primarily composed of glass and metal, with a silver-colored metal cladding that gives it a sleek and contemporary look. The cantilevered structure is supported by thin metal beams, which add to the building's futuristic aesthetic. The sky above is a deep grey, with darker clouds gathering in the distance.

    **Emotions and Actions**
    There are no visible emotions or actions in the image, as it appears to be a static photograph of the building.

    **Significant Objects or Landmarks**
    The building itself is the most significant object in the image, with its unique design and architecture making it a notable landmark. In the background, a few trees and other buildings can be seen, but they are not as prominent as the main subject.

    **Text and Signs**
    The building's name, "Bates Hall," is visible on the front of the structure, just below the cantilevered section. There are no other signs or text visible in the image.

    **Clothing and Attire**
    There are no people visible in the image, so there is no clothing or attire to describe.

    **Surrounding Environment and Background**
    The surrounding environment is a public space, with a sidewalk leading up to the building and a grassy area to the right. In the background, a few trees and other buildings can be seen, but they are not as prominent as the main subject.

    **Overall Mood and Atmosphere**
    """



# Step 2: Encode the resized image file as Base64
def encode_image_to_base64(image_path):
    """
    Encode the image file as a Base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        exit()

# Step 3: Send the Base64 image to the OpenAI endpoint
def send_image_to_openai(base64_image):
    """
    Send the Base64-encoded image to the OpenAI endpoint.
    """
    try:
        full_response = ""
        prompt = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": """Describe the image in detail, focusing on the following aspects: 
                 1. Main subject(s) or people in the photo, 
                 2. Location or setting, 
                 3. Time of day and weather conditions, 
                 4. Colors and visual elements that stand out, 
                 5. Any visible emotions or actions, 
                 6. Significant objects or landmarks, 
                 7. Any text or signs visible in the image, 
                 8. Clothing or attire of people (if applicable), 
                 9. Surrounding environment or background details, 
                 10. Overall mood or atmosphere of the scene, Please provide a comprehensive description that captures both the visual elements and the context of the image, as if explaining it to someone who cannot see it. 
                 Include any details that might be particularly memorable or emotionally significant."""}
            ]}
        ]
        while True:
            response = client.chat.completions.create(
                # was meta.llmama
                model="llama-3.2-11b-vision-instruct",  # Replace with your model name
                messages=prompt,
                max_tokens=500,  # Adjust as needed
                temperature=0.7
            )
            response_content = response.choices[0].message.content
#            print(response_content)
            full_response += response_content
            
            # Check if the response ends abruptly and prompt continuation
            if not response_content.strip().endswith("..."):
                break  # No continuation needed
            prompt = [{"role": "assistant", "content": "Continue describing the image."}]
        
        print("Full Response:", full_response)
    except Exception as e:
        print("Error:", e)



#add_document(collection, "Pineapples are tropical fruits known for their sweet and tangy flavor. They are rich in vitamin C and manganese, and they provide dietary fiber. Pineapples have a spiky outer skin and a juicy, yellow interior. They are often used in desserts, smoothies, and savory dishes.", {"fruit": "pineapple", "climate": "tropical"})
#add_document(collection, "Oranges are citrus fruits famous for their bright color and refreshing taste. They are an excellent source of vitamin C, folate, and antioxidants. Oranges have a thick, bumpy peel and are juicy on the inside. They are often consumed as a snack, in juices, or as a flavoring in various dishes.", {"fruit": "orange", "climate": "temperate"})

# documents = retrieve_document(collection, "I want a pineapple", {"climate": "tropical"})
# augmentation_process(documents)
    
st.title('Dejavu 🌄')


if user_input := st.chat_input():

    tagging = True

    if tagging:
        client = OpenAI(api_key= OpenAI_API_KEY)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]



        prompt = f'''
        Please analyze the following detailed description of an image and generate relevant metadata in the form of a json object. The dictionary should include the following keys: 'description' (under 100 words), 'people', 'date', 'location', 'emotion', and 'category'. 
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

        st.write_stream(response)
        metadata = response.choices[0].message.content
        print(metadata[7:-3])
        add_document(collection, user_input, json.loads(metadata[7:-3]))
        print(collection.get(offset=0, limit=10))



        resized_image_path = "../resized_image.jpg"  # Resize the image
        base64_image = encode_image_to_base64(resized_image_path)  # Encode resized image
        responseImage = send_image_to_openai(base64_image)  # Send the imag
        print(responseImage)
        st.write(responseImage)
            
