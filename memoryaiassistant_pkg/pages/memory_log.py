import streamlit as st
from PIL import Image
import io
import json
import openai
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
from io import BytesIO
import base64
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import markdown
from openai import AzureOpenAI
import json
import azure
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
import os
from dotenv import load_dotenv, find_dotenv
from pydub import AudioSegment
import random
from openai import OpenAI

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")



st.title('Make a Memory')
st.caption("Upload an image, then record or write a message to create a memory.")



# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
speech_config = speechsdk.SpeechConfig(subscription= AZURE_OPENAI_API_KEY, region="eastus")


text = ''

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

if "image_count" not in st.session_state:
    st.session_state.image_count = 0
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

# Start the chatbot loop
def run_chatbot(image_description):
    # Maximum interactions allowed
    max_interactions = 2

    # Initialize the conversation if it's the first image
    if st.session_state.image_count == 0:
        initial_response = initialize_conversation(image_description)
        st.write(f"Assistant: {initial_response}")
        speak_text(initial_response)
        st.audio("outputReal.mp3", start_time=0, autoplay=True)
        st.session_state.image_count += 1  # Increment image count to avoid reinitialization

    print("\nChatbot is ready! Type 'exit' to end the conversation.")

    # Continue conversation based on interaction count
    while st.session_state.interaction_count < max_interactions:
        # Get user input
        audio_bytes = audio_recorder()
        user_input = None

        if audio_bytes:
            audio_location = "../audio_file.mp3"
            os.makedirs(os.path.dirname(audio_location), exist_ok=True)
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)
            text = transcribe_audio(audio_location)
            user_input = text

            if user_input.lower() == "exit":
                st.write("Assistant: Thank you for chatting with me. Goodbye!")
                break

            # Generate assistant's response
            assistant_response = chat_with_assistant(user_input)
            st.write(f"Assistant: {assistant_response}")
            speak_text(assistant_response)
            st.audio("outputReal.mp3", start_time=0, autoplay=True)

            # Increment interaction count
            st.session_state.interaction_count += 1

        if not user_input:
            st.info("Waiting for input (audio or text)...")

        # Nuanced handling when interaction limit is reached
        if st.session_state.interaction_count >= max_interactions:
            st.write("Assistant: Thanks for the details! Feel free to keep going or say 'Stop' to move on.")
            
            # Await user input for final interaction
            audio_bytes = audio_recorder()
            if audio_bytes:
                audio_location = "../audio_file.mp3"
                os.makedirs(os.path.dirname(audio_location), exist_ok=True)
                with open(audio_location, "wb") as f:
                    f.write(audio_bytes)
                final_input = transcribe_audio(audio_location)

                if final_input.strip().lower() in ["no", "nothing else", "stop", "exit"]:
                    st.write("Assistant: I'm glad I could help. Take care and have a wonderful day!")
                else:
                    st.write("Assistant: Thank you for sharing! If you have any more details to add later, feel free to revisit.")

            # End the session
            st.session_state.interaction_count = 0  # Reset interaction count for the next session
            # break
        
        return conversation_history


    def generate_metadata(conversation_history):
        prompt = f"""
        Analyze the following conversation history and extract key information to create metadata:
        {conversation_history}
        Generate a JSON object with the following keys: 'location', 'date', 'people', 'emotions', 'activity'.
        If specific data is not mentioned, leave those fields as empty strings or empty lists.
        Only return the JSON object without any additional text.
        """
        
    client = OpenAI(api_key= OpenAI_API_KEY)

    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5
    )


def format_memory_data(image_description, conversation_history):
    memory_data = {
        "description": image_description,
        "metadata": {
            "location": "",
            "date": "",
            "people": [],
            "emotions": [],
            "activity": ""
        },
        "conversation": conversation_history
    }
    return memory_data


def save_to_seed_data(memory_data):
    try:
        with open("seed_data.json", "r+") as file:
            seed_data = json.load(file)
            seed_data["documents"].append(memory_data)
            file.seek(0)
            json.dump(seed_data, file, indent=2)
            file.truncate()
        st.success("Memory saved successfully!")
    except Exception as e:
        st.error(f"Error saving memory: {e}")

       

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
            client = OpenAI(api_key= OpenAI_API_KEY)

            response = client.chat.completions.create(
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
        return full_response
    except Exception as e:
        print("Error:", e)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"], accept_multiple_files=False)
if uploaded_file is not None:
    try:
        # image = Image.open(uploaded_file)
        with Image.open(uploaded_file) as img:
            max_size=(512, 512)
            st.image(img, caption='Uploaded Image', use_container_width=True)
            # file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
            # st.write(file_details)
            img.thumbnail(max_size)  # Resize the image
            resized_path = "resized_image.jpg"  # Temporary file to save the resized image
            img.save(resized_path, format="JPEG")

            print(f"Image resized and saved as '{resized_path}'")
            st.success("Saved File")
            saved = True

            # if saved:
            #     base64_image = encode_image_to_base64(resized_path)
            #     run_chatbot(send_image_to_openai(base64_image))
                
            if saved:
                base64_image = encode_image_to_base64(resized_path)
                conversation_history = run_chatbot(send_image_to_openai(base64_image))
                memory_data = format_memory_data(send_image_to_openai(base64_image), conversation_history)
                save_to_seed_data(memory_data)


    except Exception as e:
        pass
else:
    st.info("Please upload an image to start.")
