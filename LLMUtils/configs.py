import os
import google.generativeai as genai  
from langchain_google_genai import ChatGoogleGenerativeAI
# Loading the API key using lod_dotenv
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Loading the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)


class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-3-flash-preview')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = genai.GenerationConfig(
            temperature=0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )


class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.llm=ChatGoogleGenerativeAI(
            temperature=0.7,
            model="gemini-3-flash-preview", 
            google_api_key=api_key,
            top_p=1.0,
            top_k=32,
            max_output_tokens=3000)