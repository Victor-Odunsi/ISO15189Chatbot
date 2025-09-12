import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
mistral_api_key = os.getenv('MISTRALAI_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")