import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (to get your GEMINI_API_KEY)
load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

print("Listing available Gemini models and their capabilities:")
print("-" * 50)

for m in genai.list_models():
    # Filter for models that support text generation
    if 'generateContent' in m.supported_generation_methods:
        print(f"Name: {m.name}")
        print(f"  Description: {m.description}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print(f"  Version: {m.version}")
        print("-" * 50)
    else:
        # Optionally, print models that don't support generateContent if you want to see everything
        # print(f"Name: {m.name} (Does NOT support generateContent)")
        # print("-" * 50)
        pass

print("\nFinished listing models.")