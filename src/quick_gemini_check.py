
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    raise ValueError("Gemini API key is missing.")

print("Testing model:", model)
print("Sending a minimal request...")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=120000)
)

start = time.time()

try:
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: CreditLens connection successful.",
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=30,
        ),
    )

    print("\nResponse:", response.text)
    print("Time taken:", round(time.time() - start, 2), "seconds")

except Exception as error:
    print("\nError type:", type(error).__name__)
    print("Error details:", error)
    print("Time taken:", round(time.time() - start, 2), "seconds")
