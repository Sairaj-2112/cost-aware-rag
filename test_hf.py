import os
from huggingface_hub import InferenceClient


token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is not set")


client = InferenceClient(
    api_key=token
)


response = client.chat.completions.create(
    model="openai/gpt-oss-120b:groq",
    messages=[
        {
            "role": "user",
            "content": "Explain autoencoders in very simple words."
        }
    ],
    max_tokens=200
)


print("\nLLM Response:")
print(response.choices[0].message.content)