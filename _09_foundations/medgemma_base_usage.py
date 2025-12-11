from transformers import pipeline
from PIL import Image
import requests
import torch
from huggingface_hub import login

login(token="")  # replace with your token
print("Logged in successfully")

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    dtype=torch.bfloat16,
    device="cuda",
)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image = Image.open("/home/phd2/Scrivania/CorsoData/blastocisti/no_blasto/D2013.10.09_S0847_I141_3/D2013.10.09_S0847_I141_3_96_0_24.0h.jpg")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert embryologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "classify the following image of human embryo at day 1 and tell me if it will develop into a blastocyst. Answer with only yes or no"},
            {"type": "image", "image": image}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
