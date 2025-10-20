from transformers import ViTMAEForPreTraining, AutoImageProcessor
from PIL import Image
import requests
from huggingface_hub import login

login()

# Load the model
model = ViTMAEForPreTraining.from_pretrained("surajraj99/FEMI")

# Load the image processor
processor = AutoImageProcessor.from_pretrained("surajraj99/FEMI")

# Load an example image
url = "http://example.com/path_to_embryo_image.png"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
loss = outputs.loss
reconstructed_pixel_values = outputs.logits