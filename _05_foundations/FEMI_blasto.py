from transformers import ViTMAEForPreTraining, AutoImageProcessor
from PIL import Image
import requests
from huggingface_hub import login

# Load the model
model = ViTMAEForPreTraining.from_pretrained("ihlab/FEMI")

# Load the image processor
processor = AutoImageProcessor.from_pretrained("ihlab/FEMI", use_fast=True, trust_remote_code=True)

# Load an example image
image_path = "/home/phd2/Scrivania/CorsoData/blastocisti_images_24.0h_cropped/blasto/D2013.03.09_S0695_I141_10_96_0_24.0h.jpg"
image = Image.open(image_path).convert("RGB") 

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
loss = outputs.loss
reconstructed_pixel_values = outputs.logits

p = AutoImageProcessor.from_pretrained("ihlab/FEMI", trust_remote_code=True)
print("Processor class:", p.__class__.__name__)
# stampa tutta la config utile
print(p)
# valori tipici da controllare:
print("size:", getattr(p, "size", None))
print("image_mean:", getattr(p, "image_mean", None))
print("do_resize:", getattr(p, "do_resize", None))

print("Loss:", loss.item())
print("Reconstructed pixel values shape:", reconstructed_pixel_values.shape)


import numpy as np
import matplotlib.pyplot as plt
import torch

# plot the original image, preprocessed image, and reconstructed image
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(image)        # your PIL image (224x224 RGB)
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Preprocessed")
# get preprocessed image from inputs
preprocessed = inputs["pixel_values"][0].permute(1,2,0).detach().cpu().numpy()
plt.imshow(preprocessed)  # may be in normalized space
#plt.imshow((preprocessed - preprocessed.min()) / (preprocessed.max() - preprocessed.min()))  # scale to 0..1 for display
plt.axis("off")
plt.subplot(1,3,3)


plt.title("Reconstructed")
# outputs is your model output
logits = outputs.logits  # torch.Size([1, 196, 768])
logits = logits.detach().cpu()  # move to cpu and detach

batch, n_patches, flat = logits.shape
# compute patch size and grid
channels = 3
patch_size = int((flat // channels) ** 0.5)
assert patch_size * patch_size * channels == flat, "Unexpected patch shape"
grid = int(n_patches ** 0.5)
assert grid * grid == n_patches, "Non-square patch grid"

# reshape to (batch, grid, grid, patch_h, patch_w, channels)
patches = logits.view(batch, grid, grid, patch_size, patch_size, channels).numpy()

# assemble full image(s)
recons = np.zeros((batch, grid * patch_size, grid * patch_size, channels), dtype=np.float32)
for b in range(batch):
    for i in range(grid):
        for j in range(grid):
            y0 = i * patch_size
            x0 = j * patch_size
            recons[b, y0:y0+patch_size, x0:x0+patch_size, :] = patches[b, i, j]

# At this point recons is likely in the *normalized* space used by the processor.
# Undo normalization (if processor has mean/std attributes)
if hasattr(processor, "image_mean") and hasattr(processor, "image_std"):
    mean = np.array(processor.image_mean).reshape(1,1,1,channels)  # e.g. [0.485, 0.456, 0.406]
    std  = np.array(processor.image_std).reshape(1,1,1,channels)   # e.g. [0.229, 0.224, 0.225]
    # The processor usually normalized images in range [0,1] by (img/255 - mean)/std,
    # so the model predictions are in that normalized space. Undo that:
    recons = recons * std + mean
    # Now recons should be in approx [0,1] range; scale to 0..255:
    recons = recons * 255.0
else:
    # If no mean/std available, try common defaults (works often)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,1,3)
    recons = recons * std + mean
    recons = recons * 255.0

# Clip and convert to uint8
recons = np.clip(recons, 0, 255).astype(np.uint8)

# get reconstructed image from model output
plt.imshow(recons[0]) # may be in normalized space
plt.axis("off")
plt.show()
