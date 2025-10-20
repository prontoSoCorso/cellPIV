import torch
import os

# Configurazioni
BASE_PATH = "/home/phd2/Scrivania/CorsoRepo/cellPIV/_06_ViT"
DATA_PATH = "/home/phd2/Scrivania/CorsoData/blastocisti_images_24.0h"
RESULTS_DIR = os.path.join(BASE_PATH, "results")

# Parametri del modello
IMG_SIZE = 256
PATCH_SIZE = 16
IN_CHANS = 1
NUM_CLASSES = 2
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0
QKV_BIAS = True
DROP_RATE = 0.0

# Parametri di training
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")