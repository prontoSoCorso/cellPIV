import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import login
import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# ----------------------------
# Utility per parsing ore dai filename e selezione frame defined threshold hours
# ----------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
THRES_HOURS = 24.0  # soglia di default
HOUR_REGEX = re.compile(r"_(\d+(?:\.\d+)?)h(?:\.[A-Za-z]+)?$", re.IGNORECASE)
# Esempi validi:
#   ..._24.0h.jpg
#   ..._24h.png
#   ..._24.1h.jpeg

def parse_hour_from_name(name: str) -> Optional[float]:
    base = os.path.basename(name)
    m = HOUR_REGEX.search(base)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]

def first_image_at_or_after_xh(video_dir: Path, hours_threshold: float) -> Optional[Tuple[Path, float]]:
    candidates = []
    for img in list_images(video_dir):
        h = parse_hour_from_name(img.name)
        if h is not None and h >= hours_threshold:
            candidates.append((img, h))
    if not candidates:
        return None
    # Prendi la prima con ora minima >= x
    candidates.sort(key=lambda x: x[1])
    return candidates[0]

def collect_xhours_dataset(base_dir: Path, hours_threshold: float) -> pd.DataFrame:
    """
    Ritorna un DataFrame con colonne:
      ['video_id','class_name','image_path','hour','y_true']
    class_name in {'blasto','no_blasto'}, y_true in {1,0}
    """
    rows = []
    for class_name, y_true in [("blasto", 1), ("no_blasto", 0)]:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            print(f"Warning: cartella non trovata: {class_dir}")
            continue

        # Ogni sotto-cartella rappresenta un video
        for video_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
            sel = first_image_at_or_after_xh(video_dir, hours_threshold)
            if sel is None:
                # Nessuna immagine >=xh per questo video
                continue
            img_path, hour = sel
            rows.append({
                "video_id": video_dir.name,
                "class_name": class_name,
                "image_path": str(img_path),
                "hour": hour,
                "y_true": y_true
            })
    return pd.DataFrame(rows)


# ----------------------------
# MedGEMMA inference
# ----------------------------
SYSTEM_PROMPT = "You are an expert embryologist. Respond strictly with 0 or 1."
USER_TASK = (
    "Classify this day-1 embryo image (24h frame). "
    "Respond with exactly one character: 1 if it will reach blastocyst, 0 otherwise."
)


def extract_label_from_output(output) -> Optional[int]:
    """
    Estrae 0/1 dall'output del pipeline, gestendo diversi formati.
    """
    text = None
    try:
        # Caso: generated_text è lista di turni chat
        gt = output[0].get("generated_text")
        if isinstance(gt, list) and len(gt) > 0:
            # prendi il contenuto dell'ultimo turno
            text = gt[-1].get("content") if isinstance(gt[-1], dict) else str(gt[-1])
        elif isinstance(gt, str):
            text = gt
        elif "text" in output[0]:
            text = output[0]["text"]
    except Exception:
        pass

    if text is None:
        text = str(output)

    low = str(text).strip().lower()

    # Cerca cifra singola 0/1
    m = re.search(r"\b([01])\b", low)
    if m:
        return int(m.group(1))

    # Fallback yes/no
    if "yes" in low and "no" not in low:
        return 1
    if "no" in low and "yes" not in low:
        return 0

    # Ultimo tentativo: prima cifra 0 o 1 nel testo
    m2 = re.search(r"([01])", low)
    if m2:
        return int(m2.group(1))

    return None


def build_pipe(device_str: str = None):
    use_cuda = torch.cuda.is_available() and (device_str in (None, "", "cuda", "cuda:0"))
    device = 0 if use_cuda else -1
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        device=device,
        dtype=torch_dtype,
    )
    return pipe


def infer_one(pipe, image_path: str) -> Tuple[Optional[int], str]:
    """
    Ritorna (pred, raw_text) dove pred è 0/1 o None se non estraibile.
    """
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": USER_TASK},
            {"type": "image", "image": img}
        ]},
    ]
    try:
        out = pipe(messages, max_new_tokens=4, do_sample=False, temperature=0.0)
    except TypeError:
        # Alcune versioni richiedono text=messages
        out = pipe(text=messages, max_new_tokens=4, do_sample=False, temperature=0.0)

    pred = extract_label_from_output(out)
    raw = str(out)
    return pred, raw



# ----------------------------
# Confusion matrix plot
# ----------------------------
def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main(base_dir: str,
         output_dir: Optional[str] = None,
         device: Optional[str] = None):
    base_dir = Path(base_dir)
    assert base_dir.exists(), f"Base dir non trovata: {base_dir}"

    df = collect_xhours_dataset(base_dir, hours_threshold=THRES_HOURS)
    if df.empty:
        print("Nessuna immagine >= xh trovata. Controlla il naming dei file.")
        return

    print(f"Trovati {len(df)} video con frame >= {THRES_HOURS}h "
          f"(blasto={sum(df.y_true==1)}, no_blasto={sum(df.y_true==0)})")

    pipe = build_pipe(device)

    preds, raws = [], []
    for p in tqdm(df["image_path"].tolist(), desc="Inferenza MedGEMMA"):
        pred, raw = infer_one(pipe, p)
        preds.append(pred if pred is not None else -1)
        raws.append(raw)

    df["y_pred"] = preds
    df["raw_output"] = raws

    # Filtra eventuali pred non validi (-1)
    valid = df["y_pred"] >= 0
    df_valid = df[valid].copy()

    if df_valid.empty:
        print("Nessuna predizione valida ottenuta.")
        out_csv = Path(output_dir) if output_dir else (base_dir / "medgemma_predictions_24h.csv")
        df.to_csv(out_csv, index=False)
        print(f"CSV salvato in: {out_csv}")
        return

    acc = accuracy_score(df_valid["y_true"], df_valid["y_pred"])
    cm = confusion_matrix(df_valid["y_true"], df_valid["y_pred"], labels=[0, 1])

    print(f"\nAccuracy (valid only): {acc:.4f}")
    print("\nConfusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nClassification report:\n",
          classification_report(df_valid["y_true"], df_valid["y_pred"], target_names=["no_blasto", "blasto"]))

    ## Salvataggi
    # Salvataggio report
    out_report = output_dir / f"medgemma_report_{THRES_HOURS}h.txt"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        f.write(f"Accuracy (valid only): {acc:.4f}\n\n")
        f.write("Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(classification_report(df_valid["y_true"], df_valid["y_pred"], target_names=["no_blasto", "blasto"]))
    print(f"Report salvato in: {out_report}")

    # Salvataggio confusion matrix
    cm_path = output_dir / f"medgemma_confusion_matrix_{THRES_HOURS}h.png"
    save_confusion_matrix(cm, labels=["no_blasto", "blasto"], out_path=cm_path)
    print(f"Confusion matrix salvata in: {cm_path}")


if __name__ == "__main__":
    base_dir = "/home/phd2/Scrivania/CorsoData/blastocisti"  # Cambia con il path corretto
    output_dir = "/home/phd2/Scrivania/CorsoRepo/cellPIV/_05_foundations/results" # Opzionale, altrimenti usa default
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    main(base_dir, output_dir, device)